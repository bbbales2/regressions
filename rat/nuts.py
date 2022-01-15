import jax
import math
import numpy
import pandas
import scipy
from typing import Callable


class Hamiltonian:
    negative_log_density: Callable[[numpy.array], float]
    """Negative log density"""

    grad_negative_log_density: Callable[[numpy.array], float]
    """Gradient of negative log density"""

    M_inv: numpy.array
    """Inverse metric"""

    L_inv: numpy.array
    """Cholesky factor of inverse metric"""

    size: int
    """Dimension of hamiltonian"""

    def __init__(self, negative_log_density, M_inv: numpy.array):
        self.negative_log_density = jax.jit(negative_log_density)
        self.M_inv = M_inv
        self.L_inv = numpy.linalg.cholesky(M_inv)
        self.size = self.L_inv.shape[0]
        self.grad_negative_log_density = jax.jit(jax.grad(negative_log_density))

    def sample_momentum(self) -> numpy.array:
        z = numpy.random.normal(size=self.size)

        return numpy.linalg.solve(self.L_inv.transpose(), z)

    def H(self, q0: numpy.array, p0: numpy.array) -> float:
        return 0.5 * numpy.dot(p0, numpy.dot(self.M_inv, p0)) + self.negative_log_density(q0)

    def gradU(self, q0: numpy.array) -> numpy.array:
        return numpy.array(self.grad_negative_log_density(q0))


def leapfrog_step(q0: numpy.array, p0: numpy.array, h: float, ham: Hamiltonian):
    M_inv = ham.M_inv

    # leapfrog step
    p_half = p0 - (h / 2) * ham.gradU(q0)
    q1 = q0 + h * numpy.dot(M_inv, p_half)
    p1 = p_half - (h / 2) * ham.gradU(q1)

    # return new state
    return {"q": q1, "p": p1}


def uturn(q_plus: numpy.array, q_minus: numpy.array, p_plus: numpy.array, p_minus: numpy.array) -> bool:
    no_uturn_forward = numpy.dot(p_plus, q_plus - q_minus) > 0
    no_uturn_backward = numpy.dot(-p_minus, q_minus - q_plus) > 0

    return not (no_uturn_forward and no_uturn_backward)


def one_sample_nuts(q0: numpy.array, h: float, ham: Hamiltonian, seed, max_treedepth: int = 10, debug: bool = False):
    """
    Generate a draw using Multinomial NUTS (https://arxiv.org/abs/1701.02434 with the original
    Uturn criteria https://arxiv.org/abs/1111.4246)

    * q0 - current draw
    * h - leapfrog stepsize
    * ham - system hamiltonian (generated from hamwrapper library)
    * max_treedepth - max treedepth
    * debug - if true, then return a bunch of debugging info, otherwise return only the next draw
    * seed - use the same seed to get the same behavior -- if NULL, make one up. Whatever is used is returned if debug == TRUE

    If debug is false, return list containing single element (q) containing next draw. If debug is true, a lot of stuff
    """

    p0 = ham.sample_momentum()

    H0 = ham.H(q0, p0)

    # directions is a length max_treedepth vector that maps each treedepth
    #  to an integration direction (left or right)
    directions = numpy.random.randint(0, 2, size=max_treedepth + 1) * 2 - 1
    # depth_map will be a vector of length 2^max_treedepth that maps each of
    #  the possibly 2^max_treedepth points to a treedepth
    depth_map = [0]
    for depth in range(1, max_treedepth + 1):
        direction = directions[depth]

        if direction < 0:
            depth_map = ([depth] * 2 ** max(0, depth - 1)) + depth_map
        else:
            depth_map = depth_map + ([depth] * 2 ** max(0, depth - 1))
    depth_map = numpy.array(depth_map)

    # Steps is a dict that maps treedepth to which leapfrog steps were
    #  computed in that treedepth (kinda the opposite of depth_map)
    steps = {}
    # qs stores our positions
    qs = numpy.zeros((2 ** max_treedepth, len(q0)))
    # ps stores our momentums
    ps = numpy.zeros((2 ** max_treedepth, len(p0)))
    # log_pi defined in section A.2.3 of https://arxiv.org/abs/1701.02434
    log_pi = numpy.zeros(2 ** max_treedepth)
    # index of initial state
    i_first = numpy.nonzero(depth_map == 0)[0][0]

    qs[i_first, :] = q0
    ps[i_first, :] = p0
    log_pi[i_first] = -H0
    # i_left and i_right are indices that track the leftmost and rightmost
    #  states of the integrated trajectory
    i_left = i_first
    i_right = i_first
    # log_sum_pi_old is the log of the sum of the pis (of log_pi) for the
    #  tree processed so far
    log_sum_pi_old = log_pi[i_first]
    # i_old will be the sample chosen (the sample from T(z' | told) in section
    #  A.3.1 of https://arxiv.org/abs/1701.02434)
    i_old = i_first
    # We need to know whether we terminated the trajectory cause of a uturn or we
    #  hit the max trajectory length
    uturn_detected = False
    # For trees of increasing treedepth
    for depth in range(1, max_treedepth + 1):
        # Figure out what leapfrog steps we need to compute. If integrating in the
        #  positive direction update the index that points at the right side of the
        #  trajectory. If integrating in the negative direction update the index pointing
        #  to the left side of the trajectory.
        if directions[depth] < 0:
            depth_steps = numpy.flip(numpy.nonzero(depth_map == depth)[0])
            i_left = depth_steps[-1]
        else:
            depth_steps = numpy.nonzero(depth_map == depth)[0]
            i_right = depth_steps[-1]

        steps[depth] = depth_steps

        # Actually do the integrationg
        for i in depth_steps:
            if directions[depth] < 0:
                i_prev = i + 1
            else:
                i_prev = i - 1

            # This doesn't take advantage of the leapfroggy-nature of leapfrog.
            z = leapfrog_step(qs[i_prev, :], ps[i_prev, :], h * directions[depth], ham)
            qs[
                i,
            ] = z["q"]
            ps[
                i,
            ] = z["p"]
            log_pi[i] = -ham.H(z["q"], z["p"])

        # What we're doing here is generating a trajectory composed of a number of leapfrog states.
        # We apply a set of comparisons on this trajectory that can be organized to look like a binary tree.
        # Sometimes I say trajectory instead of tree. Each piece of the tree in the comparison corresponds
        #  to a subset of the trajectory. When I say trajectory I'm referring to a piece of the trajectory that
        #  also corresponds to some sorta subtree in the comparisons.
        #
        # This is probably confusing but what I want to communicate is trajectory and tree are very related
        #  but maybe technically not the same thing.
        #
        # Detect U-turns in newly integrated subtree
        uturn_detected_new_tree = False
        tree_depth = round(numpy.log2(len(depth_steps)))
        if tree_depth > 0:
            # Start at root of new subtree and work down to leaves
            for uturn_depth in range(tree_depth, -1, -1):
                # The root of the comparison tree compares the leftmost to the rightmost states of the new
                #  part of the trajectory.
                #  The next level down in the tree of comparisons cuts that trajectory in two and compares
                #  the leftmost and rightmost elements of those smaller trajectories.
                #  Etc. Etc.
                div_length = 2 ** (uturn_depth)

                # Starts are relative indices pointing to the leftmost state of each comparison to be done
                starts = numpy.arange(0, len(depth_steps), div_length)  # seq(1, length(depth_steps), div_length)
                # Ends are relative indices pointing to the rightmost state for each comparison to be done
                ends = starts + div_length - 1

                # Starts and ends are relative because they point to the ith leapfrog step in a sub-trajectory
                #  of size 2^tree_depth which needs to be mapped to the global index of qs
                #  (which is defined in the range 1:2^max_treedepth)
                #
                # The sort is necessary because depth_steps is sorted in order of leapfrog steps taken
                #  which might be backwards in time (so decreasing instead of increasing)
                #
                # This sort is important because we need to keep track of what is left and what is right
                #  in the trajectory so that we do the right comparisons
                sorted_depth_steps = sorted(depth_steps)

                for j in range(len(starts)):
                    is_uturn = uturn(
                        qs[sorted_depth_steps[ends[j]], :],
                        qs[sorted_depth_steps[starts[j]], :],
                        ps[sorted_depth_steps[ends[j]], :],
                        ps[sorted_depth_steps[starts[j]], :],
                    )

                    if is_uturn:
                        uturn_detected_new_tree = True
                        break

                if uturn_detected_new_tree:
                    break

        # Merging the two trees requires one more uturn check from the overall left to right states
        uturn_detected = uturn_detected_new_tree | uturn(qs[i_right, :], qs[i_left, :], ps[i_right, :], ps[i_left, :])

        if uturn_detected:
            break

        # log of the sum of pi (A.3.1 of https://arxiv.org/abs/1701.02434) of the new subtree
        log_sum_pi_new = scipy.special.logsumexp(log_pi[depth_steps])

        # sample from the new subtree according to the equation in A.2.1 in https://arxiv.org/abs/1701.02434
        #  (near the end of that section)
        if len(depth_steps) > 1:
            i_new = sample(depth_steps, 1, prob=softmax(log_pi[depth_steps]))
        else:
            i_new = depth_steps

        # Pick between the samples generated from the new and old subtrees using the biased progressive sampling in
        #  A.3.2 of https://arxiv.org/abs/1701.02434
        p_new = min(1, numpy.exp(log_sum_pi_new - log_sum_pi_old))
        if numpy.random.random_sample() < p_new:
            i_old = i_new

        # Update log of sum of pi of overall tree
        log_sum_pi_old = numpy.logaddexp(log_sum_pi_old, log_sum_pi_new)

    # Get the final sample
    q = qs[i_old, :]

    if debug:
        q_columns = [f"q{i}" for i in range(len(q0))]
        p_columns = [f"p{i}" for i in range(len(p0))]

        qs_df = pandas.DataFrame(qs, columns=q_columns)
        ps_df = pandas.DataFrame(ps, columns=p_columns)

        # For a system with N parameters, this tibble will have
        #  2N + 2 columns. The first column (log_pi) stores log of pi (A.2.3 of https://arxiv.org/abs/1701.02434)
        #  The second column (depth_map) stores at what treedepth that state was added to the trajectory
        #  The next N columns are the positions (all starting with q)
        #  The next N columns are the momentums (all starting with p)
        trajectory_df = (
            pandas.concat([pandas.DataFrame({"log_pi": log_pi, "depth_map": depth_map}), qs_df, ps_df], axis=1).assign(
                valid=lambda df: True if uturn_detected else df["depth_map"] < depth
            )
        ).iloc[numpy.nonzero(depth_map <= depth)]

        return {
            "q": q,  # new sample
            "i": i_old - min(which(depth_map <= depth)) + 1,  # q is the ith row of trajectory
            "q0": q0,
            "h": h,
            "max_treedepth": max_treedepth,
            "seed": seed,  # seed used by random number generator, for reproducibility
            "trajectory": trajectory,  # tibble containing details of trajectory
            "directions": directions,  # direction integrated in each subtree
        }
    else:
        return {"q": q}
