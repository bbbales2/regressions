from contextlib import contextmanager
import jax
import math
import numpy
import pandas
import scipy
import threading
from typing import Callable, Set, Tuple, Dict


class Potential:
    value_and_grad_negative_log_density: Callable[[numpy.array], Tuple[float, numpy.array]]
    """Gradient of negative log density"""

    active_threads: Set[int]
    """Set of threads this potential parallizes over"""

    waiting_threads: Set[int]
    """Set of threads that are actively waiting to run"""

    context_lock: threading.Lock
    """Use this lock to modify variables shared between threads"""

    gradient_computed_event: threading.Event
    """This event signals that the gradient has been computed"""

    arguments: Dict[int, numpy.array]
    """Dictionary of thread id to argument"""

    results: Dict[int, Tuple[float, numpy.array]]
    """Dictionary of thread id to results"""

    def __init__(self, negative_log_density):
        self.value_and_grad_negative_log_density = jax.jit(jax.value_and_grad(negative_log_density))
        self.active_threads = set()
        self.waiting_threads = set()
        self.context_lock = threading.Lock()
        self.gradient_computed_event = threading.Event()
        self.arguments = {}
        self.results = {}

    def _value_and_grad(self, q0: numpy.array) -> numpy.array:
        active_thread = threading.get_ident()

        if active_thread not in self.active_threads:
            raise Exception("Called gradient without activating thread")
        
        self.arguments[active_thread] = q0

        while True:
            with self.context_lock:
                self.waiting_threads.add(active_thread)

                if len(self.active_threads - self.waiting_threads) == 0:
                    for thread in self.active_threads:
                        device_U, device_gradient = self.value_and_grad_negative_log_density(self.arguments[thread])
                        self.results[thread] = float(device_U), numpy.array(device_gradient)
                    self.gradient_computed_event.set()
                    self.gradient_computed_event.clear()
                    break
            
            # There's a race condition here that hopefully doesn't lead to errors. If a non-running thread doesn't
            # get to wait before the running thread sets and clears the event, it will wait for the next round to run
            gradients_computed = gradient_computed_event.wait(timeout = 0.1)
            if gradients_computed:
                break

        return self.results[active_thread]

    @contextmanager
    def activate_thread(self) -> Callable[[numpy.array], Tuple[float, numpy.array]]:
        active_thread = threading.get_ident()
        with self.context_lock:
            self.active_threads.add(active_thread)
        
        try:
            yield self._value_and_grad
        finally:
            self.active_threads.remove(active_thread)


def uturn(q_plus: numpy.array, q_minus: numpy.array, p_plus: numpy.array, p_minus: numpy.array) -> bool:
    no_uturn_forward = numpy.dot(p_plus, q_plus - q_minus) > 0
    no_uturn_backward = numpy.dot(-p_minus, q_minus - q_plus) > 0

    return not (no_uturn_forward and no_uturn_backward)


def one_sample_nuts(
    q0: numpy.array,
    h: float,
    potential: Potential,
    M_inv: numpy.array,
    rng: numpy.random.Generator,
    max_treedepth: int = 10,
    debug: bool = False
):
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
    L_inv = numpy.linalg.cholesky(M_inv)
    size = L_inv.shape[0]

    z = rng.normal(size=size)

    p0 = numpy.linalg.solve(L_inv.transpose(), z)

    with potential.activate_thread() as value_and_grad:
        U0, grad0 = value_and_grad(q0)
        H0 = 0.5 * numpy.dot(p0, numpy.dot(M_inv, p0)) + U0

    # directions is a length max_treedepth vector that maps each treedepth
    #  to an integration direction (left or right)
    directions = rng.integers(low = 0, high = 2, size=max_treedepth + 1) * 2 - 1
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
    accept_stat = None
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
        with potential.activate_thread() as value_and_grad:
            dt = h * directions[depth]

            i_prev = depth_steps[0] - directions[depth]

            q = qs[i_prev, :]
            p = ps[i_prev, :]

            U, grad = value_and_grad(q)
            H_prev = 0.5 * numpy.dot(p, numpy.dot(M_inv, p)) + U

            for i in depth_steps:
                # leapfrog step
                p_half = p - (dt / 2) * grad
                q = q + dt * numpy.dot(M_inv, p_half)
                U, grad = value_and_grad(q)
                p = p_half - (dt / 2) * grad

                H = 0.5 * numpy.dot(p, numpy.dot(M_inv, p)) + U

                qs[i] = q
                ps[i] = p
                log_pi[i] = -H

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
            for uturn_depth in range(tree_depth, 0, -1):
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

        # Accept statistic from ordinary HMC
        p_step_accept = numpy.minimum(1.0, numpy.exp(H_prev + log_pi[depth_steps]))

        # Average the acceptance statistic for each step in this branch of the tree
        p_tree_accept = numpy.mean(p_step_accept)

        if accept_stat is None:
            accept_stat = p_tree_accept
        else:
            accept_stat = (accept_stat * (len(depth_steps) - 1) + p_tree_accept * len(depth_steps)) / (2 * len(depth_steps) - 1)

        # Merging the two trees requires one more uturn check from the overall left to right states
        uturn_detected = uturn_detected_new_tree | uturn(qs[i_right, :], qs[i_left, :], ps[i_right, :], ps[i_left, :])

        if uturn_detected:
            break

        # log of the sum of pi (A.3.1 of https://arxiv.org/abs/1701.02434) of the new subtree
        log_sum_pi_new = scipy.special.logsumexp(log_pi[depth_steps])

        # sample from the new subtree according to the equation in A.2.1 in https://arxiv.org/abs/1701.02434
        #  (near the end of that section)
        if len(depth_steps) > 1:
            i_new = rng.choice(depth_steps, p = scipy.special.softmax(log_pi[depth_steps]))
        else:
            i_new = depth_steps[0]

        # Pick between the samples generated from the new and old subtrees using the biased progressive sampling in
        #  A.3.2 of https://arxiv.org/abs/1701.02434
        p_new = min(1, numpy.exp(log_sum_pi_new - log_sum_pi_old))
        if rng.random() < p_new:
            i_old = i_new

        # Update log of sum of pi of overall tree
        log_sum_pi_old = numpy.logaddexp(log_sum_pi_old, log_sum_pi_new)

    # Get the final sample
    q = qs[i_old, :]

    steps_taken = numpy.nonzero(depth_map <= depth)[0]

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
        ).iloc[steps_taken]

        return {
            "q": q,  # new sample
            "i": i_old - min(numpy.nonzero(depth_map <= depth)[0]) + 1,  # q is the ith row of trajectory
            "q0": q0,
            "h": h,
            "max_treedepth": max_treedepth,
            "trajectory": trajectory_df,  # tibble containing details of trajectory
            "directions": directions,  # direction integrated in each subtree
            "accept_stat": accept_stat
        }
    else:
        return {"q": q, "accept_stat": accept_stat}
