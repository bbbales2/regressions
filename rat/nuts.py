from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import jax
import math
import numpy
import pandas
import scipy
import threading
import time
from typing import Callable, Set, Tuple, Dict
from tqdm import tqdm


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

    metrics: Dict[int, Tuple[int, float]]
    """Performance metrics"""

    maximum_threads: int
    """Maximum number of client threads"""

    local_identity: Dict[int, int]
    """Remap thread identities to integers"""

    def __init__(self, negative_log_density, maximum_threads, size):
        self.value_and_grad_negative_log_density = jax.jit(jax.vmap(jax.value_and_grad(negative_log_density)))
        self.active_threads = numpy.zeros(maximum_threads, dtype=bool)
        self.waiting_threads = numpy.zeros(maximum_threads, dtype=bool)
        self.context_lock = threading.Lock()
        self.gradient_computed_condition = threading.Condition(self.context_lock)

        self.arguments = numpy.zeros((maximum_threads, size), dtype=numpy.float32)
        self.potential_results = numpy.zeros(maximum_threads, dtype=numpy.float32)
        self.gradient_results = numpy.zeros((maximum_threads, size), dtype=numpy.float32)

        self.metrics = defaultdict(lambda: [0, 0.0])
        self.maximum_threads = maximum_threads
        self.local_identity = {}

    def _single_value_and_grad(self, q0: numpy.array) -> Tuple[float, numpy.array]:
        device_U, device_gradient = self.value_and_grad_negative_log_density(q0)
        return float(device_U), numpy.array(device_gradient)

    def _get_ident(self):
        threading_identity = threading.get_ident()

        try:
            return self.local_identity[threading_identity]
        except:
            if len(self.local_identity) == self.maximum_threads:
                raise Exception("This potential is already at it's maximum number of threads")

            new_local_identity = len(self.local_identity)

            self.local_identity[threading_identity] = new_local_identity

            return new_local_identity

    def _value_and_grad(self, q0: numpy.array) -> Tuple[float, numpy.array]:
        active_thread = self._get_ident()

        if not self.active_threads[active_thread]:
            raise Exception("Called gradient without activating thread")

        self.arguments[active_thread, :] = q0

        while True:
            with self.context_lock:
                self.waiting_threads[active_thread] = True

                number_waiting_threads = self.waiting_threads.sum()

                if (self.active_threads.sum() - number_waiting_threads) == 0:
                    start = time.time()

                    device_U, device_gradients = self.value_and_grad_negative_log_density(self.arguments[self.waiting_threads, :])

                    self.potential_results[self.waiting_threads] = device_U
                    self.gradient_results[self.waiting_threads, :] = device_gradients

                    self.gradient_computed_condition.notify_all()

                    metrics = self.metrics[number_waiting_threads]

                    metrics[0] += number_waiting_threads
                    metrics[1] = 0.1 * (time.time() - start) / number_waiting_threads + 0.9 * metrics[1]

                    self.waiting_threads[active_thread] = False

                    break

            # There's a race condition here that hopefully doesn't lead to errors. If a non-running thread doesn't
            # get to wait before the running thread sets and clears the event, it will wait for the next round to run
            with self.context_lock:
                gradients_computed = self.gradient_computed_condition.wait(timeout=0.01)

            if gradients_computed:
                with self.context_lock:
                    self.waiting_threads[active_thread] = False
                break

        return self.potential_results[active_thread], self.gradient_results[active_thread, :]

    @contextmanager
    def activate_thread(self) -> Callable[[numpy.array], Tuple[float, numpy.array]]:
        active_thread = self._get_ident()
        with self.context_lock:
            self.active_threads[active_thread] = True

        try:
            yield self._value_and_grad
        finally:
            with self.context_lock:
                self.active_threads[active_thread] = False


def uturn(q_plus: numpy.array, q_minus: numpy.array, p_plus: numpy.array, p_minus: numpy.array) -> bool:
    # no_uturn_forward = numpy.dot(p_plus, q_plus - q_minus) > 0
    # no_uturn_backward = numpy.dot(-p_minus, q_minus - q_plus) > 0

    # return not(no_uturn_forward or no_uturn_backward)
    uturn_forward = numpy.dot(p_plus, q_plus - q_minus) <= 0
    uturn_backward = numpy.dot(-p_minus, q_minus - q_plus) <= 0

    return uturn_forward and uturn_backward


def one_draw(
    potential: Potential,
    rng: numpy.random.Generator,
    current_draw: numpy.array,
    stepsize: float,
    diagonal_inverse_metric: numpy.array,
    max_treedepth: int = 10,
    debug: bool = False,
):
    """
    Generate a draw using Multinomial NUTS (https://arxiv.org/abs/1701.02434 with the original
    Uturn criteria https://arxiv.org/abs/1111.4246)

    * potential - Potential object representing the unnormalized negative log density of distribution to sample
    * rng - Instance of numpy.random.Generator for generating rngs
    * current_draw - current draw
    * stepsize - leapfrog stepsize
    * diagonal_inverse_metric - Diagonal of the inverse metric (usually diagonal of covariance of draws)
    * max_treedepth - max treedepth
    * debug - if true, then return a bunch of debugging info, otherwise return only the next draw

    If debug is false, return dictionary containing containing next draw and acceptance probability statistic.
    If debug is true, a lot of stuff
    """
    q0 = current_draw

    # TODO: Clean up code with changed variable names
    h = stepsize
    diag_M_inv = diagonal_inverse_metric

    # L_inv = numpy.linalg.cholesky(M_inv)
    diag_L_inv = numpy.sqrt(diag_M_inv)
    size = diag_L_inv.shape[0]

    z = rng.normal(0.0, 1.0, size)

    # p0 = numpy.linalg.solve(L_inv.transpose(), z)
    p0 = z / diag_L_inv

    def kinetic_energy(p):
        # return 0.5 * numpy.dot(p, numpy.dot(M_inv, p))
        return 0.5 * numpy.dot(p, p * diag_M_inv)

    with potential.activate_thread() as value_and_grad:
        U0, grad0 = value_and_grad(q0)
        H0 = kinetic_energy(p0) + U0

        # directions is a length max_treedepth vector that maps each treedepth
        #  to an integration direction (left or right)
        directions = rng.integers(low=0, high=2, size=max_treedepth + 1) * 2 - 1

        choice_draws = rng.random(size = max_treedepth)
        pick_draws = rng.random(size = max_treedepth)

        # depth_map will be a vector of length 2^max_treedepth that maps each of
        #  the possibly 2^max_treedepth points to a treedepth
        depth_map = numpy.zeros(1)
        for depth in range(1, max_treedepth + 1):
            direction = directions[depth]

            new_section = numpy.repeat(depth, 2 ** max(0, depth - 1))
            if direction < 0:
                # depth_map = ([depth] * 2 ** max(0, depth - 1)) + depth_map
                depth_map = numpy.hstack((new_section, depth_map))
            else:
                # depth_map = depth_map + ([depth] * 2 ** max(0, depth - 1))
                depth_map = numpy.hstack((depth_map, new_section))

        depth_map = depth_map.astype(int)

        # Steps is a dict that maps treedepth to which leapfrog steps were
        #  computed in that treedepth (kinda the opposite of depth_map)
        steps = {}
        # qs stores our positions
        qs = numpy.zeros((2 ** max_treedepth, size))
        # ps stores our momentums
        ps = numpy.zeros((2 ** max_treedepth, size))
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

            checks = []
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

                    for start, end in zip(starts, ends):
                        checks.append((start, end))

            # Sort into order that checks happen
            checks.sort(key = lambda x : x[1])

            # Initialize u-turn check variable
            uturn_detected_new_tree = False

            # Actually do the integrationg
            if True:
                # with potential.activate_thread() as value_and_grad:
                dt = h * directions[depth]

                i_prev = depth_steps[0] - directions[depth]

                q = qs[i_prev, :].copy()
                p = ps[i_prev, :].copy()

                U, grad = value_and_grad(q)

                # Initialize pointer into checks list
                check_i = 0

                # These are a bunch of temporaries to minimize numpy
                # allocations during integration
                p_half = numpy.zeros(size)
                half_dt = dt / 2
                half_dt_grad = half_dt * grad
                dt_diag_M_inv = dt * diag_M_inv
                for leapfrogs_taken, i in enumerate(depth_steps, start=1):
                    # leapfrog step
                    # p_half = p - (dt / 2) * grad
                    p -= half_dt_grad  # p here is actually p_half
                    # q = q + dt * numpy.dot(M_inv, p_half)
                    # q = q + dt * diag_M_inv * p_half
                    q += dt_diag_M_inv * p
                    U, grad = value_and_grad(q)
                    half_dt_grad = half_dt * grad
                    # p = p_half - (dt / 2) * grad
                    p -= half_dt_grad  # p here is indeed p

                    K = kinetic_energy(p)
                    H = K + U

                    qs[i] = q
                    ps[i] = p
                    log_pi[i] = -H

                    while check_i < len(checks) and checks[check_i][1] < leapfrogs_taken:
                        start, end = checks[check_i]

                        start_i = sorted_depth_steps[start]
                        end_i = sorted_depth_steps[end]

                        is_uturn = uturn(
                            qs[end_i, :],
                            qs[start_i, :],
                            ps[end_i, :],
                            ps[start_i, :],
                        )

                        if is_uturn:
                            uturn_detected_new_tree = True
                            break

                        check_i += 1

                    if uturn_detected_new_tree:
                        break

            # Merging the two trees requires one more uturn check from the overall left to right states
            uturn_detected = uturn_detected_new_tree | uturn(qs[i_right, :], qs[i_left, :], ps[i_right, :], ps[i_left, :])

            # Accept statistic from ordinary HMC
            # Only compute the accept probability for the steps done
            log_pi_steps = log_pi[depth_steps[0:leapfrogs_taken]]
            energy_loss = H0 + log_pi_steps
            p_step_accept = numpy.minimum(1.0, numpy.exp(energy_loss))

            # Average the acceptance statistic for each step in this branch of the tree
            p_tree_accept = numpy.mean(p_step_accept)

            # Divergence
            if max(numpy.abs(energy_loss)) > 1000 or numpy.isnan(energy_loss).any():
                if accept_stat is None:
                    accept_stat = 0.0

                break

            if uturn_detected:
                # If we u-turn on the first step, grab something for the accept_stat
                if accept_stat is None:
                    accept_stat = p_tree_accept

                break

            old_accept_stat = accept_stat

            if accept_stat is None:
                accept_stat = p_tree_accept
            else:
                accept_stat = (accept_stat * (len(depth_steps) - 1) + p_tree_accept * leapfrogs_taken) / (
                    leapfrogs_taken + len(depth_steps) - 1
                )

            # log of the sum of pi (A.3.1 of https://arxiv.org/abs/1701.02434) of the new subtree
            log_sum_pi_new = scipy.special.logsumexp(log_pi_steps)

            # sample from the new subtree according to the equation in A.2.1 in https://arxiv.org/abs/1701.02434
            #  (near the end of that section)
            if depth > 1:
                i_new = depth_steps[numpy.where(choice_draws[depth - 1] < numpy.cumsum(scipy.special.softmax(log_pi_steps)))[0][0]]
            else:
                i_new = depth_steps[0]

            # Pick between the samples generated from the new and old subtrees using the biased progressive sampling in
            #  A.3.2 of https://arxiv.org/abs/1701.02434
            p_new = min(1, numpy.exp(log_sum_pi_new - log_sum_pi_old))
            if pick_draws[depth - 1] < p_new:
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
                valid=lambda df: True if not uturn_detected else df["depth_map"] < depth
            )
        ).iloc[steps_taken]

        return (
            q,
            accept_stat,
            2 ** depth,
            {
                "i": i_old - min(numpy.nonzero(depth_map <= depth)[0]) + 1,  # q is the ith row of trajectory
                "q0": q0,
                "h": h,
                "max_treedepth": max_treedepth,
                "trajectory": trajectory_df,  # tibble containing details of trajectory
                "directions": directions,  # direction integrated in each subtree
            },
        )
    else:
        return q, accept_stat, 2 ** depth


@dataclass
class StepsizeAdapter:
    target_accept_stat: float
    initial_stepsize: float
    gamma: float = 0.05
    kappa: float = 0.75
    t0: float = 10.0
    _counter: int = 0
    _s_bar: int = 0
    _x_bar: int = 0

    def adapt(self, accept_stat: float) -> float:
        self._counter += 1

        accept_stat = min(1.0, accept_stat)

        # Nesterov Dual-Averaging of log(epsilon)
        eta = 1.0 / (self._counter + self.t0)

        self._s_bar = (1.0 - eta) * self._s_bar + eta * (self.target_accept_stat - accept_stat)

        x = math.log(self.initial_stepsize) - self._s_bar * math.sqrt(self._counter) / self.gamma
        x_eta = self._counter ** (-self.kappa)

        self._x_bar = (1.0 - x_eta) * self._x_bar + x_eta * x

        return math.exp(x)

    def adapted_stepsize(self):
        return math.exp(self._x_bar)


def warmup(
    potential: Potential,
    rng: numpy.random.Generator,
    initial_draw: numpy.array,
    initial_stepsize: float = 1.0,
    initial_diagonal_inverse_metric: numpy.array = None,
    max_treedepth: int = 10,
    target_accept_stat: float = 0.8,
    stage_1_size: int = 100,
    stage_2_size: int = 850,
    stage_2_window_count: int = 4,
    stage_3_size: int = 50,
    debug: bool = False,
):
    size = initial_draw.shape[0]
    stepsize = initial_stepsize

    if initial_diagonal_inverse_metric is None:
        diagonal_inverse_metric = numpy.ones(size)
    else:
        diagonal_inverse_metric = initial_diagonal_inverse_metric
        assert diagonal_inverse_metric.shape[0] == size and diagonal_inverse_metric.shape[1] == size

    with tqdm(total=stage_1_size + stage_2_size + stage_3_size, desc="Moving from initial condition") as progress_bar:
        # Find an initial stepsize that is large enough we are under our target accept rate
        while True:
            next_draw, accept_stat, steps = one_draw(potential, rng, initial_draw, stepsize, diagonal_inverse_metric)
            if accept_stat < target_accept_stat:
                break
            stepsize = stepsize * 2.0

        # Back off until we hit our target accept rate
        while True:
            stepsize = stepsize / 2.0
            next_draw, accept_stat, steps = one_draw(potential, rng, initial_draw, stepsize, diagonal_inverse_metric)
            if accept_stat > target_accept_stat:
                break

        # Stage 1, leave initial conditions in the dust, adapting stepsize
        current_draw = initial_draw
        stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
        for i in range(stage_1_size):
            current_draw, accept_stat, steps = one_draw(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
            stepsize = stepsize_adapter.adapt(accept_stat)
            progress_bar.update()
            progress_bar.set_description(f"Moving from initial condition [leapfrogs {steps}]")

        # Stage 2, estimate diagonal of covariance in windows of increasing size
        stage_2_windows = []

        # Compute window size such that we have a sequence of stage_2_window_count
        # windows that sequentially double in size and in total take less than
        # or equal to stage_2_size draws. In math terms that is something like:
        # window_size + 2 * window_size + 4 * window_size + ... <= stage_2_size
        window_size = math.floor(stage_2_size / (2 ** stage_2_window_count - 1))

        for i in range(stage_2_window_count - 1):
            stage_2_windows.append(window_size * 2 ** i)

        # The last window is whatever is left
        stage_2_windows.append(stage_2_size - sum(stage_2_windows))

        for current_window_size in stage_2_windows:
            stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
            window_draws = numpy.zeros((current_window_size, size))
            for i in range(current_window_size):
                current_draw, accept_stat, steps = one_draw(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
                window_draws[i] = current_draw
                stepsize = stepsize_adapter.adapt(accept_stat)
                progress_bar.update()
                progress_bar.set_description(f"Building a metric [leapfrogs {steps}]")
            new_diagonal_inverse_metric = numpy.var(window_draws, axis=0)
            max_scale_change = max(diagonal_inverse_metric / new_diagonal_inverse_metric)
            stepsize = stepsize * max_scale_change
            diagonal_inverse_metric = new_diagonal_inverse_metric

        # Stage 3, fine tune timestep
        stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
        for i in range(stage_3_size):
            current_draw, accept_stat, steps = one_draw(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
            stepsize = stepsize_adapter.adapt(accept_stat)
            progress_bar.update()
            progress_bar.set_description(f"Finalizing timestep [leapfrogs {steps}]")

    return current_draw, stepsize_adapter.adapted_stepsize(), diagonal_inverse_metric


def sample(
    potential: Potential,
    rng: numpy.random.Generator,
    initial_draw: numpy.array,
    stepsize: float,
    diagonal_inverse_metric: numpy.array,
    num_draws: int,
    max_treedepth: int = 10,
):
    size = initial_draw.shape[0]
    draws = numpy.zeros((num_draws, size))

    with tqdm(total=num_draws, desc="Sampling") as progress_bar:
        current_draw = initial_draw
        for i in range(num_draws):
            current_draw, accept_stat, steps = one_draw(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
            draws[i, :] = current_draw
            progress_bar.update()
            progress_bar.set_description(f"Sampling [leapfrogs {steps}]")

    return draws
