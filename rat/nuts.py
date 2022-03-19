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


from . import one_draw


class Potential:
    """
    Potential is a tool for handling gradient calculations on many threads
    """

    value_and_grad_negative_log_density: Callable[[numpy.array], Tuple[float, numpy.array]]
    """Gradient of negative log density"""

    potential_results: numpy.array
    """Pre-allocated array of potential energies"""

    gradient_results: numpy.array
    """Pre-allocated array of gradients"""

    maximum_threads: int
    """Maximum number of client threads"""

    local_identity: Dict[int, int]
    """Remap thread identities to integers"""

    def __init__(self, negative_log_density, maximum_threads, size):
        """
        negative_log_density should be a jax-friendly negative log density function that
        returns a scalar and takes in a vector of length size.

        maximum_threads is the maximum number of threads that will be using this
        potential for its lifetime (which could be greater than the number using
        it at any one point).

        maximum_threads and size are used to pre-allocate memory, so they must be
        specified.
        """
        self.value_and_grad_negative_log_density = jax.jit(jax.value_and_grad(negative_log_density))

        self.potential_results = numpy.zeros(maximum_threads, dtype=numpy.float32)
        self.gradient_results = numpy.zeros((maximum_threads, size), dtype=numpy.float32)

        self.maximum_threads = maximum_threads
        self.local_identity = {}

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

    def value_and_grad(self, q0: numpy.array) -> Tuple[float, numpy.array]:
        active_thread = self._get_ident()

        device_U, device_gradients = self.value_and_grad_negative_log_density(q0)

        self.potential_results[active_thread] = device_U
        self.gradient_results[active_thread, :] = device_gradients

        return self.potential_results[active_thread], self.gradient_results[active_thread, :]


@dataclass
class StepsizeAdapter:
    """
    Do stepsize adaptation like Stan (https://github.com/stan-dev/stan/blob/develop/src/stan/mcmc/stepsize_adaptation.hpp)

    I've renamed delta from the Stan implementation to target_accept_stat and mu is equal to log(initial_stepsize)
    """

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


def one_draw_potential(
    potential: Potential,
    rng: numpy.random.Generator,
    current_draw: numpy.array,
    stepsize: float,
    diagonal_inverse_metric: numpy.array,
    max_treedepth: int = 10,
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

    Return a tuple containing containing next draw, the acceptance probability statistic, and how many leapfrog
    steps were used to compute the draw
    """
    return one_draw.do_it(potential.value_and_grad, rng, current_draw, diagonal_inverse_metric, stepsize, max_treedepth)


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
):
    """
    Do warmup for a NUTS sampler given a potential (function proportional to the negative log
    density of the distribution we want to sample), a random number generator, and an
    initial position in sample space

    Return a tuple (draw, stepsize, diagonal_inverse_metric) where draw is a suitable
    draw for starting an MCMC calculation, stepsize is a stepsize that will hopefully
    allow the sampler to hit the target acceptance rate, and diagonal_inverse_metric
    is a scaling of parameter space to make the NUTS sampler reasonably efficient

    Though stepsize and diagonal_inverse_metric are estimated in the process, optional
    initial values can be provided.

    target_accept_stat is the target acceptance rate for the sampler.

    Adaptation is broken into three stages, each using a specified number of MCMC draws to
    accomplish their task.

    Stage 1 adapts a stepsize and tries to move the sampler close to where it will equilibrate.

    Stage 2 adapts a metric iteratively. It is further broken into stages (how many times the
    metric is recomputed). The number of stages here can be specified with stage_2_window_count.

    Stage 3 adapts a final timestep after an appropriate metric has been found.
    """

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
            next_draw, accept_stat, steps, divergence = one_draw_potential(potential, rng, initial_draw, stepsize, diagonal_inverse_metric)
            if accept_stat < target_accept_stat:
                break
            stepsize = stepsize * 2.0

        # Back off until we hit our target accept rate
        while True:
            stepsize = stepsize / 2.0
            next_draw, accept_stat, steps, divergence = one_draw_potential(potential, rng, initial_draw, stepsize, diagonal_inverse_metric)
            if accept_stat > target_accept_stat:
                break

        # Stage 1, leave initial conditions in the dust, adapting stepsize
        current_draw = initial_draw
        stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
        for i in range(stage_1_size):
            current_draw, accept_stat, steps, divergence = one_draw_potential(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
            stepsize = stepsize_adapter.adapt(accept_stat)
            progress_bar.update()
            progress_bar.set_description(f"Moving from initial condition [leapfrogs {steps}]")

        # Stage 2, estimate diagonal of covariance in windows of increasing size
        stage_2_windows = []

        # Compute window size such that we have a sequence of stage_2_window_count
        # windows that sequentially double in size and in total take less than
        # or equal to stage_2_size draws. In math terms that is something like:
        # window_size + 2 * window_size + 4 * window_size + ... <= stage_2_size
        window_size = math.floor(stage_2_size / (2**stage_2_window_count - 1))

        for i in range(stage_2_window_count - 1):
            stage_2_windows.append(window_size * 2**i)

        # The last window is whatever is left
        stage_2_windows.append(stage_2_size - sum(stage_2_windows))

        for current_window_size in stage_2_windows:
            stepsize_adapter = StepsizeAdapter(target_accept_stat, stepsize)
            window_draws = numpy.zeros((current_window_size, size))
            for i in range(current_window_size):
                current_draw, accept_stat, steps, divergence = one_draw_potential(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
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
            current_draw, accept_stat, steps, divergence = one_draw_potential(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
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
    thin: int = 1,
    max_treedepth: int = 10,
):
    """
    Run the NUTS sampler given a potential (function proportional to negative log density of distribution
    we want to sample), a random number generator, an initial draw, a stepsize, a diagonal inverse metric,
    a desired number of draws, and the maximum treedepth to be used in the calculation.

    The initial draw, stepsize, and diagonal inverse metric should be computed with warmup in most situations.
    """
    size = initial_draw.shape[0]
    draws = numpy.zeros((num_draws, size))
    leapfrog_steps = numpy.zeros(num_draws, dtype = int)
    divergences = numpy.zeros(num_draws, dtype = bool)

    desc = "Sampling" if thin == 1 else "Sampling (pre-thin)"

    with tqdm(total=num_draws * thin, desc=desc) as progress_bar:
        current_draw = initial_draw
        for i in range(num_draws):
            for j in range(thin):
                current_draw, accept_stat, steps, divergence = one_draw_potential(potential, rng, current_draw, stepsize, diagonal_inverse_metric)
                progress_bar.update()
                progress_bar.set_description(f"Sampling [leapfrogs {steps:4d}]")
            draws[i, :] = current_draw
            leapfrog_steps[i] = steps
            divergences[i] = divergence

    return draws, leapfrog_steps, divergences
