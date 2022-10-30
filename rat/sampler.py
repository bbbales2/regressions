from concurrent.futures import ThreadPoolExecutor
import numpy
import pandas

from . import fit
from . import nuts


def sample(self, num_draws=200, num_warmup=1000, chains=4, init=2, thin=1, target_acceptance_rate=0.85):
    """
    Sample the target log density using NUTS.

    Sample using `chains` different chains with parameters initialized in unconstrained
    space [-2, 2]. Use `num_warmup` draws to warmup and collect `num_draws` draws in each
    chain after warmup.

    If `thin` is greater than 1, then compute internally `num_draws * thin` draws and
    output only every `thin` draws (so the output is size `num_draws`).

    `target_acceptance_rate` is the target acceptance rate for adaptation. Should be less
    than one and greater than zero.
    """
    # Currently only doing warmup on one chain
    initial_position = 2 * init * numpy.random.uniform(size=(self.size)) - init

    assert target_acceptance_rate < 1.0 and target_acceptance_rate > 0.0
    assert num_warmup > 200

    def negative_log_density(q):
        return -self.log_density(q)

    potential = nuts.Potential(negative_log_density, chains, self.size)
    rng = numpy.random.default_rng()

    # Ordered as (draws, chains, param)
    unconstrained_draws = numpy.zeros((num_draws, chains, self.size))
    leapfrog_steps = numpy.zeros((num_draws, chains), dtype=int)
    divergences = numpy.zeros((num_draws, chains), dtype=bool)

    def generate_draws():
        stage_1_size = 100
        stage_3_size = 50
        stage_2_size = num_warmup - stage_1_size - stage_3_size

        initial_draw, stepsize, diagonal_inverse_metric = nuts.warmup(
            potential,
            rng,
            initial_position,
            target_accept_stat=target_acceptance_rate,
            stage_1_size=stage_1_size,
            stage_2_size=stage_2_size,
            stage_3_size=stage_3_size,
        )

        return nuts.sample(potential, rng, initial_draw, stepsize, diagonal_inverse_metric, num_draws, thin)

    with ThreadPoolExecutor(max_workers=chains) as e:
        results = []
        for chain in range(chains):
            results.append(e.submit(generate_draws))

        for chain, result in enumerate(results):
            unconstrained_draws[:, chain, :], leapfrog_steps[:, chain], divergences[:, chain] = result.result()

    constrained_draws, base_dfs = self._prepare_draws_and_dfs(unconstrained_draws)
    computational_diagnostic_variables = {"__leapfrog_steps": leapfrog_steps, "__divergences": divergences}

    for name, values in computational_diagnostic_variables.items():
        if name in constrained_draws:
            print(f"{name} already exists in sampler output, not writing diagnostic variable")
        else:
            constrained_draws[name] = values
            base_dfs[name] = pandas.DataFrame()

    return fit.SampleFit._from_constrained_variables(constrained_draws, base_dfs, computational_diagnostic_variables.keys())
