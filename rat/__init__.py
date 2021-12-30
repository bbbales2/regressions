r"""

Rat is an attempt to build an easy to use regression syntax, particularly
focused on player skill models. It is similar in theme to the many fine
regression packages (lm, lme4, rstanarm, brms, etc.), but tries to take
its own twist on the problem.

Some central Rat features are:

1. Parameters are named explicitly
2. Parameters are defined by their use
3. Long-form dataframes are the data structure for everything (data and parameters)

Some central technical pieces are:

1. Rat uses a No-U-Turn-Sampler (implementation from [blackjax](https://github.com/blackjax-devs/blackjax/tree/main/blackjax))
2. Rat uses autodiff from [jax](https://github.com/google/jax)

Rat works in a limited language space to keep the backend stuff simple
(no sampling discrete parameters and no loops).

# Language

## Anatomy of a Rat (program)

There are two full Rat examples included with the repo that are worth glancing at.
The first ([examples/mrp](https://github.com/bbbales2/regressions/tree/main/examples/mrp))
is an MRP example ported from
[MRP Case Studies](https://bookdown.org/jl5522/MRP-case-studies/introduction-to-mrp.html).

The second ([examples/fakeball](https://github.com/bbbales2/regressions/tree/main/examples/fakeball))
is an attempt to simulate some fake basketball-like data and estimate player on-off
effectiveness numbers.

The example folders contain information on how to run these models, but a quick look at
the second model might be useful to see where we're going with all this:

```
# We're modeling whether or not shots were made as a function of the time
# varying skill of the five players playing offense and the five players
# playing defense. `made`, `date`, `o0-o4`, and `d0-d4` come from the input,
# dataframe. o0-o4 and d0-d4 are names of the five players on the floor
# playing offense and defense
made ~ bernoulli_logit(
    offense[o0, date] + offense[o1, date] + offense[o2, date] + offense[o3, date] + offense[o4, date] -
    (defense[d0, date] + defense[d1, date] + defense[d2, date] + defense[d3, date] + defense[d4, date])
);

# A player's skill is a function of their initial skill plus some random
# walk that changes over time
offense[player, date] = offense0[player] + offense_rw[player, date];
defense[player, date] = defense0[player] + defense_rw[player, date];

# Parameters are defined by use -- we need to define offense0 and defense0
# because they are used elsewhere
offense0[player] ~ normal(0.0, tau0_offense);
defense0[player] ~ normal(0.0, tau0_defense);

# This is the random walk -- read on to understand how `shift` actually works
offense_rw[player, date] ~ normal(offense_rw[player, shift(date, 1)], tau_offense);
defense_rw[player, date] ~ normal(defense_rw[player, shift(date, 1)], tau_defense);

# Some parameters have constraints!
tau_offense<lower = 0.0> ~ log_normal(0.0, 0.5);
tau_defense<lower = 0.0> ~ log_normal(0.0, 0.5);
tau0_offense<lower = 0.0> ~ log_normal(0.0, 0.5);
tau0_defense<lower = 0.0> ~ log_normal(0.0, 0.5);
```

Assuming we've saved appropriate data in a file `shots.csv`, then a model like this can
be run on the command-line (or from Python, but the command line is convenient for
this sort of thing):

```
rat fakeball.rat shots.csv samples
```

The output can be extracted and summarized in Python like:

```python
from rat.fit import load
import numpy

# Rat fits are serialized as parquet tables in folders
fit = load("samples")

# Each parameter is stored in its own table -- these can be joined together
# or summarized on their own
offense_df = fit.draws("offense")

# In this case we can build a table mapping player, date tuples to parameter
# summaries
offense_summary_df = (
    offense_df
    .groupby(["player", "date"])
    .agg(
        median=("offense", numpy.median),
        q10=("offense", lambda x: numpy.quantile(x, 0.1)),
        q90=("offense", lambda x: numpy.quantile(x, 0.9)),
    )
)
```

## Likelihood

Every Rat program comes with a dataframe. The dataframe defines the data which
the Rat model fits itself to.

Rat programs are broken up into statements separated by semicolons. There are
two types of statements, sampling statements (those where the lefthand side
and righthand side are separated by `~`) and assignments (the lefthand side
and righthand side are separated by `=`).

Sampling statements come in two varieties, likelihoods and priors. Likelihoods
are the sampling statements where the variable name on the left hand side of the
`~` comes from the input dataframe (and priors are the other ones). This section
discusses the basics of likelihood statements. Priors and assignments are discussed
later (under section [Priors](#priors) and [Transformed Parameters](#transformed-parameters)).

Assume we have the following dataframe:

```
    game_id  home_score  away_score home_team away_team  score_diff  year
0         1         117          88       CLE       NYK        29.0  2016
1         2         113         104       POR       UTA         9.0  2016
2         3         100         129       GSW       SAS       -29.0  2016
3         4          96         108       ORL       MIA       -12.0  2016
4         5         130         121       IND       DAL         9.0  2016
5         6         122         117       BOS       BKN         5.0  2016
6         7         109          91       TOR       DET        18.0  2016
7         8          96         107       MIL       CHA       -11.0  2016
8         9         102          98       MEM       MIN         4.0  2016
9        10         102         107       NOP       DEN        -5.0  2016
10       11          97         103       PHI       CLE        -6.0  2016
```

The first line of a Rat program modeling this data might be:

```
score_diff ~ normal(skill[home_team], sigma);
```

Because this is a sampling statement (the `~`) and name on the lefthand side
is in the dataframe, this is a likelihood statement.

This says model `score_diff` (the score difference between the home and away teams for a
handful of games in the 2016 NBA season) as a normally distributed random variable given
a mean `skill[home_team]` and standard deviation `sigma`. There will be one
term in the likelihood for every row in the dataframe (so each row corresponds
to a conditionally independent term).

The other column of the dataframe that is being used is `home_team`. If we
look back at the dataframe, `home_team` is a string identifying which NBA
team was playing at home in each game. In the model it appears in brackets
after the variable `skill` -- in Rat terms `home_team` *subscripts* `skill`.
This means for each row of the input dataframe, take the entry of the variable
skill that corresponds to the value of `home_team`.

Because `skill` is subscripted (and is not a column in the dataframe), Rat
infers that it is a parameter in the model. There will be as many unique
elements of `skill` as there are unique elements of `home_team`, because
this is exactly how many parameters need to exist to evaluate the likelihood.

Because `sigma` is not a column in the dataframe, Rat infers it is a parameter.
Because it is not subscripted, Rat infers it is a scalar parameter.

## Priors

Priors in Rat are sampling statements that are not likelihoods (the name on the
left hand side does not appear in the input dataframe). Priors cannot reference
columns of the input dataframe (it's an error).

The above example can be extended to have a prior on `skill`:

```
score_diff ~ normal(skill[home_team], sigma);
skill[home_team] ~ normal(0.0, tau);
```

The first line defines `skill` by its use (it is a parameter with as many entries
as there are unique values of `home_team`). The second line says, for any entry of
skill, use a normal with standard deviation `tau`. Because `tau` is not used
anywhere yet, Rat will infer that it is a new scalar random variable.

The `home_team` subscript on the second line *matches* the subscript in the original
use. In the example above, this probably seems extraneous, but it is useful in more
complex cases. First consider a two-sided skill model:

```
score_diff ~ normal(skill[home_team] - skill[away_team], sigma);
skill[team] ~ normal(0.0, tau);
```

In this case, `skill` is subscripted both by `home_team` and `away_team` and
there will be as many elements of `skill` as required in both cases (it's possible
some teams only played away games -- perhaps we're running this regression early in the
year). This leads to the question -- how should the first subscript of `skill` be
referenced? It is not obvious, and so when the prior is defined, the subscripts
are matched by position and we provide a new name. The first subscript to `skill`
will be called `team` for the purposes of defining the prior and handling output. The
subscript values themselves are defined by the use of the `skill` parameter.

Naming the subscripts with the match is also useful for deeper parameter hierarchies.
For instance, this model may extend over many years, in which case we could write:

```
score_diff ~ normal(skill[home_team, year] - skill[away_team, year], sigma);
skill[team, year] ~ normal(all_time_skill[team], tau);
```

`skill` is now subscripted by two columns of the dataframe in two different ways,
`[home_team, year]` and `[away_team, year]`. There will be an element of `skill`
defined for every combination of home team and year, and away team and year.

The prior for the `[team, year]` element of `skill` is now a normal with mean
`all_time_skill[team]` and standard deviation `tau`. Because `all_time_skill`
does not exist and is a subscripted variable, Rat will infer that it is a parameter
with unique elements given by all possible values of teams.

Matching is done by parameter position. If we had swapped the `year` and `team`
subscripts in the prior we would still have a valid Rat model, but the elements of the `team`
subscript would correspond to the unique values of year in the original dataframe!

(Beware this:)

```
skill[year, team] ~ normal(all_time_skill[team], tau);
```

## Constraints

The model above won't get far without a prior on `tau`. Because `tau` is a
standard deviation, it must be constrained to be positive.

Rat adopts a similar constraint syntax to Stan:

```
score_diff ~ normal(skill[home_team], sigma);
skill[home_team] ~ normal(0.0, tau);
tau<lower = 0.0> ~ normal(0.0, 1.0);
```

The constraint goes after the parameter name but before the subscripts. Rat
supports a `lower`, `upper` and a combination of the two constraints.

## Transformed parameters

Rat may infer that a variable is a parameter by its use, but this parameter
doesn't necessarily need to be a parameter of the joint distribution sampled
with MCMC. Transformed parameters are immutable functions of other parameters
that are set in assignment statements (statements where the left and righthand
side is separated by an `=`).

One of the basic things transformed parameters let us do is implement a
non-centered parameterization. Internally Rat uses the NUTS sampler in
[blackjax](https://github.com/blackjax-devs/blackjax). It is useful to
reparameterize hierarchical models for NUTS to avoid
[divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html).

The eight schools data is as follows:
```
    y  sigma  school
0  28     15       1
1   8     10       2
2  -3     16       3
3   7     11       4
4  -1      9       5
5   1     11       6
6  18     10       7
7  12     18       8
```

The rat code for the centered eight schools model is the following:

```
y ~ normal(theta[school], sigma);
theta[school] ~ normal(mu, tau);
mu ~ normal(0, 5);
tau<lower = 0.0> ~ log_normal(0, 1);
```

`y` and `sigma` come from the dataframe. Rat infers `theta` is a parameter
with one element for each school, and `mu` and `tau` are both scalar parameters.

This model is difficult for NUTS to sample. The non-centered eight
schools model is the following: 

```
y ~ normal(theta[school], sigma);
theta[school] = mu + z[school] * tau;
z[school] ~ normal(0, 1);
mu ~ normal(0, 5);
tau<lower = 0.0> ~ log_normal(0, 1);
```

The new line of code is the second -- in this case we say the elements of
`theta` are equal to the expression on the right hand side. Transformed
parameters are immutable, so once they are set they cannot be changed.
Similarly, a transformed parameter cannot be used on the righthand side of
its own assignment.

Variables on the right hand side of an assignment that Rat does not recognize
will be inferred as other parameters. In this case, `mu`, `tau`, and `z`,
the untransformed versions of `theta`.

## Shift operator

The elements of non-scalar Rat parameters are sorted with respect to the
values of the subscripts (in the order of the subscripts, so the rightmost
subscript is sorted last). Because elements have an order, we can think
about a previous and next element. This is useful for time series models.

Going back to the basketball model, we might be interested in how a team's
skill changes year to year:

```
score_diff ~ normal(skill[home_team, year] - skill[away_team, year], sigma);
skill[team, year] ~ normal(skill[team, shift(year, 1)], tau);
```

The shift of 1 on the `skill` year subscript means, for the given team, take
the previous year's `skill` as the mean in the prior for the current year.

Positive shifts mean take previous values of the `skill` parameter; negative
shifts mean take following values of the `skill` parameter. Any out of
bounds access on the `skill` parameter is replaced with a zero.

Multiple variables can be shifted different lengths (though each variable
gets its own shift). The shifts are done only within groups defined by the
unshifted parameters. That is a mouthful, but in terms of basketball example
this looks like:

| `skill[team, year]` | `skill[team, shift(year, 1)]` |
| --------------------------------------------------- |
| `skill[CHA, 2016]`  | `0`                           |
| `skill[CHA, 2017]`  | `skill[CHA, 2016]`            |
| `skill[ATL, 2017]`  | `0`                           |

A negative shift produces a different result:

| `skill[team, year]` | `skill[team, shift(year, -1)]` |
| --------------------------------------------------- |
| `skill[CHA, 2016]`  | `skill[CHA, 2017]`            |
| `skill[CHA, 2017]`  | `0`                           |
| `skill[ATL, 2017]`  | `0`                           |

## Execution order

Rat sorts statements to make sure they are evaluated in an order so that
all transformed parameters are set before they are used. This means the user
does not need to worry about statement order -- just that there are either priors
or assignments for every parameter used in the program.

## Distributions

$\mathcal{R}$ here means all real numbers and $\mathcal{R}^+$ means real numbers greater than zero.


| Distribution | Constraints |
| ---------------------------|
| `y ~ bernoulli_logit(logit_p)` | $y = 0$ or $y = 1$, $\text{logit_p} \in \mathcal{R}$ |
| `y ~ cauchy(location, scale)` | $y, \text{location} \in \mathcal{R}$, $\text{scale} \in \mathcal{R}^+$ |
| `y ~ exponential(scale)` | $y, \text{scale} \in \mathcal{R}^+$ |
| `y ~ log_normal(mu, sigma)` | $\text{mu} \in \mathcal{R}$, $y, \text{sigma} \in \mathcal{R}^+$ |
| `y ~ normal(mu, sigma)` | $y, \text{mu} \in \mathcal{R}$, $\text{sigma} \in \mathcal{R}^+$ |

## Functions

* `abs(x)`
* `arccos(x)`
* `arcsin(x)`
* `arctan(x)`
* `ceil(x)`
* `cos(x)`
* `exp(x)`
* `floor(x)`
* `inverse_logit(x)`
* `log(x)`
* `logit(x)`
* `round(x)`
* `sin(x)`
* `tan(x)`

## Operator Precedence Table

|  Operator   |   Precedence   |
|------|:-----:|
| function calls(`exp`, `log`, etc.) | 100, leftmost derivative  |
| prefix negation(`-10`, `-(1+2)`, etc.) | 50 |
| `^`  | 40  |
| `*`, `/`, `%`  | 30  |
| `+`, `-`  | 10  |

# Installation and Use

Rat is only available from Github:

```
git clone https://github.com/bbbales2/regressions
cd regressions
pip install .
```

## Command line interface

Rat is a Python library, but comes with a helper script `rat` to quickly compile and
run models.

For example, to fit a model `mrp.rat` with the data `mrp.csv` and save the results in
`output` we can do:

```
rat mrp.rat mrp.csv output
```

Type `rat -h` for full usage info.
"""

__all__ = ["model", "fit"]

_todo_figure_out_internal_docs = """

---

## Internals - `rat.variables`, `rat.ops`, `rat.variables.Index`, and `rat.variables.IndexUse`
(This portion is for people who want to dig into rat's source)

If you look closely at rat's source, you might notice something weird: There's a `rat.ops.Param` class and a `rat.variables.Param`. This also goes for `rat.ops.Data`, `rat.ops.Index`. `rat.ops` is designated as elements for the nodes of the parse tree that's generated by the parser. Since rat uses the parse tree to transpile to Python, we need to inject additional information regarding subscripts to the parse tree. This is where `rat.variables` comes into play: they are used to hold information regarding subscripts for a given parameter. I'll use the [skill model](#subscripts) example to explain:

The parser converts the skill model above into the following parse tree representations:
```
ops.Normal(
    ops.Data("score_diff"),
    ops.Diff(
        ops.Param("skills", ops.Index(("home_team", "year"))),
        ops.Param("skills", ops.Index(("away_team", "year"))),
    ),
    ops.Param("sigma"),
),
ops.Normal(
    ops.Param("skills", ops.Index(("team", "year"))),
    ops.Param("skills_mu", ops.Index(("year",))),
    ops.Param("tau"),
),
ops.Normal(
    ops.Param("skills_mu", ops.Index(("year",))),
    ops.RealConstant(0.0),
    ops.RealConstant(1.0),
),
ops.Normal(
    ops.Param("tau", lower=ops.RealConstant(0.0)),
    ops.RealConstant(0.0),
    ops.RealConstant(1.0),
),
ops.Normal(
    ops.Param("sigma", lower=ops.RealConstant(0.0)),
    ops.RealConstant(0.0),
    ops.RealConstant(1.0),
)
```

For each parameter we create a `rat.variables.Index` object which in essence hold the factors of the subscript. Recall `skills[team, year]` was in fact `skills[union(home_team, away_team), year]`. So we need a parameter for each team-year combination. `rat.variables.Index` internally stores a dataframe with columns `team` and `year`, which hold these unique combinations as reference subscripts.

But we might want to just subscript `home_team` from `team`. That is, we might only want a portion of the subscript. `rat.variables.IndexUse` is the object that maps a variable's subscripts to another variable's reference subscript(its `rat.variables.Index` object). If we just wanted to index `home_team` from `team`, it will compare the `home_team` dataframe with `team`'s reference dataframe, and only select the combinations that are necessary.

---

## Language function reference
Below are individual links to supported math functions and distributions

- **Distributions**
    - `rat.ops.Normal` : `normal(mu, sigma)`
    - `rat.ops.BernoulliLogit`, `rat.compiler.bernoulli_logit` : `bernoulli_logit(p)`
    - `rat.ops.LogNormal`, `rat.compiler.log_normal`  : `log_normal(mu, sigma)`
    - `rat.ops.Cauchy`  : `cauchy(location, scale)`
    - `rat.ops.Exponential`  :  `exponential(scale)`
- **Functions**
    - `rat.ops.Log`: `log(x)`
    - `rat.ops.Exp` : `exp(x)`
    - `rat.ops.Abs` : `abs(x)`
    - `rat.ops.Floor` : `floor(x)`
    - `rat.ops.Ceil` : `ceil(x)`
    - `rat.ops.Round` : `round(x)`
    - `rat.ops.Sin` : `sin(x)`
    - `rat.ops.Cos` : `cos(x)`
    - `rat.ops.Tan` : `tan(x)`
    - `rat.ops.Arcsin` : `arcsin(x)`
    - `rat.ops.Arccos` : `arccos(x)`
    - `rat.ops.Arctan` : `arctan(x)`
    - `rat.ops.Logit` : `logit(x)`
    - `rat.ops.InverseLogit` : `inverse_logit(x)`

"""
