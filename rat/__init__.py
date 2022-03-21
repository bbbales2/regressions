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

1. Rat uses a No-U-Turn-Sampler (following implementation in [Betancourt](https://arxiv.org/pdf/1701.02434.pdf))
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

## Basics

Rat syntax is built for writing vectorized expressions across dataframes. The idea
is that this is a useful framework for conceptualizing multilevel regressions, and
Rat tries to make it easy to do this in code.

Rat is made of *statements*. A statement in Rat consists of two *expressions* (the left hand side
and right hand side expressions) separated by either an `~` or an `=` and ending in `;`.
Those where the lefthand side and righthand side are separated by `~` are called sampling statements,
and those where the lefthand side and righthand side are separated by `=` are called assignments.
Expressions are composed of *functions* and *variable references* and do not contain any intervening `~`, `=`, or
`;` characters.

A statement can be multiple lines, and any line can end with a *comment* separated with `#`.
There are no multiline comments.

For example, this is a valid sampling statement in Rat:
```
score_diff' ~ normal( # It's a sampling statement!
    skill[home_team], # It's a regression with one group-level intercept!
    sigma);
```

Rat syntax vectorizes over dataframes. This means that each statement will produce code
that runs for each row in a dataframe. The dataframe that a statement vectorizes across is
called the *primary dataframe* -- there is exactly one primary dataframe per statement.

Rat statements resolve their primary dataframes in a process called *primary variable deduction*.
Every variable (a variable being a named symbol that is not a function) has exactly one dataframe
attached to it. To identify the primary dataframe, rat must identify the primary variable.

The rules for primary variable deduction are as follows (executed in order):

1. There can only be one primary variable in a statement.
2. If a variable reference is *primed*, then that variable is the primary variable.
3. If there is no primed variable references, then all variables with non-empty dataframes are treated as prime.
4. If there are no variables with non-empty dataframes, the leftmost one is the primary one.
5. It is an error if anything other than one primary variable is identified.
6. A *parameter* can only be used as the primary variable in one statement.

A variable reference can be primed if it ends with a `'`. A variable reference itself
is a variable name, an optional constraint, an optional sequence of *subscripts*, and an
followed by the optional prime symbol. In the example above, `score_diff`, `skill`, and
`sigma` are variable names. `home_team` is a subscript. `score_diff'`, `skill[home_team]`,
and `sigma` are the variable references, and `score_diff` is the primary variable.

Variables are associated with dataframes with the process of *variable dataframe deduction* (executed in order):

1. Variables with names that match columns in the *input dataframe* take the input dataframe
2. Otherwise, variables references are associated with parameters. The dataframe of the parameter is the minimum
dataframe required to execute all statements it is used in.

For the purposes of both dataframe deductions, Rat programs are understood top to bottom.

The input dataframe is a single dataframe passed to a Rat program. For the regression above, a
suitable example might be (the point differentials for a number of NBA games from the 2016 season):

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

Because `score_diff` matches a column in the input dataframe, the input dataframe
is the primary dataframe for this datement. Because `skill` and `sigma` do not appear
as columns in this dataframe, they will be parameters.

The statement will *execute* once for each line of this dataframe. As the statement runs,
it will substitute values from this dataframe into variables and subscripts that match
column names.

In this case, that means there must be a value in `skill` corresponding to each
of the home teams (`CLE, POR, GSW, ORL, IND, BOS, TOR, MIL, MEM, PHI`). Because there
is one subscript, the `skill` dataframe will have one column. The values in that column
will be extended as necessary to allow all the `home_team` references.

Because `sigma` has no subscripts, the minimum dataframe necessary to support it is
the empty dataframe.

Execution for a sampling statement means evaluating and accumulating the log density
given by the name of the distribution on the right hand side of the `~`.

Execution of assignments is described in [Transformed Parameters](#transformed-parameters) and
[Shifts in transformed Parameters](#shifts-in-transformed-parameters).

Considering the data, it may seem useful to predict the score differential of an NBA game
in terms of both teams' skills. A suitable model for this would be:
```
score_diff' ~ normal(skill[home_team] - skill[away_team], sigma);
```

In this statement, there is an extra variable reference, `skill[away_team]`. Because of
this, the `skill` dataframe will need to be extended to support all the away teams as well
(adding all the teams but CLE [Cleveland], which is already there).

## Transformed parameters

Modeling the `skill` parameter in the model above hierarchically with a non-centered
parameterization will be a good demonstration of assignment statements in Rat. As a reminder,
this sort of parameterization is useful for avoiding [divergences](https://mc-stan.org/users/documentation/case-studies/divergences_and_bias.html),
when doing MCMC using NUTS on multilevel models.

The non-centered parameterization takes the form:
```
score_diff' ~ normal(skill[home_team] - skill[away_team], sigma);
skill[team]' = skill_z[team] * tau;
skill_z ~ normal(0.0, 1.0);
```

For the second statement `skill` is the primary variable, so the dataframe defined implicitly
in the first statement will be the primary dataframe for the second line. Because of the subscript,
the parameter `skill_z` will be created with a non-empty dataframe. Because it has no subscript,
the parameter `tau` is created using the empty dataframe.

The subscript of `skill` is *renamed* to `team` by use as a primary variable. Because the subscript
is referenced by two names in the first statement, its name is ambiguous. Subscripts in a
primary variable reference are *renaming subscripts*. Because a parameter can only be used
as a primary variable once, this renaming only happens once. The renaming is necessary
for two reasons:
    
1. Rat needs the subscript to be named for output to work
2. Creating an underlying parameter `skill_z` for each `skill` requires that `skill_z` be
subscripted by the `skill` dataframe.

The assignment itself works by evaluating the expression on the right hand side and writing
the transformed parameter on the left (and the variable on the left hand side must
be a parameter, not data). Transformed parameters are immutable, so once they are set
they cannot be changed. All uses of a transformed parameter must preceed the definition. This
may seem non-intuitive coming from other languages, but in Rat parameter use defines the
parameters themselves -- the assignment will simply guarantee that the necessary values
get set. Effectively Rat statements will be executed in reverse order as they are written.

In the final statement the prior for `skill_z` is defined. Because `skill_z` is the only variable
in the statement, its dataframe will be the primary dataframe (and there is no need to prime it
manually). Because the use uniquely defines a name for the subscript, there is no need to rename it.

## Constraints

The model above won't get far without a prior on `tau`. Because `tau` is a
standard deviation, it must be constrained to be positive.

Rat adopts a similar constraint syntax to Stan:

```
score_diff' ~ normal(skill[home_team] - skill[away_team], sigma);
skill[team]' = skill_z[team] * tau;
skill_z ~ normal(0.0, 1.0);
tau<lower = 0.0> ~ normal(0.0, 1.0);
```

Constraints can only be used on parameters.

The constraint goes after the parameter name but before the subscripts. Rat
supports a `lower`, `upper` and a combination of the two constraints.

----


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



Rat may infer that a variable is a parameter by its use, but this parameter
doesn't necessarily need to directly be a parameter of the distribution sampled
with MCMC. Transformed parameters are immutable functions of other parameters
that are set in assignment statements (statements where the left and right hand
side is separated by an `=`).

One of the basic things transformed parameters let us do is implement a
non-centered parameterization. Internally Rat uses a NUTS sampler. It is useful to
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

## Shifts in transformed parameters

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

Rat is only available from Github, and requires [Rust](https://www.rust-lang.org/) to
be installed to work.

First, follow the [Rust installation directions](https://www.rust-lang.org/tools/install).

Secondly, install rat from Github:

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

## Internals - `rat.variables`, `rat.ops`, `rat.variables.Subscript`, and `rat.variables.SubscriptUse`
(This portion is for people who want to dig into rat's source)

If you look closely at rat's source, you might notice something weird: There's a `rat.ops.Param` class and a `rat.variables.Param`. This also goes for `rat.ops.Data`, `rat.ops.Subscript`. `rat.ops` is designated as elements for the nodes of the parse tree that's generated by the parser. Since rat uses the parse tree to transpile to Python, we need to inject additional information regarding subscripts to the parse tree. This is where `rat.variables` comes into play: they are used to hold information regarding subscripts for a given parameter. I'll use the [skill model](#subscripts) example to explain:

The parser converts the skill model above into the following parse tree representations:
```
ops.Normal(
    ops.Data("score_diff"),
    ops.Diff(
        ops.Param("skills", ops.Subscript(("home_team", "year"))),
        ops.Param("skills", ops.Subscript(("away_team", "year"))),
    ),
    ops.Param("sigma"),
),
ops.Normal(
    ops.Param("skills", ops.Subscript(("team", "year"))),
    ops.Param("skills_mu", ops.Subscript(("year",))),
    ops.Param("tau"),
),
ops.Normal(
    ops.Param("skills_mu", ops.Subscript(("year",))),
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

For each parameter we create a `rat.variables.Subscript` object which in essence hold the factors of the subscript. Recall `skills[team, year]` was in fact `skills[union(home_team, away_team), year]`. So we need a parameter for each team-year combination. `rat.variables.Subscript` internally stores a dataframe with columns `team` and `year`, which hold these unique combinations as reference subscripts.

But we might want to just subscript `home_team` from `team`. That is, we might only want a portion of the subscript. `rat.variables.SubscriptUse` is the object that maps a variable's subscripts to another variable's reference subscript(its `rat.variables.Subscript` object). If we just wanted to subscript `home_team` from `team`, it will compare the `home_team` dataframe with `team`'s reference dataframe, and only select the combinations that are necessary.

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
