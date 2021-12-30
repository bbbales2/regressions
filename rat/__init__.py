r"""

Rat is an attempt to build an easy to use regression syntax. It is inspired by
the many fine R regression packages (lm, lme4, rstanarm, brms, etc.),
but tries to take its own twist on the problem. The design principles behind
rat are:

1. Parameters are named explicitly
2. Parameters are allocated as they are used
3. Long-form dataframes are the central data structure

The first principle comes from wanting to write likelihoods where a parameter
shows up more than once. For instance we can model the score difference
at the end of a game ($y_{ij}$) as a function of the skills of two teams
($s_i$ and $s_j$) with a simple regression:

$$
y_{ij} \sim \mathcal{N}(s_i - s_j)
$$

Those skills on the left and the right are the same parameter! We didn't know
how to write these models with existing regression packages. Sure we could make a
sparse matrix and each row could have two non-zeroes corresponding to the teams
that played and multiply it by a vector -- but annoying. We're fond of the
R regression shorthands and wanted to use those.

The second principle came from the fact that, while a model like the one above
can be written in a general purpose probabilistic programming language, it's
mildly annoying. Converting player names to factors, converting them to integers,
allocating parameters -- it all kinda takes time. It seemed easier to write
a new regression language.

The third principle comes from a couple places, but the basic jist is that it's
very easy to get data into the R regression packages and that's something we want
to keep. We also wanted to make sure it was easy to get data out and also easy
to tie that output back to the input. In Rat, everything is represented as long
form dataframes to make it easy to join outputs to inputs and each other.

# Rat Language

## Likelihood

Rat inherits from the regression languages. As such, the first line of a Rat
program is the likelihood. This line is special because it can access named
variables in the input dataframe.

For instance, assume we have the following dataframe:

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

The first line of a Rat program could be:

```
score_diff ~ normal(skill[home_team], sigma);
```

This says model `score_diff` as a normally distributed random variable given
a mean `skill[home_team]` and standard deviation `sigma`. There will be one
term in the likelihood for every row in the dataframe (so each row corresponds
to a conditionally independent term).

Where does `score_diff` come from? `score_diff` is a column of our input
dataframe (and is the score difference between the home and away teams for a
handful of games in the 2016 NBA season). Again, the first line of a Rat
model defines the likelihood and has special access to the input dataframe.

The other column of the dataframe that is being used is `home_team`. If we
look back at the dataframe, `home_team` is a string identifying which NBA
team was playing at home in each game. In the model it appears in brackets
after the variable `skill` -- in Rat terms `home_team` *subscripts* `skill`.
This means for each row of the input dataframe, take the entry of the variable
skill that corresponds to the value of `home_team`.

Because `skill` is subscripted (and is not a column in the dataframe), Rat
infers that it is a parameter in the model that must be estimated. There
will be as many unique elements of `skill` as there are unique elements
of `home_team`, because this is exactly how many parameters need to exist
to evaluate the likelihood. When it comes time to 

Because `sigma` is not a column in the dataframe, Rat infers it is a parameter.
Because it is not subscripted, Rat infers it is a scalar parameter.

## Priors

Priors in Rat are defined by every line but the first. The first line is special
because it has access to the input dataframe -- the rest do not.

For instance, the above example can be extended to have a prior on `skill`:

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
will be called `team` for the purposes of defining the prior and handling I/O. The
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
with MCMC. Transformed parameters are immutable functions of other parameters.

One of the basic things transformed parameters let us do is implement a
non-centered parameterization. Internally RAT uses the NUTS sampler in
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
parameters are immutable, so once they are defined they cannot be changed.
Similarly, a transformed parameter cannot be used in its own definition.

Variables on the right hand side expression that Rat does not recognize
will be inferred as other parameters. In this case, `mu`, `tau`, and `z`,
the untransformed version of `theta`.

## Shift operator

The elements of non-scalar Rat parameters are sorted with respect to the
values of the subscripts. Because elements have an order, we can think
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

## Subscripts

In general, subscripts are resolved and generated
when found on the **right-hand-side of a statement that has a left-hand-side
with resolved subscripts**. However since rat topologically sorts before
evaluation accordingly, the user just has to make sure parameters that they
have declared/assigned with subscripts must be resolvable by checking they
are present as right-hand-side.


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

## CLI

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
