from rat.fit import load
import numpy
import os
import pandas
import plotnine
import scipy.special

mrp_folder = os.path.dirname(__file__)

fit = load(os.path.join(mrp_folder, "samples"))

state_df = pandas.read_csv(os.path.join(mrp_folder, "statelevel_predictors.csv"))
poststrat_df = pandas.read_csv(os.path.join(mrp_folder, "poststrat_df.csv"))

variables = ["state", "eth", "male", "age", "educ"]

full_poststrat_df = (
    poststrat_df
    .merge(fit.draws("a_age"), how = "outer") # First is an outer so poststrat table gets chain/draw
    .merge(fit.draws("a_state"), how = "left") # The rest can be lefts
    .merge(fit.draws("a_eth"), how = "left")
    .merge(fit.draws("a_educ"), how = "left")
    .merge(fit.draws("a_male_eth"), how = "left")
    .merge(fit.draws("a_educ_age"), how = "left")
    .merge(fit.draws("a_educ_eth"), how = "left")
    .merge(fit.draws("b_repvote"), how = "left")
    .merge(fit.draws("b_male"), how = "left")
    .merge(state_df[["state", "repvote"]], how = "left")
    .assign(p = lambda df : scipy.special.expit(
        df["a_state"] +
        df["a_age"] +
        df["a_eth"] +
        df["a_educ"] +
        df["a_male_eth"] +
        df["a_educ_age"] +
        df["a_educ_eth"] +
        df["b_repvote"] * df["repvote"] +
        df["b_male"] * df["male"]
    ))
    [["state", "eth", "male", "age", "educ", "draw", "chain", "p"]]
)

total_n_df = (
    poststrat_df
    .groupby("state")
    .agg({ "n" : "sum" })
    .rename(columns = { "n" : "total_n" })
    .reset_index()
)

weights_df = (
    poststrat_df
    .merge(total_n_df, how = "left")
    .assign(weight = lambda df: df["n"] / df["total_n"])
)

weighted_df = full_poststrat_df.merge(weights_df, how = "left", validate = "many_to_one")

states_weighted_p_df = (
    weighted_df
    .assign(weighted_p = lambda df : df["p"] * df["weight"])
    .groupby(["draw", "chain", "state"])
    .agg({ "weighted_p" : "sum" })
)

states_weighted_p_summary_df = (
    states_weighted_p_df
    .groupby("state")
    .agg(
        median = ("weighted_p", numpy.median),
        q10 = ("weighted_p", lambda x : numpy.quantile(x, 0.10)),
        q90 = ("weighted_p", lambda x : numpy.quantile(x, 0.90)),
    )
    .reset_index()
    .merge(state_df, how = "left")
)

plot = (
    plotnine.ggplot(states_weighted_p_summary_df) +
    plotnine.geom_pointrange(plotnine.aes(
        "reorder(state, repvote)", "median", ymin = "q10", ymax = "q90"
    )) +
    plotnine.ylim(0.0, 1.0) +
    plotnine.ylab("support")
)

plot.save("states.png", width = 12, height = 4, dpi = 200)
