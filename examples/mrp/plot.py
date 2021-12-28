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


def grab(name):
    return fit.draws(name).rename(columns={"value": name})


variables = ["state", "eth", "male", "age", "educ"]

full_poststrat_df = (
    poststrat_df.merge(grab("a_age"), how="outer")  # First is an outer so poststrat table gets chain/draw
    .merge(grab("a_state"), how="left")  # The rest can be lefts
    .merge(grab("a_eth"), how="left")
    .merge(grab("a_educ"), how="left")
    .merge(grab("a_male_eth"), how="left")
    .merge(grab("a_educ_age"), how="left")
    .merge(grab("a_educ_eth"), how="left")
    .merge(grab("b_repvote"), how="left")
    .merge(grab("b_male"), how="left")
    .merge(state_df[["state", "repvote"]], how="left")
    .assign(
        p=lambda df: scipy.special.expit(
            df["a_state"]
            + df["a_age"]
            + df["a_eth"]
            + df["a_educ"]
            + df["a_male_eth"]
            + df["a_educ_age"]
            + df["a_educ_eth"]
            + df["b_repvote"] * df["repvote"]
            + df["b_male"] * df["male"]
        )
    )[["state", "eth", "male", "age", "educ", "draw", "chain", "p"]]
)


def poststratify(variable, groupby=[]):
    if len(groupby) == 0:
        weights_df = poststrat_df.assign(weight=lambda df: df["n"] / df["n"].sum())
    else:
        total_df = poststrat_df.groupby(groupby).agg({"n": "sum"}).rename(columns={"n": "total_n"}).reset_index()

        weights_df = poststrat_df.merge(total_df, how="left").assign(weight=lambda df: df["n"] / df["total_n"])

    weighted_df = full_poststrat_df.merge(weights_df, how="left", validate="many_to_one")

    return (
        weighted_df.assign(weighted_variable=lambda df: df[variable] * df["weight"])
        .groupby(["draw", "chain"] + groupby)
        .agg({"weighted_variable": "sum"})
        .rename(columns={"weighted_variable": variable})
    )


states_p_df = poststratify("p", ["state"])

states_p_summary_df = (
    states_p_df.groupby("state")
    .agg(
        median=("p", numpy.median),
        q10=("p", lambda x: numpy.quantile(x, 0.10)),
        q90=("p", lambda x: numpy.quantile(x, 0.90)),
    )
    .reset_index()
    .merge(state_df, how="left")
)

plot = (
    plotnine.ggplot(states_p_summary_df)
    + plotnine.geom_pointrange(plotnine.aes("reorder(state, repvote)", "median", ymin="q10", ymax="q90"))
    + plotnine.ylim(0.0, 1.0)
    + plotnine.ylab("support")
)

plot.save("states.png", width=12, height=4, dpi=200)
