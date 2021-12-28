from rat.fit import load
import numpy
import os
import pandas
import plotnine

fakeball_folder = os.path.dirname(__file__)

players_df = pandas.read_csv(os.path.join(fakeball_folder, "players.csv"))
shots_df = pandas.read_csv(os.path.join(fakeball_folder, "shots.csv"))

fit = load(os.path.join(fakeball_folder, "samples"))

fit_offense_df = (
    fit.draws("offense")
    .groupby("player")
    .agg(
        median=("offense", numpy.median),
        q10=("offense", lambda x: numpy.quantile(x, 0.1)),
        q90=("offense", lambda x: numpy.quantile(x, 0.9)),
    )
)

plot = (
    plotnine.ggplot(players_df.merge(fit_offense_df, on="player", how="left"))
    + plotnine.geom_pointrange(plotnine.aes("offense", "median", ymin="q10", ymax="q90"))
    + plotnine.geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
    + plotnine.ylab("Estimated offensive ability (10%/median/90%)")
    + plotnine.xlab("Actual offensive ability")
)

plot.save(os.path.join(fakeball_folder, "offense.png"), width=8, height=6, dpi=200)
