from rat.fit import load
import pandas
import plotnine

players_df = pandas.read_csv("examples/fakeball/players.csv").rename(columns = {"offense" : "reference_offense"})
shots_df = pandas.read_csv("examples/fakeball/shots.csv")

optimum = load("examples/fakeball/samples")

fit_offense_df = optimum.draws("offense").groupby("player").agg({"offense": "mean"})

(
    plotnine.ggplot(players_df.merge(fit_offense_df, on="player", how="left"))
    + plotnine.geom_point(plotnine.aes("offense", "reference_offense"))
    + plotnine.geom_abline(intercept=0, slope=1, color="red", linetype="dashed")
).draw(show=True)
