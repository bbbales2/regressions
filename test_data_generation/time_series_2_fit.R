library(tidyverse)
library(cmdstanr)

options(digits = 8)

df = read_csv("test_data_generation/time_series_2.csv")
mod = cmdstan_model("test_data_generation/time_series_2.stan")

fit = mod$optimize(data = list(
  N = nrow(df),
  N_teams = max(df$team1, df$team2),
  N_years = max(df$year),
  score_diff = df$score_diff,
  team1 = df$team1,
  team2 = df$team2,
  year = df$year
))

fit$mle()
