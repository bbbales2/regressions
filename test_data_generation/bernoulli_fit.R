library(tidyverse)
library(cmdstanr)

options(digits = 8)

df = read_csv("test_data_generation/bernoulli.csv")
mod = cmdstan_model("test_data_generation/bernoulli.stan")

fit = mod$optimize(data = list(
  N = nrow(df),
  N_groups = max(df$group),
  y = df$y,
  group = df$group
))

fit$mle()
