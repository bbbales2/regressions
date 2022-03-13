library(tidyverse)
library(cmdstanr)

options(digits = 8)

df = read_csv("test_data_generation/time_series.csv")
mod = cmdstan_model("test_data_generation/time_series.stan")

fit = mod$optimize(data = list(
  N = nrow(df),
  y = df$y,
  i = df$i
))

fit$mle()
