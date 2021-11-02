library(tidyverse)
library(cmdstanr)

options(digits = 8)

df = read_csv("test_data_generation/normal.csv")
mod = cmdstan_model("test_data_generation/normal_mu.stan")

fit = mod$optimize(data = list(
  N = nrow(df),
  y = df$y
))

fit$mle()
