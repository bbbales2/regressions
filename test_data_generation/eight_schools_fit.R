library(tidyverse)
library(cmdstanr)

options(digits = 8)

df = read_csv("test_data_generation/eight_schools.csv")
mod = cmdstan_model("test_data_generation/eight_schools.stan")

fit = mod$optimize(data = list(
  J = nrow(df),
  y = df$y,
  sigma = df$sigma
))

fit$mle()

fit_sample = mod$sample(data = list(
  J = nrow(df),
  y = df$y,
  sigma = df$sigma
), chains = 4)

fit_sample$summary()
