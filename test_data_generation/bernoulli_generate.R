library(tidyverse)

inv_logit = function(x) { return(1 / (1 + exp(-x))); }

N = 150
G = 5
groups = sample(1:G, N, replace = TRUE)
mu = 1.7 - (1:G)
y = rbinom(N, 1, inv_logit(mu[groups]))

df = tibble(y = y, group = groups)

write_csv(df, "test_data_generation/bernoulli.csv")
