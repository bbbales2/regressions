library(tidyverse)

N = 150
G = 5
groups = sample(1:G, N, replace = TRUE)
mu = 1.7 - (1:G)
sd = 2.3 + 0.5 * (1:G)
y = rnorm(N, mu[groups], sd[groups])

df = tibble(y = y, group = groups)

write_csv(df, "test_data_generation/normal.csv")
