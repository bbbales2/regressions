library(tidyverse)

N = 10

sigma = 0.1
mu = rep(0, N)
mu[1] = rnorm(1, mean = 0.0, sd = 0.3)
for(i in 2:N) {
  mu[i] = rnorm(1, mean = mu[i - 1], sd = 0.3)
}
y = rnorm(N, mu, sigma)

df = tibble(y = y, i = 1:N)
write_csv(df, "test_data_generation/time_series.csv")
