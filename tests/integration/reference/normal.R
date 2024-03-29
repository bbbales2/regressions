library(tidyverse)
library(cmdstanr)

df = read_csv("tests/integration/normal.csv")

df %>% group_by(group) %>% summarize(m = mean(y))

data = list(
  N = nrow(df),
  G = max(df$group),
  group = df$group,
  y = df$y
)

model = cmdstan_model("tests/integration/reference/normal.stan")

fit = model$optimize(data = data)

fit$print(digits = 5)
