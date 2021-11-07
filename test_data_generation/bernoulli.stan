data {
  int N;
  int N_groups;
  int y[N];
  int group[N];
}

parameters {
  vector[N_groups] mu;
}

model {
  y ~ bernoulli_logit(mu[group]);
  mu ~ normal(-0.5, 0.3);
}
