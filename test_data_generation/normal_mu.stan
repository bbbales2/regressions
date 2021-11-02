data {
  int N;
  vector[N] y;
}

parameters {
  real mu;
}

model {
  y ~ normal(mu, 1.5);
  mu ~ normal(-0.5, 0.3);
}
