data {
  int N;
  vector[N] y;
}

parameters {
  real mu;
  real<lower = 0.0> sigma;
}

model {
  y ~ normal(mu, sigma);
  mu ~ normal(-0.5, 0.3);
  sigma ~ normal(0.0, 0.7);
}
