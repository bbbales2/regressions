data {
  int<lower=0> J;
  real y[J];
  real<lower=0> sigma[J];
}

parameters {
  real mu;
  real<lower=0> tau;
  vector[J] z;
}

transformed parameters {
  vector[J] theta = mu + tau * z;
}

model {
  mu ~ normal(0, 5);
  tau ~ lognormal(0, 1);
  z ~ normal(0, 1);
  y ~ normal(theta, sigma);
}