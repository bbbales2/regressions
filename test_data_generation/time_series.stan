data {
  int N;
  vector[N] y;
  int i[N];
}

parameters {
  vector[N] mu;
}

model {
  mu[i[1]] ~ normal(0.0, 0.3);
  for(n in 2:N) {
    mu[i[n]] ~ normal(mu[i[n - 1]], 0.3);
  }

  y ~ normal(mu, 0.1);
}
