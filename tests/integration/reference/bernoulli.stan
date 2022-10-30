data {
  int N;
  int G;
  array[N] int y;
  array[N] int<lower=1, upper=G> group;
}

parameters {
  vector[G] theta;
}

model {
  theta ~ normal(-0.5, 0.3);
  for(n in 1:N) {
    y[n] ~ bernoulli_logit(theta[group[n]]);
  }
}