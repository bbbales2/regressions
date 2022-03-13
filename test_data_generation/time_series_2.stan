data {
  int N;
  int N_teams;
  int N_years;
  vector[N] score_diff;
  int team1[N];
  int team2[N];
  int year[N];
}

parameters {
  real skills[N_teams, N_years];
  real<lower = 0.0> sigma;
}

model {
  for(n in 1:N_teams) {
    skills[n, 1] ~ normal(0.0, 0.5);
    for(m in 2:N_years) {
      skills[n, m] ~ normal(skills[n, m - 1], 0.5);
    }
  }
  
  for(i in 1:N) {
    score_diff[i] ~ normal(skills[team1[i], year[i]] - skills[team2[i], year[i]], sigma);
  }
  
  sigma ~ normal(0.0, 1.0);
}
