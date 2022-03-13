library(tidyverse)

#score_diff ~ normal(skills[team1, year] - skills[team2, year], sigma);
#skills[team, year] ~ normal(skills[team, lag(year)], 0.5);
#sigma<lower = 0.0> ~ normal(0, 1.0);

N_teams = 3
N_years = 5
N_games_per_year = 100

skills = matrix(0, nrow = N_teams, ncol = N_years)
for(i in 1:N_teams) {
  skills[i,] = sin((1:N_years) + i)
}
sigma = 3.5

team1s = c()
team2s = c()
years = c()
score_diffs = c()
for(j in 1:N_years) {
  for(i in 1:N_games_per_year) {
    game = sample(1:N_teams, 2)
    team1 = game[1]
    team2 = game[2]
    score_diff = rnorm(1, skills[team1, j] - skills[team2, j], sigma)
    
    years = c(years, j)
    team1s = c(team1s, team1)
    team2s = c(team2s, team2)
    score_diffs = c(score_diffs, score_diff)
  }
}

df = tibble(score_diff = score_diffs, year = years, team1 = team1s, team2 = team2s)
write_csv(df, "test_data_generation/time_series_2.csv")
