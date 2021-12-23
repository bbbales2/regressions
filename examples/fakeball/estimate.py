import pandas
from rat.model import Model

shots_df = pandas.read_csv("examples/fakeball/shots.csv")

model_string = """
made ~ bernoulli_logit(offense[o0, date] + offense[o1, date] + offense[o2, date] + offense[o3, date] + offense[o4, date] - (defense[d0, date] + defense[d1, date] + defense[d2, date] + defense[d3, date] + defense[d4, date]));
offense[player, date] = offense_rw[player, date] + offense0[player];
defense[player, date] = defense_rw[player, date] + defense0[player];
offense0[player] ~ normal(0.0, tau0_offense);
defense0[player] ~ normal(0.0, tau0_defense);
offense_rw[player, date] ~ normal(offense[player, shift(date, 1)], tau_offense);
defense_rw[player, date] ~ normal(defense[player, shift(date, 1)], tau_defense);
tau_offense<lower = 0.0> ~ log_normal(0.0, 0.5);
tau_defense<lower = 0.0> ~ log_normal(0.0, 0.5);
tau0_offense<lower = 0.0> ~ log_normal(0.0, 0.5);
tau0_defense<lower = 0.0> ~ log_normal(0.0, 0.5);
"""

model = Model(shots_df, model_string=model_string)

#print("Running optimization")
#fit = model.optimize(chains = 1)
#print("Writing output")
#fit.save("examples/fakeball/optimum", overwrite = True)

print("Running MCMC")
fit = model.sample(num_warmup = 1000, num_draws = 1000)
print("Writing output")
fit.save("examples/fakeball/samples", overwrite = True)
