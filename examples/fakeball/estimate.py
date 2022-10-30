import os
import pandas
from rat.model import Model

fakeball_folder = os.path.dirname(__file__)

shots_df = pandas.read_csv(os.path.join(fakeball_folder, "shots.csv"))

with open(os.path.join(fakeball_folder, "fakeball.rat")) as f:
    model_string = f.read()

model = Model(model_string=model_string, data=shots_df)

print("Running MCMC")
fit = rat.sample(model, num_warmup=1000, num_draws=1000)

print("Writing output")
fit.save(os.path.join(fakeball_folder, "samples"), overwrite=True)
