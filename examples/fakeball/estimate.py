import os
import pandas
from rat.model import Model

fakeball_folder = os.path.dirname(__file__)

shots_df = pandas.read_csv(os.path.join(fakeball_folder, "shots.csv"))

with open(os.path.join(mrp_folder, "fakeball.rat")) as f:
    model_string = f.read()

model = Model(shots_df, model_string=model_string)

print("Running MCMC")
fit = model.sample(num_warmup=1000, num_draws=1000)

print("Writing output")
fit.save(os.path.join(fakeball_folder, "samples"), overwrite=True)
