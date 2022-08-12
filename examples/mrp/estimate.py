import os
import pandas
from rat.model import Model

mrp_folder = os.path.dirname(__file__)

mrp_df = pandas.read_csv(os.path.join(mrp_folder, "clean_small.csv"))

with open(os.path.join(mrp_folder, "mrp.rat")) as f:
    model_string = f.read()

model = Model(model_string=model_string, data=mrp_df)

print("Running MCMC")
fit = rat.sample(model, num_warmup=1000, num_draws=100)

print("Writing output")
fit.save(os.path.join(mrp_folder, "samples"), overwrite=True)
