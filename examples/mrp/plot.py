from rat.fit import load
import os
import pandas
import plotnine

mrp_folder = os.path.dirname(__file__)

fit = load(os.path.join(mrp_folder, "samples"))
