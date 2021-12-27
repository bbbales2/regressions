from collections import defaultdict
import os
import pandas

mrp_folder = os.path.dirname(__file__)

cces_df = pandas.read_csv(os.path.join(mrp_folder, "cces18_common_vv.csv"))
fips_df = pandas.read_csv(os.path.join(mrp_folder, "fips.csv"))
state_df = pandas.read_csv(os.path.join(mrp_folder, "statelevel_predictors.csv"))

# This is a python version of the preprocessing done for the MRP case studies
# here: https://bookdown.org/jl5522/MRP-case-studies/downloading-and-processing-data.html

## Abortion -- dichotomous (0 - Oppose / 1 - Support)
abortion = (cces_df["CC18_321d"] - 2).abs().astype(pandas.Int64Dtype())
  
## State -- factor
fips = cces_df["inputstate"]
  
## Gender -- dichotomous (coded as -0.5 Female, +0.5 Male)
male = (cces_df["gender"] - 2.0).abs() - 0.5
  
## ethnicity -- factor
eth_map = defaultdict(lambda : "Other")
eth_map.update({ 1 : "White", 2 : "Black", 3 : "Hispanic" })
eth = cces_df["race"].map(eth_map)
  
## Age -- cut into factor
age = pandas.cut(
    2018 - cces_df["birthyr"].astype(int),
    bins = [0, 29, 39, 49, 59, 69, 120],
    labels = ["18-29", "30-39", "40-49", "50-59", "60-69", "70+"],
    ordered = True
)
  
## Education -- factor
educ_map = { 1 : "No HS", 2 : "HS", 3 : "Some college", 4 : "Some college", 5 : "4-Year College", 6 : "Post-grad" }
educ = cces_df["educ"].astype(int).map(educ_map)

clean_df = (
    pandas.DataFrame({
        "abortion" : abortion,
        "fips" : fips,
        "eth" : eth,
        "male" : male,
        "age" : age,
        "educ" : educ
    })
    .merge(fips_df, on = "fips", how = "left")
    .merge(state_df, on = "state", how = "left")
    .dropna()
)

clean_df.to_csv(os.path.join(mrp_folder, "clean.csv"), index = False)
