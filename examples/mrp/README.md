The model, data cleaning code, `poststrat_df.csv`, and `statelevel_predictors.csv` come from the
[Multilevel Regression and Poststratification Case Studies](https://bookdown.org/jl5522/MRP-case-studies/)
([Github](https://github.com/JuanLopezMartin/MRPCaseStudy)).

Specifically we're trying to rewrite the first rstanarm model from `01-mrp-intro.Rmd`. `cces_small.csv`
is the approximately N=5000 sample of data used in that part of the case study.

The CCES data comes from [here](`https://dataverse.harvard.edu/api/access/datafile/3588803?format=original&gbrecs=true`).
It's been re-encoded in parquet here to save space.

To run the examples, navigate to the `examples/mrp` folder and run:

```bash
python clean.py
rat mrp.rat clean.csv samples --num_draws=100
python plot.py
```

If you want to use just the N=5000 sample of data from the example, skip the first command and
replace `clean.csv` with `clean_small.csv` in the second command.

There is also a script `estimate.py` that does the equivalent of the middle command.

The number of draws is really low because my laptop runs out of memory in the poststratification
if I take many more.