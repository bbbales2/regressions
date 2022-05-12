import logging
import os
import pathlib
import pandas
import pytest

from rat import ast
from rat.model import Model

test_dir = pathlib.Path(__file__).parent

def test_lines_in_wrong_order_for_assignments():
    data_df = pandas.read_csv(os.path.join(test_dir, "eight_schools.csv"))

    model_string = """
    y' ~ normal(theta[school], sigma);
    theta' = mu + z[school] * tau;
    z ~ normal(0, 1);
    mu ~ normal(0, 5);
    tau<lower = 0.0> ~ log_normal(0, 1);
    school_id[school] = ifelse(school' == 1, 
                         1,
                         ifelse(school == 2,
                                2,
                                ifelse(school == 3,
                                       3,
                                       ifelse(school == 4,
                                              4,
                                              ifelse(school == 5,
                                                     5,
                                                     ifelse(school == 6,
                                                            6,
                                                            ifelse(school == 7,
                                                                   7,
                                                                   ifelse(school == 8,
                                                                          8,
                                                                          -1
                                                                   )
                                                            )
                                                    )
                                              )
                                       )
                                )
                         )
                 );
    """

    model = Model(data_df, model_string)
    fit = model.sample()

    school_id = fit.draws("school_id")

    for x in range(1, 9):
        assert school_id[school_id.school == x].school_id.mean() == x


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
