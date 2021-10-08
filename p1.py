import pandas
import tokens

data_df = (
    pandas.read_csv("games_small.csv")
    .assign(score_diff=lambda df: (df.home_score - df.away_score).astype("float"))
    .assign(year=lambda df: df["date"].str[0:4].astype("int"))
)

tokenized_lines = [
    tokens.Normal(
        tokens.Data("score_diff"),
        tokens.Diff(
            tokens.Param("skills", tokens.Index(("home_team", "year"))),
            tokens.Param("skills", tokens.Index(("away_team", "year")))
        ),
        tokens.Param("sigma")
    ),
    tokens.Normal(
        tokens.Param("skills", tokens.Index(("team", "year"))),
        tokens.Param("skills_mu", tokens.Index("year")),
        tokens.Param("tau")
    ),
    tokens.Normal(
        tokens.Param("skills_mu", tokens.Index("year_mu")),
        tokens.Constant(0.0),
        tokens.Constant(1.0)
    ),
    tokens.Normal(
        tokens.Param("tau"),
        tokens.Constant(0.0),
        tokens.Constant(1.0)
    )
]

tree = parser.parse(data_df, tokenized_lines)
