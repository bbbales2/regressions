import blackjax
import blackjax.nuts
import blackjax.inference
import pandas
from typing import List, Dict


class Fit:
    draw_dfs: Dict[str, pandas.DataFrame]

    def __init__(self, draw_dfs: Dict[str, pandas.DataFrame]):
        self.draw_dfs = draw_dfs

    def draws(self, parameter_name: str) -> pandas.DataFrame:
        return self.draw_dfs[parameter_name]
