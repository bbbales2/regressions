import pandas
import pytest
from rat import variables


@pytest.fixture
def index():
    group1 = [2, 2, 1, 1, 1, 1]
    group2 = [2, 1, 1, 2, 3, 3]
    unprocessed_df = pandas.DataFrame(zip(group1, group2), columns=("group1", "group2"))
    return variables.Index(unprocessed_df)


def test_index_constructor(index):
    assert len(index.df) == 5
    assert index.df["group1"].to_list() == [1, 1, 1, 2, 2]
    assert index.df["group2"].to_list() == [1, 2, 3, 1, 2]


def test_incorporate_shifts(index):
    index.incorporate_shifts(("group1", "group2"), 1)
    assert index.df["group1"].to_list()[:-1] == [1, 1, 1, 2, 2]
    assert index.df["group2"].to_list()[:-1] == [1, 2, 3, 1, 2]
    assert index.df["group1"].isna().to_list() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]
    assert index.df["group2"].isna().to_list() == [
        False,
        False,
        False,
        False,
        False,
        True,
    ]

    index.incorporate_shifts(("group2",), 1)
    assert index.df["group1"].to_list()[:-3] == [1, 1, 1, 2, 2]
    assert index.df["group1"].to_list()[-2:] == [1, 2]
    assert index.df["group2"].to_list()[:-3] == [1, 2, 3, 1, 2]
    assert index.df["group1"].isna().to_list() == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
        False,
    ]
    assert index.df["group2"].isna().to_list() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_compute_shifted_df(index):
    # def compute_shifted_df(self, df, shift_columns, shift):
    pass


def test_rebuild_df(index):
    # def rebuild_df(self):
    pass


def test_get_numpy_indices(index):
    # def get_numpy_indices(self, df):
    pass


if __name__ == "__main__":
    pytest.main([__file__])
