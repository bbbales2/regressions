import pandas
import pytest
from rat import variables


@pytest.fixture
def index():
    group1 = [2, 2, 1, 1, 1, 1]
    group2 = [2, 1, 1, 2, 3, 3]
    unprocessed_df = pandas.DataFrame(zip(group1, group2), columns=("group1", "group2"))
    return variables.Index(unprocessed_df, [{"group1"}, {"group2"}])


def test_index_constructor(index):
    assert len(index.df) == 5
    assert index.df["group1"].to_list() == [1, 1, 1, 2, 2]
    assert index.df["group2"].to_list() == [1, 2, 3, 1, 2]


def test_incorporate_shifts(index):
    index.incorporate_shifts((1, 1))
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

    index.incorporate_shifts((None, 1))
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

    index.incorporate_shifts((2, 1))
    assert index.df["group1"].to_list()[:-4] == [1, 1, 1, 2, 2]
    assert index.df["group1"].to_list()[-3:-1] == [1, 2]
    assert index.df["group2"].to_list()[:-4] == [1, 2, 3, 1, 2]
    assert index.df["group2"].to_list()[-1:] == [1]
    assert index.df["group1"].isna().to_list() == [
        False,
        False,
        False,
        False,
        False,
        True,
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
        True,
        True,
        False,
    ]


def test_get_numpy_indices(index):
    # def get_numpy_indices(self, df):
    pass


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    pytest.main([__file__, "-s", "-o", "log_cli=true"])
