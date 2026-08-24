import pytest

from skrub import _dataframe as sbd
from skrub._single_column_transformer import RejectColumn
from skrub._to_categorical import ToCategorical


def test_to_categorical(df_module):
    s = df_module.make_column("c", ["a", "b", None])
    assert not sbd.is_categorical(s)
    out = ToCategorical().fit_transform(s)
    assert sbd.is_categorical(out)
    # categorial columns are accepted
    assert ToCategorical().fit_transform(out) is out
    assert ToCategorical().fit(out).transform(out) is out
    # default behaviour accepts integer and string
    # columns, but not float
    i = df_module.make_column("c", [1, 2, None])
    assert ToCategorical().fit_transform(i, accept_numeric="int")
    assert ToCategorical().fit(i).transform(i, accept_numeric="int")
    # unless accept_numeric is None, in which case
    # only string and categorical columns are accepted
    with pytest.raises(RejectColumn, match=".*does not contain strings or*"):
        ToCategorical().fit_transform(i, accept_numeric=None)
    # if accept_numeric is set to "all", then both integer and
    # float columns are accepted
    f = df_module.make_column("c", [1.1, 2.2, None])
    assert ToCategorical().fit_transform(f, accept_numeric="all")
    # if accept_numeric != "all", then float columns are rejected
    with pytest.raises(RejectColumn, match=".*does not contain strings or*"):
        ToCategorical().fit_transform(f)
    # but once accepted during fit, transform works on any column
    assert sbd.is_categorical(ToCategorical().fit(s).transform(f))
