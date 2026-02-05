import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.trend import TrendTransformer


def _make_dataset(n: int = 5) -> MMMDataSet:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "y": range(n),
        }
    )
    return MMMDataSet.from_dataframe(df)


def test_trend_transformer_normalized():
    ds = _make_dataset(5)

    transformer = TrendTransformer(normalize=True)
    out, report = transformer.fit_transform(ds)

    assert "baseline__trend" in out.df.columns
    assert out.df["baseline__trend"].min() == 0.0
    assert out.df["baseline__trend"].max() == 1.0
    assert report.added_features == ["baseline__trend"]


def test_trend_transformer_not_normalized():
    ds = _make_dataset(4)

    transformer = TrendTransformer(normalize=False)
    out, _ = transformer.fit_transform(ds)

    assert list(out.df["baseline__trend"]) == [0.0, 1.0, 2.0, 3.0]


def test_trend_collision_raises():
    ds = _make_dataset(3)

    transformer = TrendTransformer()
    ds2, _ = transformer.fit_transform(ds)

    transformer2 = TrendTransformer()
    try:
        transformer2.fit_transform(ds2)
    except ValueError as e:
        assert "baseline__trend" in str(e)
    else:
        raise AssertionError("Expected ValueError on column collision")
