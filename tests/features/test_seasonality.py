import numpy as np
import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.seasonality import SeasonalityTransformer


def _make_dataset(n: int = 10) -> MMMDataSet:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "target__sales": np.arange(n),
        }
    )
    return MMMDataSet.from_dataframe(df)


def test_seasonality_fourier_columns_created():
    ds = _make_dataset(10)

    transformer = SeasonalityTransformer(period=7, order=2)
    out, report = transformer.fit_transform(ds)

    expected = [
        "baseline__seasonality__fourier__p7__k1__sin",
        "baseline__seasonality__fourier__p7__k1__cos",
        "baseline__seasonality__fourier__p7__k2__sin",
        "baseline__seasonality__fourier__p7__k2__cos",
    ]

    for col in expected:
        assert col in out.df.columns

    assert report.added_features == expected


def test_seasonality_values_are_finite():
    ds = _make_dataset(15)

    transformer = SeasonalityTransformer(period=7, order=3)
    out, _ = transformer.fit_transform(ds)

    cols = [
        c for c in out.df.columns
        if c.startswith("baseline__seasonality__fourier")
    ]

    for col in cols:
        assert np.isfinite(out.df[col]).all()


def test_seasonality_collision_raises():
    ds = _make_dataset(8)

    t1 = SeasonalityTransformer(period=7, order=1)
    ds2, _ = t1.fit_transform(ds)

    t2 = SeasonalityTransformer(period=7, order=1)
    try:
        t2.fit_transform(ds2)
    except ValueError as e:
        assert "already exists" in str(e)
    else:
        raise AssertionError("Expected ValueError on seasonality collision")
