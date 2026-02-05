import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.events import EventTransformer
from mmm.features.pipeline import FeaturePipeline
from mmm.features.trend import TrendTransformer


def _make_dataset(n: int = 5) -> MMMDataSet:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "target__sales": range(n),
        }
    )
    return MMMDataSet.from_dataframe(df)


def test_feature_pipeline_sequential_application():
    ds = _make_dataset(5)

    pipeline = FeaturePipeline(
        transformers=[
            TrendTransformer(),
            EventTransformer(dates=["2023-01-02"]),
        ]
    )

    out, report = pipeline.run(ds)

    assert "baseline__trend" in out.df.columns
    assert "event__event" in out.df.columns

    # original column untouched
    assert "target__sales" in out.df.columns

    # traceability
    assert report.added_features == ["baseline__trend", "event__event"]


def test_feature_pipeline_order_matters():
    ds = _make_dataset(3)

    pipeline = FeaturePipeline(
        transformers=[
            EventTransformer(dates=["2023-01-01"]),
            TrendTransformer(),
        ]
    )

    out, report = pipeline.run(ds)

    assert report.added_features == ["event__event", "baseline__trend"]
