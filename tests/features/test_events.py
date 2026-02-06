import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.events import EventTransformer


def _make_dataset(n: int = 7) -> MMMDataSet:
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=n, freq="D"),
            "target__sales": range(n),
        }
    )
    return MMMDataSet.from_dataframe(df)


def test_event_transformer_single_list():
    ds = _make_dataset(5)

    transformer = EventTransformer(dates=["2023-01-02", "2023-01-05"])
    out, report = transformer.fit_transform(ds)

    assert "event__event" in out.df.columns
    assert list(out.df["event__event"]) == [0.0, 1.0, 0.0, 0.0, 1.0]
    assert report.added_features == ["event__event"]


def test_event_transformer_dict_multiple_events():
    ds = _make_dataset(5)

    transformer = EventTransformer(
        events={
            "promo": ["2023-01-01", "2023-01-03"],
            "launch": ["2023-01-05"],
        }
    )
    out, report = transformer.fit_transform(ds)

    assert "event__promo" in out.df.columns
    assert "event__launch" in out.df.columns
    assert list(out.df["event__promo"]) == [1.0, 0.0, 1.0, 0.0, 0.0]
    assert list(out.df["event__launch"]) == [0.0, 0.0, 0.0, 0.0, 1.0]
    # deterministic order (launch then promo alphabetically)
    assert report.added_features == ["event__launch", "event__promo"]


def test_event_invalid_name_raises():
    ds = _make_dataset(3)

    transformer = EventTransformer(events={"Bad-Name": ["2023-01-01"]})
    try:
        transformer.fit(ds)
    except ValueError as e:
        assert "snake_case" in str(e)
    else:
        raise AssertionError("Expected ValueError for invalid event name")


def test_event_collision_raises():
    ds = _make_dataset(3)

    t1 = EventTransformer(dates=["2023-01-01"])
    ds2, _ = t1.fit_transform(ds)

    t2 = EventTransformer(dates=["2023-01-02"])
    try:
        t2.fit_transform(ds2)
    except ValueError as e:
        assert "event__event" in str(e)
    else:
        raise AssertionError("Expected ValueError on column collision")


def test_event_dates_outside_dataset_are_ignored():
    ds = _make_dataset(3)

    transformer = EventTransformer(dates=["2022-12-31", "2023-01-02"])
    out, _ = transformer.fit_transform(ds)

    assert list(out.df["event__event"]) == [0.0, 1.0, 0.0]
