import pandas as pd
import pytest

from mmm.data.dataset import MMMDataSet
from mmm.data.validation import ValidationError


def test_dataset_ok():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-08"],
            "target__sales": [100, 120],
            "media__tv__spend": [10.0, 12.0],
            "event__promo": [0, 1],
            "control__price": [1.0, 1.1],
        }
    )
    ds = MMMDataSet.from_dataframe(df, freq="W")
    assert ds.columns_by_role("media") == ["media__tv__spend"]


def test_dataset_negative_media_raises():
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-08"],
            "target__sales": [100, 120],
            "media__tv__spend": [10.0, -1.0],
        }
    )
    with pytest.raises(ValidationError):
        MMMDataSet.from_dataframe(df)
