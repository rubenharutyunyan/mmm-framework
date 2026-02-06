# Feature Engineering — Core (V1)

## Scope & position in data flow

This document defines the **Feature Engineering — Core (V1)** layer of the `mmm-framework`.

Position in the data flow:

Raw client data  
→ ColumnMapper (mapping v1)  
→ MMMDataSet (validated)  
→ **Feature Engineering — Core (V1)**  
→ EDA / Modeling

This layer is responsible for generating **non-media explanatory features** required by a Marketing Mix Model (MMM), without introducing any modeling logic.

---

## Design principles

Feature Engineering — Core follows these principles:

- Strict compliance with `02_NAMING_CONVENTION.md`
- No modification of existing dataset columns
- Deterministic and reproducible feature generation
- Full traceability of generated features
- No dependency on modeling or inference logic
- Output dataset must pass `MMMDataSet.from_dataframe(...)`

---

## Minimal end-to-end example

```python
import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.features.trend import TrendTransformer
from mmm.features.seasonality import SeasonalityTransformer
from mmm.features.events import EventTransformer
from mmm.features.pipeline import FeaturePipeline

## Input dataset

df = pd.DataFrame(
    {
        "date": pd.date_range("2023-01-01", periods=7, freq="D"),
        "target__sales": [10, 12, 11, 13, 15, 14, 16],
    }
)

dataset = MMMDataSet.from_dataframe(df)

## Feature pipeline

pipeline = FeaturePipeline(
    transformers=[
        TrendTransformer(normalize=True),
        SeasonalityTransformer(period=7, order=2),
        EventTransformer(
            events={
                "promo": ["2023-01-03", "2023-01-06"]
            }
        ),
    ]
)
```

---

#### Run feature engineering

```python
enriched_dataset, feature_report = pipeline.run(dataset)
```

---

#### Resulting dataset columns

```python
enriched_dataset.df.columns.tolist()
```

Example output :

```text
[
  "date",
  "target__sales",
  "baseline__trend",
  "baseline__seasonality__fourier__p7__k1__sin",
  "baseline__seasonality__fourier__p7__k1__cos",
  "baseline__seasonality__fourier__p7__k2__sin",
  "baseline__seasonality__fourier__p7__k2__cos",
  "event__promo"
]
```

---

#### Feature traceability

```python
feature_report.to_dict()
```

Example output :

```json
{
  "steps": [
    {
      "transformer": "TrendTransformer",
      "params": {
        "date_col": "date",
        "normalize": true,
        "col_name": "baseline__trend"
      },
      "added_features": ["baseline__trend"],
      "notes": null
    },
    {
      "transformer": "SeasonalityTransformer",
      "params": {
        "period": 7,
        "order": 2,
        "date_col": "date"
      },
      "added_features": [
        "baseline__seasonality__fourier__p7__k1__sin",
        "baseline__seasonality__fourier__p7__k1__cos",
        "baseline__seasonality__fourier__p7__k2__sin",
        "baseline__seasonality__fourier__p7__k2__cos"
      ],
      "notes": null
    },
    {
      "transformer": "EventTransformer",
      "params": {
        "date_col": "date",
        "default_event_name": "event",
        "events": ["promo"]
      },
      "added_features": ["event__promo"],
      "notes": null
    }
  ]
}
```
---

### Examples by transformer

#### TrendTransformer

```python
TrendTransformer(normalize=True)
```
Creates :  
* `baseline__trend` ∈ [0, 1]

#### SeasonalityTransformer (Fourier)

```python
SeasonalityTransformer(period=365, order=3)
```
Creates :  
* `baseline__seasonality__fourier__p365__k1__sin`
* `baseline__seasonality__fourier__p365__k1__cos`
* `baseline__seasonality__fourier__p365__k2__sin`
* `baseline__seasonality__fourier__p365__k2__cos`
* `baseline__seasonality__fourier__p365__k3__sin`
* `baseline__seasonality__fourier__p365__k3__cos`

#### EventsTransformer (single event)

```python
EventTransformer(dates=["2023-01-01", "2023-01-15"])
```
Creates :  
* `event__event`

#### EventTransformer (multiple events)

```python
EventTransformer(
    events={
        "promo": ["2023-01-03"],
        "launch": ["2023-01-10"]
    }
)
```
Creates :  
* `event__promo`
* `event__launch`

### Common error cases

#### Column collision

```python
ValueError: Column already exists: baseline__trend
```
Raised when a transformer attempts to create a column already present in the dataset.

#### Invalid event name
```python
ValueError: Invalid event name (must be snake_case): Black-Friday
```
Event names must be valid snake_case and compatible with the naming convention.

---

## V1 — Included features

### 1. Trend (Baseline)

#### Transformer
`TrendTransformer`

#### Description
Creates a simple linear time trend based on the temporal order of the dataset.

#### Generated feature
- `baseline__trend`

#### Properties
- Numeric
- Monotonic
- Optionally normalized to `[0, 1]`

#### Notes
- Stateless transformer
- Exactly one column is created
- Collision with an existing column raises an error

---

### 2. Seasonality (Fourier)

#### Transformer
`SeasonalityTransformer`

#### Method
Fourier series expansion

#### Parameters
- `period`: seasonality period (e.g. 7, 365)
- `order`: number of harmonics

#### Generated features
For each harmonic `k ∈ [1, order]`:

- `baseline__seasonality__fourier__p{period}__k{k}__sin`
- `baseline__seasonality__fourier__p{period}__k{k}__cos`

#### Example
For `period=7`, `order=2`:

- `baseline__seasonality__fourier__p7__k1__sin`
- `baseline__seasonality__fourier__p7__k1__cos`
- `baseline__seasonality__fourier__p7__k2__sin`
- `baseline__seasonality__fourier__p7__k2__cos`

#### Properties
- Numeric
- Deterministic
- Independent of target or media features

---

### 3. Events

#### Transformer
`EventTransformer`

#### Description
Creates binary event indicators from user-provided dates.

#### Supported inputs
- A list of dates
- A dictionary `{event_name: [dates]}`

#### Generated features
- `event__event` (single list input)
- `event__<event_name>` (dictionary input)

#### Properties
- Values are strictly `0` or `1`
- Event names must be valid `snake_case`
- Dates outside the dataset range are ignored
- Column name collisions raise an error

---

## FeaturePipeline

#### Component
`FeaturePipeline`

#### Role
Orchestrates an ordered list of feature transformers.

#### Responsibilities
- Applies transformers sequentially
- Aggregates feature traceability
- Re-validates the final dataset

#### Output
- Enriched `MMMDataSet`
- Aggregated `FeatureReport`

---

## FeatureReport

#### Purpose
Ensures minimal but explicit traceability of feature generation.

#### Contents
For each transformation step:
- Transformer name
- Parameters used
- Generated feature names

#### Guarantees
- Ordered execution trace
- Reproducibility of feature construction
- Debug-friendly feature lineage

---

## Guarantees (V1)

* No existing column is modified
* All generated features follow `02_NAMING_CONVENTION.md`
* Final dataset is always validated via `MMMDataSet.from_dataframe`
* Feature generation is deterministic and reproducible

---

## V1 — Explicit exclusions

The following are **out of scope** for Feature Engineering — Core (V1):

- Media transformations (adstock, saturation)
- Automatic feature selection
- Holiday calendars with complex rules
- Any modeling or Bayesian logic
- Interaction or cross-features

These may be introduced in future versions.

---

## Versioning

- This document describes **Feature Engineering — Core V1**
- Any breaking change must be introduced in **V2**
