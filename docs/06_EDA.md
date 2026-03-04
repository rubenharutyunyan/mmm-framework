# EDA — Exploratory Data Analysis (v1)

## Scope & position in data flow

This document defines the **EDA (v1)** layer of the `mmm-framework`.

Position in the data flow:

Feature Engineering (`features/`)
→ **EDA & sanity checks (`eda/`)**
→ Bayesian modeling (`modeling/`)

The EDA layer performs **read-only analysis** of a validated `MMMDataSet`. It produces structured diagnostic reports to detect potential modeling issues (multicollinearity, weak signal) before any Bayesian inference is run.

---

## Design principles

- **Read-only**: the input `MMMDataSet` is never modified. No column is created, deleted, renamed, or mutated. Any in-memory transformation (e.g., centering for VIF computation) is performed on transient numpy arrays and never written back to the dataset.
- **Deterministic**: identical inputs always produce identical outputs
- **No modeling logic**: no Bayesian inference, no prior definition, no posterior sampling
- **No plotting**: all outputs are structured data objects; visualization is out of scope for v1
- **Structured outputs**: each analysis returns a frozen dataclass, not a raw DataFrame
- **Naming-aware**: column roles are inferred via `02_NAMING_CONVENTION.md`

---

## Variable selection

All three analyses share the same predictor selection logic, applied in the following priority order:

1. **`columns` provided** → use exactly those columns as predictors (must all exist in the dataset)
2. **`roles` provided** → select all columns whose role prefix matches one of the given roles (e.g., `["media", "control"]`)
3. **Default** → select all columns with roles `baseline`, `control`, `event`, `media`

In all cases:
- `date` is always excluded
- `target_col` is always excluded from predictors
- The resolved predictor set must be non-empty; an empty predictor set raises a `ValueError`

If `columns` is provided, values are used as-is without role inference. Each column must be present in `dataset.df` and must not be `date` or `target_col`.

If `roles` is provided, each role must be one of the allowed roles defined in `02_NAMING_CONVENTION.md`.

---

## V1 — Included analyses

### 1. Correlation matrix

#### Component
`compute_correlation(dataset, *, columns=None, roles=None)`

#### Description
Computes pairwise Pearson correlations between the selected predictor columns in the dataset.

#### Input
- `dataset: MMMDataSet`
- `columns: list[str] | None` — explicit list of columns to include (see Variable selection)
- `roles: list[str] | None` — list of roles to filter by (see Variable selection)

#### Output
`CorrelationReport` — frozen dataclass:
- `matrix: pd.DataFrame` — shape `(n_cols, n_cols)`, symmetric, values in `[-1, 1]`
- `columns: list[str]` — column names in the same order as matrix rows/columns

#### Notes
- Predictor set is resolved following the Variable selection rules above
- Column order follows resolution order (explicit list, role-filtered, or default)
- NaN values in the correlation matrix are not possible if the dataset contract is met

---

### 2. Variance Inflation Factor (VIF)

#### Component
`compute_vif(dataset, target_col, *, columns=None, roles=None)`

#### Description
Computes the Variance Inflation Factor for each predictor column.
VIF measures how much variance of a coefficient is inflated due to collinearity with other predictors.

For each predictor `j`, VIF is computed as:

```
VIF_j = 1 / (1 - R²_j)
```

where `R²_j` is the coefficient of determination of regressing column `j` on all other predictors (excluding `target_col` and `date`).

#### Input
- `dataset: MMMDataSet`
- `target_col: str` — name of the target column (always excluded from predictors)
- `columns: list[str] | None` — explicit list of predictor columns (see Variable selection)
- `roles: list[str] | None` — list of roles to filter by (see Variable selection)

#### Output
`VIFReport` — frozen dataclass:
- `scores: dict[str, float]` — `{column_name: vif_value}` for each predictor; `np.inf` when perfect collinearity is detected
- `warnings: list[str]` — one entry per column where VIF was set to `np.inf`
- `target_col: str` — target column used

#### Interpretation thresholds (informational, not enforced)
- VIF < 5: low collinearity
- VIF in [5, 10]: moderate collinearity, investigate
- VIF > 10: high collinearity, likely problematic
- VIF = `np.inf`: perfect collinearity detected

#### Notes
- Computed using numpy (no statsmodels dependency)
- `target_col` must be present in `dataset.df` and must follow naming convention v1
- Predictor set is resolved following the Variable selection rules above
- A constant column (zero variance) raises a `ValueError`
- If fewer than 2 predictors are present, VIF cannot be computed; a `ValueError` is raised
- If R² ≥ 1 − 1e-12 for a column, VIF is set to `np.inf` and a warning is recorded in `VIFReport.warnings` (no silent failure)
- If the predictor matrix is singular and regression cannot be completed, a `ValueError` is raised with an explicit message

---

### 3. Ridge sanity check

#### Component
`compute_ridge_sanity(dataset, target_col, alpha=1.0, *, columns=None, roles=None)`

#### Description
Fits a Ridge regression (L2-regularized OLS) using sklearn, with the resolved predictor columns as features and `target_col` as the dependent variable.

This provides a quick frequentist sanity check:
- Do predictor signs match expectations?
- Is overall fit reasonable before Bayesian modeling?

#### Input
- `dataset: MMMDataSet`
- `target_col: str` — name of the target column (must be explicitly provided, always excluded from predictors)
- `alpha: float` — Ridge regularization strength (default: `1.0`)
- `columns: list[str] | None` — explicit list of predictor columns (see Variable selection)
- `roles: list[str] | None` — list of roles to filter by (see Variable selection)

#### Output
`RidgeSanityReport` — frozen dataclass:
- `coefficients: dict[str, float]` — `{column_name: coefficient}` for each predictor
- `intercept: float`
- `r2_score: float` — in-sample R²
- `target_col: str`
- `alpha: float`

#### Notes
- `target_col` must be explicitly provided; there is no default inference of the target
- Predictor set is resolved following the Variable selection rules above
- No train/test split in v1 (in-sample R² only)
- No feature scaling applied (raw values as in the dataset)
- `sklearn` is required for this component

---

## EDARunner

#### Component
`EDARunner`

#### Role
Convenience class that runs all three analyses in sequence and aggregates results into a single `EDAReport`.

#### Constructor
```python
EDARunner(
    target_col: str,
    ridge_alpha: float = 1.0,
    *,
    columns: list[str] | None = None,
    roles: list[str] | None = None,
)
```

`columns` and `roles` are forwarded to all three analyses unchanged.

#### Method
```python
runner.run(dataset: MMMDataSet) -> EDAReport
```

#### Output
`EDAReport` — frozen dataclass:
- `correlation: CorrelationReport`
- `vif: VIFReport`
- `ridge: RidgeSanityReport`
- `target_col: str`

---

## Minimal end-to-end example

```python
import pandas as pd

from mmm.data.dataset import MMMDataSet
from mmm.eda.runner import EDARunner

df = pd.DataFrame(
    {
        "date": pd.date_range("2023-01-01", periods=52, freq="W"),
        "target__sales": range(52),
        "media__tv__spend": range(52),
        "baseline__trend": [i / 51 for i in range(52)],
        "control__price_index": [1.0] * 52,
    }
)

dataset = MMMDataSet.from_dataframe(df)

runner = EDARunner(target_col="target__sales")
report = runner.run(dataset)

# Correlation matrix
print(report.correlation.matrix)

# VIF scores
print(report.vif.scores)

# Ridge sanity check
print(report.ridge.coefficients)
print(report.ridge.r2_score)
```

---

## Individual usage

Each analysis can also be used independently:

```python
from mmm.eda.correlation import compute_correlation
from mmm.eda.vif import compute_vif
from mmm.eda.ridge import compute_ridge_sanity

corr_report = compute_correlation(dataset)

vif_report = compute_vif(dataset, target_col="target__sales")

ridge_report = compute_ridge_sanity(
    dataset,
    target_col="target__sales",
    alpha=1.0,
)
```

---

## Package structure

```
src/mmm/eda/
├── __init__.py
├── correlation.py      # compute_correlation → CorrelationReport
├── vif.py              # compute_vif → VIFReport
├── ridge.py            # compute_ridge_sanity → RidgeSanityReport
└── runner.py           # EDARunner → EDAReport
```

---

## Common error cases

#### Target column not found
```python
ValueError: target_col 'target__sales' not found in dataset columns.
```

#### Target column name does not follow naming convention v1
```python
ValueError: target_col 'Sales' does not follow naming convention v1.
```

#### Explicit column not found in dataset
```python
ValueError: Column 'media__tv__spend' listed in `columns` not found in dataset.
```

#### Invalid role provided
```python
ValueError: Role 'unknown' is not an allowed role (see naming convention v1).
```

#### Empty predictor set after resolution
```python
ValueError: No predictor columns resolved. Check `columns`, `roles`, or dataset content.
```

#### Fewer than 2 predictors for VIF
```python
ValueError: VIF requires at least 2 predictor columns.
```

#### Constant column (zero variance)
```python
ValueError: Column 'control__price_index' has zero variance; VIF cannot be computed.
```

#### Singular predictor matrix (VIF)
```python
ValueError: Predictor matrix is singular for column 'baseline__trend'; VIF cannot be computed.
```

#### Perfect collinearity detected (VIF — not an error)
Column is assigned `VIF = np.inf`; a warning string is added to `VIFReport.warnings`. Execution continues.
```python
# Example warning entry:
"Column 'media__tv__spend' is perfectly collinear with other predictors (R² >= 1 - 1e-12); VIF set to inf."
```

---

## Dependencies

| Dependency | Usage | Already present |
|------------|-------|-----------------|
| `pandas>=2.0` | DataFrame manipulation | Yes |
| `numpy` | VIF computation (matrix operations) | Implied |
| `scikit-learn` | Ridge regression | **New — must be added to `pyproject.toml`** |

---

## Design constraints

- Input is always `MMMDataSet`; raw DataFrames are not accepted
- Column roles are inferred from names using naming convention v1 (see `02_NAMING_CONVENTION.md`)
- `target_col` is excluded from the predictor matrix in VIF and Ridge
- `date` is always excluded from all analyses
- All output objects are frozen dataclasses (immutable)
- No module in `eda/` may import from `modeling/`, `diagnostics/`, `results/`, or `optimization/`

---

## V1 — Explicit exclusions

The following are **out of scope** for EDA v1:

- Plotting or visualization of any kind
- Bayesian inference or probabilistic analysis
- Statistical significance tests (p-values, t-tests, F-tests)
- Time-series specific diagnostics (ACF, PACF, stationarity tests)
- Automated feature selection or ranking
- Outlier detection
- Cross-validation or train/test split
- Multi-target analysis (only one `target_col` at a time)
- LASSO or other regularization variants (Ridge only)
- Interaction or cross-feature analysis

These may be introduced in future versions.

---

## Implementation Edge Case Handling (V1)

This section summarizes the decisions made for cases that require explicit handling during implementation.

### Read-only enforcement
The `MMMDataSet` passed to any EDA function must not be modified in any way. Any intermediate computation (column centering, matrix construction) must use transient in-memory structures (numpy arrays, local DataFrames) that are discarded after the function returns. Implementations must not call any mutating method on `dataset.df`.

### Variable selection
Predictor resolution follows a strict priority order: explicit `columns` > `roles` filter > default roles (`baseline`, `control`, `event`, `media`). The resolved set always excludes `date` and `target_col`. An empty resolved set is always a hard error.

### VIF: perfect collinearity (R² ≥ 1 − 1e-12)
When regressing predictor `j` on all other predictors yields R² ≥ 1 − 1e-12, the computation does not raise an exception. Instead:
- `scores[column] = np.inf`
- A descriptive warning string is appended to `VIFReport.warnings`

This allows the caller to inspect all columns in a single pass rather than stopping at the first collinear column.

### VIF: singular matrix or regression failure
If the predictor matrix (excluding column `j`) is singular and the regression cannot be completed for a reason other than perfect collinearity, a `ValueError` is raised immediately with an explicit message identifying the problematic column. There is no silent failure or fallback value.

### No silent failures
All unexpected conditions (missing columns, invalid roles, empty predictor sets, computation failures) raise a `ValueError` with a message that identifies the exact cause. Warnings are only used for the `VIF = np.inf` case, which is a well-defined non-exceptional outcome.

---

## Versioning

- This document describes **EDA v1**
- Any breaking change must be introduced in **v2**
