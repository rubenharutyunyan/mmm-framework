# Architecture overview (v1)

This document defines the global architecture of the `mmm-framework` package.
It is the **source of truth** for module responsibilities, boundaries, and data flow.

---

## Design principles

1. **Modular architecture**
   Each module has a single responsibility and a clearly defined scope.

2. **Object-oriented design**
   Core concepts (dataset, features, models, diagnostics, optimization) are represented as objects.

3. **Explicit contracts**
   - Naming conventions
   - Dataset invariants
   - Clear input/output for each module

4. **Bayesian-first**
   The modeling layer is designed primarily for Bayesian MMM (PyMC / PyMC-Marketing).

5. **Business-oriented outputs**
   Modeling outputs must be interpretable and usable for business decisions
   (ROI, response curves, budget optimization).

---

## High-level data flow

```md
Raw data
↓
MMMDataSet (`data/`)
↓
ColumnMapper (mapping + optional normalization)
↓
Feature engineering (`features/`)
↓
EDA & sanity checks (`eda/`)
↓
Bayesian modeling (`modeling/`)
↓
Diagnostics (`diagnostics/`)
↓
Business results & decomposition (`results/`)
↓
Response curves & optimization (`optimization/`)
```

Each step produces validated, structured outputs that can be reused downstream.

---

## Package structure

```md
src/mmm/
├── config/
├── data/
├── features/
├── eda/
├── modeling/
├── diagnostics/
├── results/
├── optimization/
```

---

## Module responsibilities

### `config/`
**Purpose:** global configuration and conventions.

Responsibilities:
- Naming conventions
- Schema definitions
- Global constants and enums

Non-responsibilities:
- No data manipulation
- No modeling logic

---

### `data/`
**Purpose:** dataset representation and validation.

Responsibilities:
- `MMMDataSet` class
- Dataset validation (dates, types, invariants)
- Dataset slicing (time windows, subsets)
- Column mapping layer (client -> naming v1) via `ColumnMapper`+ `MappingReport`

Non-responsibilities:
- No feature engineering
- No modeling logic

---

### `features/`
**Purpose:** feature engineering and transformations understand­able by the model.

Responsibilities:
- Seasonality features
- Baseline / trend features
- Event features
- Media transformations (adstock, saturation, carryover)
- Feature pipelines

Non-responsibilities:
- No model fitting
- No diagnostics

---

### `eda/`
**Purpose:** exploratory analysis and pre-model sanity checks.

Responsibilities:
- Correlation analysis
- Collinearity checks
- Simple frequentist / regularized models (OLS, Ridge)
- Visual diagnostics

Non-responsibilities:
- No Bayesian inference
- No business optimization

---

### `modeling/`
**Purpose:** Bayesian MMM modeling.

Responsibilities:
- Definition of Bayesian models
- Feature grouping
- Prior rules and constraints
- Model fitting and prediction
- Support for PyMC and PyMC-Marketing backends

Non-responsibilities:
- No data cleaning
- No budget optimization

---

### `diagnostics/`
**Purpose:** model quality and statistical validity.

Responsibilities:
- Convergence diagnostics (R-hat, ESS, divergences)
- Posterior predictive checks
- Model comparison (LOO, WAIC where applicable)
- Sanity checks on coefficients and contributions

Non-responsibilities:
- No feature creation
- No business reporting

---

### `results/`
**Purpose:** business-oriented outputs.

Responsibilities:
- Contribution decomposition
- ROI / mROAS computation
- Aggregations by channel / period
- Tables and plots for reporting

Non-responsibilities:
- No model fitting
- No optimization logic

---

### `optimization/`
**Purpose:** budget and scenario optimization.

Responsibilities:
- Response curve usage
- Budget allocation optimization
- Constraints handling (min/max spend, channel locks)
- Scenario simulation

Non-responsibilities:
- No model estimation
- No diagnostics

---

## Cross-cutting rules

- No module may bypass the dataset contract explained in `03_DATASET_CONTRACT.md`
- Feature names must follow `02_NAMING_CONVENTION.md`
- Each module must be testable independently
- CI must remain green at all times

---

## Versioning philosophy

- **v1**: complete end-to-end MMM pipeline (single geo / product)
- Improvements and extensions are added **after** a stable v1
- Breaking architectural changes require an explicit update of this document

---

## Out of scope (v1)

- Multi-geo / hierarchical MMM
- Automated hyperparameter tuning
- Real-time inference
- UI or dashboarding

These may be considered in future versions.
