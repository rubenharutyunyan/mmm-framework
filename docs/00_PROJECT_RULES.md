# Project rules (Source of truth)

## Scope of this repository
This repository contains a Python package to perform Marketing Mix Modeling (MMM):
data preparation, feature engineering, Bayesian modeling, diagnostics, response curves, and budget optimization.

## Working principles
1. **This repo is the source of truth** for code and specifications.
2. We build **step-by-step** with **small increments**.
3. Each increment must be **testable** and validated by CI before moving on.
4. We prioritize a **complete V1** (end-to-end) before adding improvements.
5. Any change impacting global contracts must be decided in the **Architecture & Conventions** track.

## Repo rules
- No raw data committed to the repository.
- Code lives in `src/mmm/`.
- Tests live in `tests/`.
- The CI pipeline must remain green on supported Python versions.

## Discussions workflow
- One discussion = one feature/module spec.
- Architecture & conventions are defined in dedicated specs under `docs/`.
