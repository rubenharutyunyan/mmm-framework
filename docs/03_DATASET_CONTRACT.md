# Dataset contract (v1)

## Required columns
- `date`: unique datetime column
- At least one `target__...` column

## Before contract validation (mapping step)

Input datasets may initially be non-compliant with the MMM naming convention (v1).
A **mapping step** must be applied first to rename client columns into the internal naming format.

- Recommended entry point: `ColumnMapper.apply(df_client)` → returns `df_mapped` + `mapping_report`
- Then the dataset contract is enforced by: `MMMDataSet.from_dataframe(df_mapped)`

`MMMDataSet.from_dataframe(...)` remains the single gatekeeper for:
- date invariants (parseable, unique, increasing)
- type invariants (numeric columns)
- value invariants (media >= 0, event in [0,1], etc.)

## Date invariants
- Must be parseable as datetime
- Strictly increasing
- No duplicates

## Type invariants
- All non-date columns must be numeric

## Value invariants
- `target__*`: no missing values
- `media__*`: must be >= 0
- `event__*`: must be in [0, 1]
