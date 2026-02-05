# Dataset contract (v1)

## Required columns
- `date`: unique datetime column
- At least one `target__...` column

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
