# Naming convention (v1)

## General format
`<role>__<entity>__<metric>__<qualifiers...>`

- Separator: `__` (double underscore)
- Case: `snake_case`
- Allowed characters: `a-z`, `0-9`, `_`
- No spaces, no accents, no special characters

## Reserved column names
- `date` (mandatory, unique)

## Allowed roles (prefixes)
- `target__...`
- `media__...`
- `control__...`
- `event__...`
- `baseline__...` (optional)
- `id__...` (optional, future multi-geo/product)

## Examples
- `target__sales`
- `media__tv__spend`
- `media__meta__impressions`
- `control__price_index`
- `event__black_friday`
- `baseline__trend`
