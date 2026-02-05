# Dataset Mapping (v1)

Ce document décrit comment ingérer un dataset “client” (colonnes non standard), mapper explicitement les colonnes vers la nomenclature interne MMM (Naming convention v1), puis produire un `MMMDataSet` validé.

## Rappels (V1)

### Naming convention (v1)
- Format général : `<role>__<entity>__<metric>__<qualifiers...>`
- Séparateur : `__`
- Case : `snake_case`
- Caractères autorisés : `a-z`, `0-9`, `_`
- Nom réservé : `date` (obligatoire, unique)
- Rôles autorisés :
  - `target__...`
  - `media__...`
  - `control__...`
  - `event__...`
  - `baseline__...` (optionnel)
  - `id__...` (optionnel)

### Dataset contract (v1) — validé par `MMMDataSet.from_dataframe`
- Colonnes requises :
  - `date` (datetime, unique, strictement croissante, sans doublons)
  - au moins une colonne `target__...`
- Types :
  - toutes les colonnes hors `date` doivent être numériques
- Valeurs :
  - `target__*` : pas de valeurs manquantes
  - `media__*` : `>= 0`
  - `event__*` : dans `[0, 1]`

> ⚠️ Le mapping est une étape *avant* la construction du dataset MMM.  
> La validation du contract se fait ensuite via `MMMDataSet.from_dataframe(...)`.

---

## API (V1)

- `ColumnMapper(mapping=..., normalize_source_columns=..., keep_unmapped=...).apply(df)`
  - retourne `(df_mapped, mapping_report)`

### Paramètres importants
- `mapping` (dict) : `{source_col: target_col}`
- `normalize_source_columns` (bool) : normalise les colonnes sources (snake_case, accents, espaces, etc.)
- `keep_unmapped` (bool) :
  - `True` : conserve les colonnes non mappées (par défaut)
  - `False` : drop les colonnes non mappées

### Erreurs explicites
- colonne source absente
- collisions (plusieurs sources vers une même cible, ou cible qui collisionne avec une colonne existante non mappée)
- cible non conforme au naming v1
- collisions créées par la normalisation (deux colonnes différentes qui normalisent vers le même nom)

---

## Exemple 1 — Cas nominal (mapping explicite + validation MMMDataSet)

```python
import pandas as pd

from mmm.data.column_mapper import ColumnMapper
from mmm.data.dataset import MMMDataSet

# Dataset client (raw)
df_client = pd.DataFrame({
    "Date": ["2024-01-01", "2024-01-02"],
    "Sales": [100, 120],
    "TV Spend": [50, 60],
    "Price Index": [1.02, 1.03],
})

# Mapping explicite vers la nomenclature MMM
mapping = {
    "Date": "date",
    "Sales": "target__sales",
    "TV Spend": "media__tv__spend",
    "Price Index": "control__price_index",
}

# 1) Mapping
df_mapped, mapping_report = ColumnMapper(
    mapping=mapping,
    normalize_source_columns=False,
    keep_unmapped=True,
).apply(df_client)

# 2) Validation dataset contract + construction du MMMDataSet
dataset = MMMDataSet.from_dataframe(df_mapped)

print(df_mapped.columns.tolist())
print(mapping_report)
```

---

## Exemple 2 — Normalisation des colonnes sources (optionnelle)

Utile si les colonnes client contiennent :
* accents (`Dépenses TV`)
* espaces (`TV Spend`)
* variations de casse (`Sales`, `sales`, `SALES`)

```python
import pandas as pd

from mmm.data.column_mapper import ColumnMapper
from mmm.data.dataset import MMMDataSet

df_client = pd.DataFrame({
    "Date ": ["2024-01-01", "2024-01-02"],      # espace
    "Ventes (€)": [100, 120],                  # accents / caractères spéciaux
    "TV Spend": [50, 60],
})

mapping = {
    # ⚠️ Les clés sont aussi normalisées si normalize_source_columns=True
    "Date ": "date",
    "Ventes (€)": "target__sales",
    "TV Spend": "media__tv__spend",
}

df_mapped, report = ColumnMapper(
    mapping=mapping,
    normalize_source_columns=True,   # <-- normalisation activée
).apply(df_client)

dataset = MMMDataSet.from_dataframe(df_mapped)
```

> ⚠️ Si la normalisation transforme deux colonnes différentes vers le même nom (ex: "Sales " et "Sales" → "sales"), une exception est levée.

---

## Exemple 3 — Conserver ou supprimer les colonnes non mappées

### A) Conserver les colonnes non mappées (par défaut)

```python
df_mapped, report = ColumnMapper(
    mapping={
        "Date": "date",
        "Sales": "target__sales",
    },
    keep_unmapped=True,   # default
).apply(df_client)

# df_mapped conserve les colonnes non mappées
```

### B) Supprimer les colonnes non mappées

```python
df_mapped, report = ColumnMapper(
    mapping={
        "Date": "date",
        "Sales": "target__sales",
    },
    keep_unmapped=False,
).apply(df_client)

# df_mapped ne contient plus que les colonnes mappées
```

## Exemple 4 — Exploiter le mapping_report

`mapping_report` permet la traçabilité :
* colonnes originales
* colonnes normalisées (si option activée)
* mapping appliqué
* colonnes renommées
* colonnes ignorées / droppées

```python
df_mapped, report = ColumnMapper(
    mapping={
        "Date": "date",
        "Sales": "target__sales",
    },
    normalize_source_columns=False,
    keep_unmapped=True,
).apply(df_client)

print("Original columns:", report.original_columns)
print("Renamed columns:", report.renamed_columns)
print("Unmapped columns:", report.unmapped_columns)
```

---

## Erreurs fréquentes (et comment les résoudre)

### 1) Colonne source absente

__Cause__ : le dataset client ne contient pas une clé du mapping.  
__Solution__ : corriger le mapping ou le dataset client.

### 2) Collision de cibles (2 sources → 1 cible)

__Cause__ : deux colonnes sources mappées vers le même nom MMM.  
__Solution__ : corriger le mapping (une cible doit être unique).

### 3) Cible non conforme naming v1

__Cause__ : cible ne respecte pas :
* rôle autorisé
* séparateur `__`
* snake_case / caractères autorisés

__Solution__ : renommer la cible pour respecter la convention.

### 4) Collision entre une cible et une colonne non mappée (keep_unmapped=True)

__Cause__ : tu mappes une colonne vers un nom qui existe déjà dans le dataset (et qui n’est pas renommé).  
__Solution__ :
* mapper aussi cette colonne
* ou `keep_unmapped=False`

---

## Position dans le data flow

1. Ingestion dataset client (raw)  
2. __Mapping explicite__ (ce module)  
3. `MMMDataSet.from_dataframe(df_mapped)` :  
  * validation dataset contract (v1)  
  * construction de l’objet dataset conforme
