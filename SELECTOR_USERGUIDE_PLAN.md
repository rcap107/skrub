# User Guide Improvement Plan: Selector Documentation

**Status:** Active
**Scope:** Documentation improvements only — user guide RST files
**Files:** `doc/modules/multi_column_operations/selectors.rst`, `type_of_selectors.rst`, `advanced_selectors.rst`

---

## Current State Summary

| File | Lines | Status |
|------|-------|--------|
| `selectors.rst` | ~235 | Good foundation, missing depth |
| `type_of_selectors.rst` | ~99 | Category list only, no guidance |
| `advanced_selectors.rst` | ~126 | Good content, poorly positioned |

---

## What to Keep As-Is

**`selectors.rst`:**
- Intro paragraph explaining selectors + delayed selection concept
- Three ways to use selectors (select function, transformers, DataOps) with examples
- "Combining selectors" with all 5 operators and short-circuit evaluation note
- `expand()` and `expand_index()` section ("Visualizing a selector")
- `ApplyToCols` + `DropCols` transformer examples

**`type_of_selectors.rst`:**
- Grouped category list (dtype, content/property, name-based)

**`advanced_selectors.rst`:**
- `filter()` and `filter_names()` with pickling note
- Real-world outlier detection example
- `has_nulls` with realistic medical dataset example

---

## Gaps & Proposed Changes

### Gap 1 — No "How Selectors Work" explanation (`selectors.rst`)

**Problem:** The intro mentions *what* selectors are and *how to use* them, but skips
*how they work internally*. The per-column `_matches()` evaluation loop — and why it
enables delayed selection — is never explained. Users jumping straight to transformers
won't understand why `selector.expand(df)` sometimes behaves differently than expected.

**Change:** Add a short "How selectors work" subsection between the intro and "Type
of selectors", covering:
- The per-column `_matches()` evaluation model (text prose, no code needed)
- Why this enables delayed selection: you can pass a selector to `ApplyToCols` before
  you have data; hardcoded column names would break at train/test split time.

**Estimated size:** ~20–30 lines.

---

### Gap 2 — No "Choosing a Selector" guidance (`type_of_selectors.rst`)

**Problem:** The file lists selectors by category but gives no guidance on *which* to
pick. Users scanning this page cannot distinguish `string()` vs `object()`, or
`glob()` vs `regex()`, from the list alone.

**Change:** Add a "Choosing a selector" subsection structured as a decision tree:

```
What do you want to select by?

Column name
  ├─ Fixed list of names          → cols()
  ├─ Simple wildcard (*, ?)       → glob()
  ├─ Complex regex pattern        → regex()
  └─ Custom rule on the name      → filter_names()

Data type
  ├─ Numbers (int or float)       → numeric()   [= integer() | float()]
  ├─ Integers only                → integer()
  ├─ Floats only                  → float()
  ├─ Text / strings               → string()
  ├─ Categorical (fixed values)   → categorical()
  ├─ Dates / datetimes            → any_date()
  ├─ Booleans                     → boolean()
  ├─ Object dtype (pandas legacy) → object()
  └─ Custom / exotic dtype        → has_dtype()

Column content / statistics
  ├─ Few unique values            → cardinality_below()
  ├─ Contains missing values      → has_nulls()
  └─ Custom rule on data          → filter()

Multiple criteria
  ├─ Either condition             → selector1 | selector2
  ├─ Both conditions              → selector1 & selector2
  ├─ All except                   → selector1 - selector2
  └─ NOT                          → ~selector
```

Also add a "Selector relationships" note explaining:
- `numeric()` = `integer() | float()`
- `boolean()` is **not** included in `numeric()`
- `string()` ⊂ `object()` in pandas pre-3.0; prefer `string()` for text data
- `categorical()` ≠ `string()` — different dtypes, both text-like

**Estimated size:** ~50–70 lines.

---

### Gap 3 — No operator summary table (`selectors.rst`)

**Problem:** The "Combining selectors" section shows one code example per operator
but has no summary table. Users scanning quickly cannot see all operators at a glance.

**Change:** Add a table before the existing operator examples:

| Operator | Meaning | Example |
|----------|---------|---------|
| `\|` | Union — either matches | `s.numeric() \| s.boolean()` |
| `&` | Intersection — both match | `s.numeric() & s.cardinality_below(10)` |
| `-` | Difference — left minus right | `s.all() - s.glob('*_id')` |
| `^` | XOR — exactly one matches | `s.numeric() ^ s.integer()` |
| `~` | Inversion — NOT | `~s.string()` |

**Estimated size:** ~15 lines.

---

### Gap 4 — No "Common Patterns" section (`selectors.rst`)

**Problem:** The guide shows how selectors work in isolation but never shows idiomatic
multi-selector patterns that users actually need. The `ApplyToCols` example uses
`s.numeric()` alone but never shows realistic combining.

**Change:** Add a "Common patterns" subsection at the end of `selectors.rst`:

```python
# Apply scaling only to continuous columns (floats, not IDs)
ApplyToCols(StandardScaler(), cols=s.float()).fit_transform(df)

# Encode all text-like columns (strings and categoricals)
ApplyToCols(GapEncoder(), cols=s.string() | s.categorical()).fit_transform(df)

# Drop columns with more than 50% missing data
DropCols(cols=s.has_nulls(proportion=0.5)).fit_transform(df)

# Select discrete numeric features (low cardinality)
s.select(df, s.numeric() & s.cardinality_below(10))

# Remove ID-like columns by name pattern
DropCols(cols=s.glob('*_id') | s.glob('id_*')).fit_transform(df)
```

**Estimated size:** ~50–60 lines with prose.

---

### Gap 5 — `filter` / `filter_names` too buried (`advanced_selectors.rst`)

**Problem:** These selectors live in a separate file labeled "advanced". First-time
users reading `selectors.rst` see only a small `seealso` link and likely miss that
custom criteria are possible at all.

**Change (minimal):** Add a short "Custom criteria" paragraph in `selectors.rst`
under "Type of selectors", e.g.:

> For selection based on custom logic (e.g., column variance, specific value presence),
> use :func:`filter` and :func:`filter_names`. See :ref:`user_guide_advanced_selectors`.

**Estimated size:** ~5 lines.

---

### Gap 6 — One-line descriptions missing from category list (`type_of_selectors.rst`)

**Problem:** The category list links to API docs but gives no inline guidance. Users
cannot distinguish related selectors (e.g. `string()` vs `object()`, `glob()` vs
`regex()`) without clicking through each one.

**Change:** Add one-line annotations to the existing list, mirroring the `See Also`
relationships already in the docstrings. Examples:

```rst
- :func:`~skrub.selectors.string`: Select string columns. Prefer over
  :func:`~skrub.selectors.object` for text data.
- :func:`~skrub.selectors.categorical`: Select categorical columns
  (fixed set of values, unlike string).
- :func:`~skrub.selectors.glob`: Simple wildcard patterns. Use
  :func:`~skrub.selectors.regex` for complex patterns.
```

**Estimated size:** ~20 lines of additions to the existing list.

---

## Priority Order

| Priority | Gap | File | Effort |
|----------|-----|------|--------|
| 1 | Decision tree ("Choosing a selector") | `type_of_selectors.rst` | Medium |
| 2 | Operator summary table | `selectors.rst` | Small |
| 3 | Common patterns section | `selectors.rst` | Medium |
| 4 | "How it works" explanation | `selectors.rst` | Small |
| 5 | One-line category descriptions | `type_of_selectors.rst` | Small |
| 6 | filter/filter_names visibility link | `selectors.rst` | Trivial |
