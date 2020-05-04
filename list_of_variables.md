# Table of contents

- [`all_data`](#all-data)
- [`Key.MAIN`](#keymain)
- [`dataset`](#dataset)
- [`cyc_grp_dict`](#cyc-grp)
- [`fit_args`](#fit-args)

<a name="all-data"/>

## `all_data`
**Dictionary key**: `Key.ALL_DATA`

Dictionary indexed by barcode. Each barcode yields a dictionary with the following keys:
```python
Key.ALL_REF_MATS, Key.CYC_GRP_DICT
```


<a name="keymain"/>

## `Key.MAIN`
Structured array with `dtype`:
```python
[
    (N, "f4"),
    (V_CC, "f4", len(voltage_grid)),
    (Q_CC, "f4", len(voltage_grid)),
    (MASK_CC, "f4", len(voltage_grid)),
    (I_CV, "f4", fit_args[Key.I_MAX]),
    (Q_CV, "f4", fit_args[Key.I_MAX]),
    (MASK_CV, "f4", fit_args[Key.I_MAX]),
    (I_CC, "f4"), (I_PREV, "f4"),
    (V_PREV_END, "f4"), (V_END, "f4"),
    (Q_CC_LAST, "f4"), (Q_CV_LAST, "f4"),
    (Key.TEMP, "f4"),
]
```


<a name="dataset"/>

## `dataset`
**Dictionary key**: `Key.DATASET`

Dictionary with the following keys:
```python
Key.MAX_CAP, Key.GRID_V, Key.Q_GRID, Key.GRID_T, Key.GRID_SIGN,
Key.CELL_TO_POS, Key.CELL_TO_NEG, Key.CELL_TO_LYTE, Key.CELL_TO_LAT, Key.ALL_DATA
```


<a name="cyc-grp"/>

## `cyc_grp_dict`
**Dictionary key**: `Key.CYC_GRP_DICT`

Groups of charge/discharge steps indexed by group averages of
```python
( end_current_prev, constant_current, end_current,
  end_voltage_prev, end_voltage, sign)
```

Each group is a dictionary with the following keys:
```python
Key.MAIN, Key.I_CC_AVG, Key.I_PREV_END_AVG, Key.Q_END_AVG,
Key.V_PREV_END_AVG, Key.V_END_AVG, Key.Key.V_CC_LAST_AVG
```


<a name="fit-args"/>

## `fit_args`
**Dictionary key**: `Key.FIT_ARGS`
