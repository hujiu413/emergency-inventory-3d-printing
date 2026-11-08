# Data dictionary

## Public input files

| File | Rows | Interpretation |
| --- | ---: | --- |
| `data/parameters/items.csv` | 3 | Item, cost `c_i_yuan`, unit storage volume `h_i`, parameter `R_i` |
| `data/parameters/nodes.csv` | 11 | Model node name/type, city-proxy coordinates and construction notes |
| `data/parameters/printers.csv` | 4 | Printing-center/item mapping, equipment/material costs and production parameters |
| `data/parameters/scalar_params.csv` | 1 | `A_max`, `B_yuan`, `T`, `delta`, `money_scale` |
| `data/parameters/transport_item_to_node.csv` | 33 | Complete item/node matrix; transport `c_ij_yuan` and shortage `M_ij_yuan` parameters |
| `data/parameters/transport_printer_to_node.csv` | 44 | Complete printing-center/node cost matrix `c_kj_yuan` |
| `data/parameters/meta.csv` | 1 | Original case metadata and modeling caveats |
| `data/provenance/event_identifiers.csv` | 27 | Local `sample` mapped to EM-DAT `dis_no` (`DisNo.`) |
| `data/provenance/node_allocation.csv` | varies | Local scenario/node assignment, allocation weight and assignment reason |

Parameter names follow the original code. Detailed mathematical definitions are
in the study and `experiments_v3.py`. Printing-model settings are not independent
validation of a device specification. City proxy coordinates remain as supplied;
copied `raw_name` annotations were omitted from node notes.

## Locally reconstructed demand

The full demand CSV and JSON are excluded from public files. Obtain the source
events and use the preparation helper to construct the 891 scenario/item/node
entries locally. At event level, define `exposed_people` as Total Affected when
available; otherwise use `max(10 * No. Injured, 5 * No. Homeless)`.

- Water kits: `ceil(exposed_people)`.
- Family tents: `ceil((No. Homeless if available else 0.08 * exposed_people) / 5)`.
- Respiratory circuit connectors: `ceil(max(0.10 * No. Injured, 0.0005 * exposed_people))`.

The event totals are distributed using the supplied node weights: take integer
floors, then allocate the remaining units by descending fractional remainder,
retaining source row order for ties. The helper implements these modeling rules.
They are assumptions for numerical analysis, not observed medical-item demand.

The resulting JSON contains `meta`, `scalar_params`, `sets`, `node_params`,
`item_params`, `printer_params`, `transport_item_to_node`,
`transport_printer_to_node`, and `demand_samples`. The solver reads JSON, so CSV
changes alone do not alter a run; first rebuild and check the local JSON.

## Archived result files

| Prefix | Contents |
| --- | --- |
| `uk_baseline_summary` | Baseline DRO, SAA and No3D-DRO results |
| `uk_leave_one_out_benchmark` | Event holdout comparison and its summary |
| `uk_leave_one_out_3d_value` | Event holdout printing comparison and its summary |
| `uk_targeted_scenarios` | Targeted parameter scenarios |
| `uk_activation_scenarios` | Printing activation scenarios |
| `uk_activation_value_comparison` | Printing policy comparison in selected scenarios |
| `uk_activation_value_by_item` | Item-level aggregate demand, shortages and printed quantities |
| `uk_3d_activation_integrated` | Combined scenario summary |

Cost columns ending in `_M` are millions of yuan. Rates/shares are fractions;
multiply by 100 to display percentages. Averaging definitions and scenario changes
are implemented in the runner. DRO and SAA coincide in the supplied baseline.
Differences across scenarios with different demand or response windows do not
isolate the effect of printing.

Original `Figure8`–`Figure11` filenames are retained; no new mapping to revised
manuscript figure numbers is asserted.
