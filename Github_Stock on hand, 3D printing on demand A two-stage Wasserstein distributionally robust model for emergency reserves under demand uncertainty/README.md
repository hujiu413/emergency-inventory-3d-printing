# Emergency inventory and 3D printing: parameters, code and numerical results

Computational materials for joint emergency inventory and 3D printing capacity
planning under demand uncertainty, using a two-stage Wasserstein distributionally
robust optimization model.

This repository provides model parameters, code, event identifiers, author-defined
allocation weights and archived results for a United Kingdom flood preparedness
case. The code also contains synthetic experiments. It does not contain every
experiment output from the manuscript.

## Case scope

- 27 historical-event scenarios, 11 demand nodes, 3 item types and 4 printing centers.
- Drinking-water kits, family tents and respiratory circuit connectors.
- Baseline capacity: 1,800 cubic metres; budget: 8,250,000 yuan; response window: 72 hours.
- 10 archived numerical-result CSV files and 12 figure files (PNG/PDF).

Demand is constructed from historical event information using modeling rules,
rather than directly observed item requests. Coordinates are city proxies and
transport costs are model estimates. Budget and capacity describe the modeled
allocation, not the complete UK reserve system. Monetary inputs retain yuan units.

The original metadata describes a 2000–2025 screening window. The 27 retained
events occurred in 2000–2020; both facts are distinguished here.

## What is public and what must be obtained separately

The raw EM-DAT event export is excluded. The complete event-level demand table and
optimization JSON are also excluded: for each event, the water-kit demand total
equals the original EM-DAT Total Affected value, so distributing them would expose
those original values. The public repository therefore cannot run the UK case
until the relevant source records are obtained under EM-DAT's access terms.

Register/access the source at <https://public.emdat.be/>. Use the 27 identifiers in
`data/provenance/event_identifiers.csv` to retrieve the relevant records. Source
updates can differ from the original study snapshot. The original export date was
not supplied and is not inferred from this package's creation date.

## Files

| Path | Contents |
| --- | --- |
| `code/experiments_v3.py` | Original optimization and synthetic-experiment code |
| `code/run_uk_real_case_experiments.py` | Original UK case runner |
| `code/prepare_uk_instance.py` | Added standard-library helper for preparing local inputs from an authorized source CSV |
| `data/parameters/` | Seven CSV files: model parameters and original case metadata |
| `data/provenance/` | Source event identifiers and author-defined allocation weights |
| `results/uk_case/` | 10 supplied numerical-result CSV files |
| `figures/uk_case/` | Supplied figures, retaining original filenames |
| `docs/` | Data dictionary, source access and reproduction notes |
| `checksums.sha256` | Checksums of the other repository files |

## Prepare local inputs and run the UK case

Keep your authorized source CSV outside the public repository. The preparation
helper accepts original EM-DAT column names (`DisNo.`, `Total Affected`,
`No. Injured`, `No. Homeless`) or the study-export aliases (`dis_no`,
`total_affected`, `no_injured`, `no_homeless`).

From the repository root, substitute the real path to that local CSV:

```sh
python code/prepare_uk_instance.py --events /path/to/authorized_emdat_events.csv --output-dir reproduced/local_input
python -m pip install -r requirements.txt
python code/run_uk_real_case_experiments.py --data-path reproduced/local_input/instance_real.json --output-dir reproduced/uk_case --zeta 10 --max-iters 120 --tl-master 300 --tl-sp 30
```

Use Python 3 and a working Gurobi installation/license. The helper itself uses
only the standard library. The solver scripts require `numpy`, `matplotlib` and
`gurobipy`. Their exact original environment versions were not fully recorded.

The helper reconstructs demand and a local JSON without embedding the raw counts
in the public code. `reproduced/` is ignored by Git. Keep these locally generated
source-derived inputs out of public uploads. Additional diagnostic plots can be
requested with `--save-diagnostic-plots`.

The source-derived demand reconstruction was checked against the study's supplied
891 entries. Archived results are copied, not recalculated; optimization was not
rerun during packaging. See [reproduction notes](docs/REPRODUCING.md).

## Sources and rights

The historical source is **EM-DAT Public Data, Emergency Events Database,
UCLouvain/CRED**, available at <https://public.emdat.be/>. Source access and use
remain subject to its conditions. See [sources and access](docs/SOURCES_AND_ACCESS.md).

No blanket MIT, CC0 or other reuse license has been assigned during packaging.
Public availability does not change third-party source rights. Cite EM-DAT when
using the case and cite the study when publication details are available.
