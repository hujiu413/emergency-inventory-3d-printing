# Reproduction notes

## Dependencies

The original scripts import `numpy`, `matplotlib`, `gurobipy` and the Python
standard library. Solving requires a valid Gurobi license. The synthetic
scalability function mentions Gurobi 13.0, but the supplied archive does not record
a complete software environment. Requirements are therefore not version-pinned.
Plots request Times New Roman; font availability can change their appearance.

## Preparing source-derived inputs locally

The supplied archive lacked its earlier CSV-to-JSON preparation script. The added
`code/prepare_uk_instance.py` implements the documented event-demand rules and
the node allocation reconstructed from the supplied tables. With the study's
source export, its demand values were matched to all 891 original entries and
its numerical model inputs were compared with the supplied instance.

Use an authorized EM-DAT CSV as shown in the top-level README. Do not upload that
CSV or the resulting demand CSV/JSON to the public repository. These outputs are
written to the ignored `reproduced/local_input/` directory. Retrieval from current
EM-DAT data may give different values because the original download date was not
recorded and source records may have been updated.

The UK runner reads only the prepared local JSON. It performs baseline runs,
event holdouts and targeted/activation comparisons, involving many solver calls.
Results go to `reproduced/uk_case/`, separate from the archived result files.
Keep the original two scripts together because the runner imports
`experiments_v3`. Use the explicit `--data-path`; the original default expects
the JSON beside the runner.

The runner produces CSVs, PNGs and Markdown summaries, including an internal
Chinese narrative draft. That draft is not an archived numerical result. The
runner does not directly regenerate the final-layout `Figure8.pdf`–`Figure11.pdf`
files. Use `--save-diagnostic-plots` to request the optional diagnostic figures.

## Synthetic experiments

Synthetic experiments do not need external event data. The default main block in
`experiments_v3.py` runs experiments 1–3 and 4a–4d. Calls to 5–7 exist but are
commented out. The script writes outputs to the current directory; for example,
in PowerShell from the repository root:

```powershell
New-Item -ItemType Directory -Force -Path reproduced/synthetic
Push-Location reproduced/synthetic
python ../../code/experiments_v3.py
Pop-Location
```

## Verification scope

Packaging preserved the original two solver scripts, numerical parameter values,
archived CSV result files and figure bytes. Non-numerical raw-name annotations
were removed from node notes. Event identifiers and allocation columns were
selected from the supplied provenance tables. Documentation and the preparation
helper were added; internal drafts and duplicate inputs were omitted.

Checks covered input dimensions, demand reconstruction, Python syntax, file
integrity and ZIP contents. Optimization experiments were not rerun. Archived
results are existing outputs, not fresh reproduction results. No common-run log
was supplied to certify all archived figures/tables came from one run.
