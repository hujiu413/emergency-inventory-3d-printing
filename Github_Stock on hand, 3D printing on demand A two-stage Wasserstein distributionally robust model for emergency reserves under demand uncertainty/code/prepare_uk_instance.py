"""Build local UK case inputs from an EM-DAT CSV obtained by the user."""

import argparse
import csv
import json
import math
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = ROOT / "data" / "parameters"
PROVENANCE = ROOT / "data" / "provenance"
LOCAL_OUTPUT = ROOT / "reproduced"
EVENT_COLUMNS = {
    "dis_no": ("dis_no", "disno."),
    "total_affected": ("total_affected", "total affected"),
    "no_injured": ("no_injured", "no. injured"),
    "no_homeless": ("no_homeless", "no. homeless"),
}
DEMAND_ITEMS = (
    "Drinking_Water_3Day_Kit",
    "Family_Tent",
    "Respiratory_Circuit_Connector",
)
DEMAND_FIELDS = ("sample", "item", "node", "demand")


def read_table(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"{path.name} has no CSV header.")
        columns = {field.strip().casefold(): field for field in reader.fieldnames}
        return columns, list(reader)


def event_counts(path, required_ids):
    columns, rows = read_table(path)
    selected = {}
    for name, aliases in EVENT_COLUMNS.items():
        column = next((columns[alias] for alias in aliases if alias in columns), None)
        if column is None:
            raise ValueError(f"{path.name} needs a column named {aliases[0]} or {aliases[1]}.")
        selected[name] = column

    events = {}
    for row in rows:
        identifier = (row[selected["dis_no"]] or "").strip()
        if identifier not in required_ids:
            continue
        if identifier in events:
            raise ValueError(f"The supplied CSV has duplicate records for {identifier}.")
        events[identifier] = {
            name: optional_count(row[column], identifier, column)
            for name, column in selected.items()
            if name != "dis_no"
        }

    missing = sorted(required_ids.difference(events))
    if missing:
        raise ValueError(f"The supplied CSV is missing the following DisNo. values: {', '.join(missing)}")
    return events


def optional_count(value, identifier, column):
    value = (value or "").strip()
    if value.casefold() in {"", "na", "n/a", "nan", "null"}:
        return None
    try:
        count = Decimal(value)
    except InvalidOperation:
        raise ValueError(f"{identifier} has an invalid count in {column}.") from None
    if not count.is_finite() or count < 0:
        raise ValueError(f"{identifier} needs a finite, nonnegative count in {column}.")
    return count


def demand_totals(event, identifier):
    affected = event["total_affected"]
    injured = event["no_injured"]
    homeless = event["no_homeless"]
    if affected is None:
        if injured is None and homeless is None:
            raise ValueError(f"{identifier} has no affected, injured, or homeless count.")
        affected = max(10 * (injured or 0), 5 * (homeless or 0))
    tent_population = homeless if homeless is not None else Decimal("0.08") * affected
    connector_demand = max(Decimal("0.10") * (injured or 0), Decimal("0.0005") * affected)
    return math.ceil(affected), math.ceil(tent_population / 5), math.ceil(connector_demand)


def allocate(total, node_weights):
    # The source weights are rounded decimals. Normalize them before allocation.
    weight_sum = sum(node_weights.values())
    quotas = {node: total * weight / weight_sum for node, weight in node_weights.items()}
    units = {node: math.floor(quota) for node, quota in quotas.items()}
    remaining = total - sum(units.values())
    # Stable sorting keeps the node-allocation CSV order when remainders tie.
    order = sorted(quotas, key=lambda node: quotas[node] - units[node], reverse=True)
    for node in order[:remaining]:
        units[node] += 1
    return units


def parameter_rows(filename, text_fields=()):
    _, rows = read_table(PARAMETERS / filename)
    return [
        {field: value if field in text_fields else float(value) for field, value in row.items()}
        for row in rows
    ]


def build_instance(events_path):
    _, identifiers = read_table(PROVENANCE / "event_identifiers.csv")
    event_ids = {int(row["sample"]): row["dis_no"] for row in identifiers}
    if len(event_ids) != len(identifiers) or len(set(event_ids.values())) != len(identifiers):
        raise ValueError("The event-identifier table must contain unique samples and DisNo. values.")
    events = event_counts(events_path, set(event_ids.values()))

    _, meta = read_table(PARAMETERS / "meta.csv")
    scalars = parameter_rows("scalar_params.csv")
    node_params = parameter_rows("nodes.csv", ("node", "node_type", "notes"))
    item_params = parameter_rows("items.csv", ("item",))
    printer_params = parameter_rows("printers.csv", ("printer", "item"))
    nodes = [row["node"] for row in node_params]
    items = [row["item"] for row in item_params]
    if set(items) != set(DEMAND_ITEMS):
        raise ValueError("The item parameters must contain the three UK case items.")

    _, assignments = read_table(PROVENANCE / "node_allocation.csv")
    allocations = {sample: {} for sample in event_ids}
    for row in assignments:
        sample = int(row["sample"])
        node = row["node"]
        weight = Fraction(row["weight"])
        if event_ids.get(sample) != row["dis_no"] or node not in nodes:
            raise ValueError("The node allocation table contains an unknown sample, event, or node.")
        if weight <= 0 or node in allocations[sample]:
            raise ValueError(f"Sample {sample} has a nonpositive or duplicate allocation for {node}.")
        allocations[sample][node] = weight

    demand = []
    for sample, identifier in event_ids.items():
        weights = allocations[sample]
        if abs(sum(weights.values()) - 1) > Fraction("0.00001"):
            raise ValueError(f"The allocation weights for sample {sample} do not sum to approximately one.")
        totals = dict(zip(DEMAND_ITEMS, demand_totals(events[identifier], identifier)))
        for item in items:
            units = allocate(totals[item], weights)
            for node in nodes:
                demand.append({"sample": sample, "item": item, "node": node, "demand": units.get(node, 0)})

    return {
        "meta": meta[0],
        "scalar_params": scalars[0],
        "sets": {
            "items": items,
            "nodes": nodes,
            "printers": [row["printer"] for row in printer_params],
            "samples": list(event_ids),
        },
        "node_params": node_params,
        "item_params": item_params,
        "printer_params": printer_params,
        "transport_item_to_node": parameter_rows("transport_item_to_node.csv", ("item", "node")),
        "transport_printer_to_node": parameter_rows("transport_printer_to_node.csv", ("printer", "node")),
        "demand_samples": demand,
    }


def write_instance(instance, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output_dir) as staging_dir:
        staging = Path(staging_dir)
        csv_path = staging / "demand_samples.csv"
        json_path = staging / "instance_real.json"
        with csv_path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=DEMAND_FIELDS)
            writer.writeheader()
            writer.writerows(instance["demand_samples"])
        with json_path.open("w", encoding="utf-8") as stream:
            json.dump(instance, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
        csv_path.replace(output_dir / csv_path.name)
        json_path.replace(output_dir / json_path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", required=True, type=Path, help="Local EM-DAT CSV containing the case events.")
    parser.add_argument("--output-dir", type=Path, default=LOCAL_OUTPUT / "local_input",
                        help="Private directory for the generated CSV and JSON. Default: reproduced/local_input.")
    args = parser.parse_args()
    source = args.events.resolve()
    output_dir = args.output_dir.resolve()
    if (output_dir == ROOT or ROOT in output_dir.parents) and (
        output_dir != LOCAL_OUTPUT and LOCAL_OUTPUT not in output_dir.parents
    ):
        parser.error("Choose a private output directory outside the repository or under reproduced/.")
    if source in {output_dir / "demand_samples.csv", output_dir / "instance_real.json"}:
        parser.error("The output files would overwrite the source CSV. Choose a different output directory.")
    if not source.is_file():
        parser.error(f"The EM-DAT CSV does not exist: {source}")
    try:
        instance = build_instance(source)
        write_instance(instance, output_dir)
    except (OSError, ValueError, KeyError, IndexError, csv.Error) as error:
        parser.error(str(error))
    print(f"Wrote {len(instance['demand_samples'])} demand entries and instance_real.json to {output_dir}.")


if __name__ == "__main__":
    main()
