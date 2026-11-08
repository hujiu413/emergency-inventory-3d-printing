#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean

import experiments_v3 as exp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / 'instance_real.json'

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def subset_problem_data(data: exp.ProblemData, sample_ids):
    sample_ids = list(sample_ids)
    d_hat = {(n, i, j): data.d_hat[(n, i, j)] for n in sample_ids for i in data.I for j in data.J}
    d_low = {}
    d_up = {}
    delta = float(getattr(data, 'delta', 0.4))
    for i in data.I:
        for j in data.J:
            vals = [d_hat[(n, i, j)] for n in sample_ids]
            avg = sum(vals) / len(vals)
            d_low[(i, j)] = max(0.0, (1.0 - delta) * avg)
            d_up[(i, j)] = (1.0 + delta) * avg
    return exp._clone_problem_data(data, N=sample_ids, d_hat=d_hat, d_low=d_low, d_up=d_up)


def write_csv(path: Path, rows, fieldnames):
    with path.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, '') for k in fieldnames})
    print(f'Saved: {path}')


def baseline_case(data, zeta, max_iters, tl_master, tl_sp):
    rows = []
    for label, z, disable_3d in [('DRO', zeta, False), ('SAA', 0.0, False), ('No3D-DRO', zeta, True)]:
        res = exp.solve_ccg(data, z, max_iters=max_iters, tol=1e-3, tl_master=tl_master, tl_sp=tl_sp, disable_3d=disable_3d)
        eval_full = exp.evaluate_policy_on_dataset(data, res)
        rows.append({
            'model': label,
            'obj_M': res['obj_M'],
            'fs_M': res['fs_M'],
            'sum_xi': res['sum_xi'],
            'sum_xk': res['sum_xk'],
            'avg_mk': res['avg_mk'],
            'mean_total_M': eval_full['mean_total_M'],
            'aggregate_shortage_rate': eval_full['aggregate_shortage_rate'],
            'mean_wait_share': eval_full['mean_wait_share'],
            'mean_service_level': eval_full['mean_service_level'],
            'iters': res['iters'],
            'wall_time': res['wall_time'],
        })
    return rows


def leave_one_out(data, zeta, max_iters, tl_master, tl_sp):
    benchmark_rows = []
    value3d_rows = []
    sample_ids = list(data.N)

    for holdout in sample_ids:
        train_ids = [n for n in sample_ids if n != holdout]
        train_data = subset_problem_data(data, train_ids)
        test_data = subset_problem_data(data, [holdout])

        dro = exp.solve_ccg(train_data, zeta, max_iters=max_iters, tol=1e-3, tl_master=tl_master, tl_sp=tl_sp)
        saa = exp.solve_ccg(train_data, 0.0, max_iters=max_iters, tol=1e-3, tl_master=tl_master, tl_sp=tl_sp)
        det_data = exp.build_deterministic_instance_from_training(train_data)
        det = exp.solve_ccg(det_data, 0.0, max_iters=max_iters, tol=1e-3, tl_master=tl_master, tl_sp=tl_sp)
        no3d = exp.solve_ccg(train_data, zeta, max_iters=max_iters, tol=1e-3, tl_master=tl_master, tl_sp=tl_sp, disable_3d=True)

        benchmark_models = [('DRO', dro), ('SAA', saa), ('Deterministic', det)]
        for name, res in benchmark_models:
            ref = exp.evaluate_policy_on_dataset(train_data, res)['mean_total_M']
            test_eval = exp.evaluate_policy_on_dataset(test_data, res, ref_total_M=ref)
            benchmark_rows.append({
                'holdout_sample': holdout,
                'model': name,
                'ref_total_M': ref,
                'test_total_M': test_eval['mean_total_M'],
                'dis': test_eval['mean_dis'],
                'shortage_rate': test_eval['aggregate_shortage_rate'],
                'wait_share': test_eval['mean_wait_share'],
                'service_level': test_eval['mean_service_level'],
            })

        for name, res in [('Full 3D-enabled', dro), ('No 3D', no3d)]:
            test_eval = exp.evaluate_policy_on_dataset(test_data, res)
            value3d_rows.append({
                'holdout_sample': holdout,
                'policy': name,
                'test_total_M': test_eval['mean_total_M'],
                'shortage_rate': test_eval['aggregate_shortage_rate'],
                'wait_share': test_eval['mean_wait_share'],
                'service_level': test_eval['mean_service_level'],
                'sum_xk_train': res['sum_xk'],
            })

        print(f'Processed holdout sample {holdout}/{sample_ids[-1]}')

    return benchmark_rows, value3d_rows


def summarize_rows(rows, group_key, value_keys):
    groups = {}
    for row in rows:
        groups.setdefault(row[group_key], []).append(row)
    out = []
    for group, subset in groups.items():
        summary = {group_key: group, 'n': len(subset)}
        for key in value_keys:
            summary[f'mean_{key}'] = mean(float(r[key]) for r in subset)
            summary[f'max_{key}'] = max(float(r[key]) for r in subset)
            summary[f'min_{key}'] = min(float(r[key]) for r in subset)
        out.append(summary)
    return out


def scale_budget(data, factor):
    return exp._clone_problem_data(data, B=float(data.B) * float(factor))


def scale_connector_demand(data, factor, item_name='Respiratory_Circuit_Connector'):
    d_hat = dict(data.d_hat)
    d_low = dict(data.d_low)
    d_up = dict(data.d_up)
    for n in data.N:
        for j in data.J:
            d_hat[(n, item_name, j)] = float(d_hat[(n, item_name, j)]) * float(factor)
    for j in data.J:
        vals = [d_hat[(n, item_name, j)] for n in data.N]
        avg = sum(vals) / len(vals)
        delta = float(getattr(data, 'delta', 0.4))
        d_low[(item_name, j)] = max(0.0, (1.0 - delta) * avg)
        d_up[(item_name, j)] = (1.0 + delta) * avg
    return exp._clone_problem_data(data, d_hat=d_hat, d_low=d_low, d_up=d_up)


def scale_printer_fixed_cost(data, factor):
    f_k = {k: float(v) * float(factor) for k, v in data.f_k.items()}
    return exp._clone_problem_data(data, f_k=f_k)


def scale_connector_shortage_penalty(data, factor, item_name='Respiratory_Circuit_Connector'):
    M_ij = dict(data.M_ij)
    for j in data.J:
        M_ij[(item_name, j)] = float(M_ij[(item_name, j)]) * float(factor)
    return exp._clone_problem_data(data, M_ij=M_ij)


def scale_connector_traditional_supply(data, purchase_factor=1.0, transport_factor=1.0,
                                       item_name='Respiratory_Circuit_Connector'):
    c_i = dict(data.c_i)
    c_ij = dict(data.c_ij)
    c_i[item_name] = float(c_i[item_name]) * float(purchase_factor)
    for j in data.J:
        c_ij[(item_name, j)] = float(c_ij[(item_name, j)]) * float(transport_factor)
    return exp._clone_problem_data(data, c_i=c_i, c_ij=c_ij)


def make_existing_printer_network(data, printer_lb=1, printer_ub=1,
                                  fixed_cost_factor=0.0, space_factor=0.05):
    f_k = {k: float(v) * float(fixed_cost_factor) for k, v in data.f_k.items()}
    H_k = {k: max(1e-4, float(v) * float(space_factor)) for k, v in data.H_k.items()}
    printer_lb_map = {k: int(printer_lb) for k in data.K}
    printer_ub_map = {k: int(printer_ub) for k in data.K}
    return exp._clone_problem_data(data, f_k=f_k, H_k=H_k), printer_lb_map, printer_ub_map


def build_combined_activation_scenarios(data, item_name='Respiratory_Circuit_Connector'):
    scenarios = []

    mild_data = scale_connector_demand(data, 10.0, item_name=item_name)
    mild_data = scale_connector_shortage_penalty(mild_data, 10.0, item_name=item_name)
    mild_data = scale_connector_traditional_supply(
        mild_data, purchase_factor=10.0, transport_factor=10.0, item_name=item_name
    )
    mild_data, mild_lb, mild_ub = make_existing_printer_network(
        mild_data, printer_lb=1, printer_ub=3, fixed_cost_factor=0.0, space_factor=0.05
    )
    scenarios.append({
        'scenario_family': 'CombinedMedicalEmergency',
        'scenario_name': 'Combined medical emergency (moderate)',
        'scenario_value': 1.0,
        'data': mild_data,
        'solve_kwargs': {'printer_lb_map': mild_lb, 'printer_ub_map': mild_ub, 'item_ub_map': {item_name: 50}},
    })

    severe_data = scale_connector_demand(data, 20.0, item_name=item_name)
    severe_data = scale_connector_shortage_penalty(severe_data, 20.0, item_name=item_name)
    severe_data = scale_connector_traditional_supply(
        severe_data, purchase_factor=25.0, transport_factor=25.0, item_name=item_name
    )
    severe_data, severe_lb, severe_ub = make_existing_printer_network(
        severe_data, printer_lb=1, printer_ub=5, fixed_cost_factor=0.0, space_factor=0.05
    )
    scenarios.append({
        'scenario_family': 'CombinedMedicalEmergency',
        'scenario_name': 'Combined medical emergency (severe)',
        'scenario_value': 2.0,
        'data': severe_data,
        'solve_kwargs': {'printer_lb_map': severe_lb, 'printer_ub_map': severe_ub, 'item_ub_map': {item_name: 20}},
    })
    return scenarios


def run_targeted_scenarios(data, zeta, max_iters, tl_master, tl_sp):
    scenarios = []

    for factor in [0.8, 1.0, 1.2]:
        scenarios.append({
            'scenario_family': 'Budget',
            'scenario_name': f'Budget x{factor:.1f}',
            'scenario_value': factor,
            'data': scale_budget(data, factor),
        })

    for factor in [1.0, 5.0, 10.0]:
        scenarios.append({
            'scenario_family': 'ConnectorDemandShock',
            'scenario_name': f'Connector demand x{factor:.1f}',
            'scenario_value': factor,
            'data': scale_connector_demand(data, factor),
        })

    for factor in [1.0, 0.5, 0.25]:
        scenarios.append({
            'scenario_family': 'PrinterFixedCost',
            'scenario_name': f'Printer fixed cost x{factor:.2f}',
            'scenario_value': factor,
            'data': scale_printer_fixed_cost(data, factor),
        })

    rows = []
    for sc in scenarios:
        res = exp.solve_ccg(
            sc['data'], zeta, max_iters=max_iters, tol=1e-3,
            tl_master=tl_master, tl_sp=tl_sp
        )
        eval_full = exp.evaluate_policy_on_dataset(sc['data'], res)
        rows.append({
            'scenario_family': sc['scenario_family'],
            'scenario_name': sc['scenario_name'],
            'scenario_value': sc['scenario_value'],
            'obj_M': res['obj_M'],
            'fs_M': res['fs_M'],
            'sum_xi': res['sum_xi'],
            'sum_xk': res['sum_xk'],
            'avg_mk': res['avg_mk'],
            'mean_total_M': eval_full['mean_total_M'],
            'aggregate_shortage_rate': eval_full['aggregate_shortage_rate'],
            'mean_wait_share': eval_full['mean_wait_share'],
            'mean_service_level': eval_full['mean_service_level'],
        })
        print(f"Scenario completed: {sc['scenario_name']}")
    return rows


def run_activation_scenarios(data, zeta, max_iters, tl_master, tl_sp):
    item_name = 'Respiratory_Circuit_Connector'
    scenarios = []

    for factor in [5.0, 10.0]:
        scenarios.append({
            'scenario_family': 'ConnectorShortagePenalty',
            'scenario_name': f'Connector shortage penalty x{factor:.1f}',
            'scenario_value': factor,
            'data': scale_connector_shortage_penalty(data, factor, item_name=item_name),
            'solve_kwargs': {},
        })

    for factor in [10.0, 25.0]:
        scenarios.append({
            'scenario_family': 'ConnectorSupplyDisruption',
            'scenario_name': f'Connector traditional cost x{factor:.1f}',
            'scenario_value': factor,
            'data': scale_connector_traditional_supply(
                data, purchase_factor=factor, transport_factor=factor, item_name=item_name
            ),
            'solve_kwargs': {},
        })

    preexisting_data, preexisting_lb, preexisting_ub = make_existing_printer_network(
        data, printer_lb=1, printer_ub=1, fixed_cost_factor=0.0, space_factor=0.05
    )
    scenarios.append({
        'scenario_family': 'ExistingPrinterNetwork',
        'scenario_name': 'Pre-existing printer network',
        'scenario_value': 1.0,
        'data': preexisting_data,
        'solve_kwargs': {'printer_lb_map': preexisting_lb, 'printer_ub_map': preexisting_ub},
    })

    for cap in [50, 20]:
        scenarios.append({
            'scenario_family': 'ConnectorInventoryCap',
            'scenario_name': f'Connector inventory cap {cap}',
            'scenario_value': float(cap),
            'data': data,
            'solve_kwargs': {'item_ub_map': {item_name: cap}},
        })

    scenarios.extend(build_combined_activation_scenarios(data, item_name=item_name))

    rows = []
    for sc in scenarios:
        res = exp.solve_ccg(
            sc['data'], zeta, max_iters=max_iters, tol=1e-3,
            tl_master=tl_master, tl_sp=tl_sp, **sc['solve_kwargs']
        )
        eval_full = exp.evaluate_policy_on_dataset(sc['data'], res)
        rows.append({
            'scenario_family': sc['scenario_family'],
            'scenario_name': sc['scenario_name'],
            'scenario_value': sc['scenario_value'],
            'obj_M': res['obj_M'],
            'fs_M': res['fs_M'],
            'sum_xi': res['sum_xi'],
            'sum_xk': res['sum_xk'],
            'avg_mk': res['avg_mk'],
            'mean_total_M': eval_full['mean_total_M'],
            'aggregate_shortage_rate': eval_full['aggregate_shortage_rate'],
            'mean_wait_share': eval_full['mean_wait_share'],
            'mean_service_level': eval_full['mean_service_level'],
        })
        print(f"Activation scenario completed: {sc['scenario_name']}")
    return rows


def run_activation_value_comparison(data, zeta, max_iters, tl_master, tl_sp):
    rows = []
    for sc in build_combined_activation_scenarios(data):
        full = exp.solve_ccg(
            sc['data'], zeta, max_iters=max_iters, tol=1e-3,
            tl_master=tl_master, tl_sp=tl_sp, **sc['solve_kwargs']
        )
        no3d = exp.solve_ccg(
            sc['data'], zeta, max_iters=max_iters, tol=1e-3,
            tl_master=tl_master, tl_sp=tl_sp, disable_3d=True
        )
        full_eval = exp.evaluate_policy_on_dataset(sc['data'], full)
        no3d_eval = exp.evaluate_policy_on_dataset(sc['data'], no3d)
        rows.append({
            'scenario_name': sc['scenario_name'],
            'policy': 'Full 3D-enabled',
            'sum_xk': full['sum_xk'],
            'mean_total_M': full_eval['mean_total_M'],
            'aggregate_shortage_rate': full_eval['aggregate_shortage_rate'],
            'mean_service_level': full_eval['mean_service_level'],
        })
        rows.append({
            'scenario_name': sc['scenario_name'],
            'policy': 'No 3D',
            'sum_xk': no3d['sum_xk'],
            'mean_total_M': no3d_eval['mean_total_M'],
            'aggregate_shortage_rate': no3d_eval['aggregate_shortage_rate'],
            'mean_service_level': no3d_eval['mean_service_level'],
        })
        print(f"Activation value comparison completed: {sc['scenario_name']}")
    return rows


def write_markdown(path: Path, baseline_rows, benchmark_summary, value3d_summary):
    lines = ['# UK Real-Case Experiment Output Summary', '', '## Baseline solve', '']
    for row in baseline_rows:
        lines.append(f"- {row['model']}: obj={row['obj_M']:.3f} M, total inventory={row['sum_xi']}, total printers={row['sum_xk']}, mean service={row['mean_service_level']:.2%}")
    lines.extend(['', '## Leave-one-out benchmarking', ''])
    for row in benchmark_summary:
        lines.append(f"- {row['model']}: mean test cost={row['mean_test_total_M']:.3f} M, mean DIS={row['mean_dis']:.2%}, mean shortage={row['mean_shortage_rate']:.2%}")
    lines.extend(['', '## 3D value comparison', ''])
    for row in value3d_summary:
        lines.append(f"- {row['policy']}: mean test cost={row['mean_test_total_M']:.3f} M, mean shortage={row['mean_shortage_rate']:.2%}, mean service={row['mean_service_level']:.2%}")
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved: {path}')


def save_figure(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f'Saved: {path}')


def plot_baseline_summary(path: Path, baseline_rows):
    labels = [row['model'] for row in baseline_rows]
    xs = np.arange(len(labels))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(xs - width / 2, [row['sum_xi'] for row in baseline_rows], width=width,
                color="0.75", edgecolor="black", label="Inventory units")
    axes[0].bar(xs + width / 2, [row['sum_xk'] for row in baseline_rows], width=width,
                color="0.30", edgecolor="black", label="Printer count")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("Quantity", fontweight="bold")
    axes[0].set_title("(a) First-stage deployment", fontweight="bold", pad=10)
    axes[0].grid(True, axis="y", ls=":", alpha=0.3)
    axes[0].legend()

    axes[1].bar(xs - width / 2, [row['mean_total_M'] for row in baseline_rows], width=width,
                color="0.55", edgecolor="black", label="Mean total cost")
    axes[1].bar(xs + width / 2, [100 * row['mean_service_level'] for row in baseline_rows], width=width,
                color="0.88", edgecolor="black", label="Service level (%)")
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels)
    axes[1].set_ylabel("Value", fontweight="bold")
    axes[1].set_title("(b) Baseline performance", fontweight="bold", pad=10)
    axes[1].grid(True, axis="y", ls=":", alpha=0.3)
    axes[1].legend()

    fig.suptitle("UK Case Figure 1  Baseline Real-Case Comparison", fontweight="bold", y=1.03)
    save_figure(fig, path)


def plot_leave_one_out_benchmark(path: Path, benchmark_rows):
    models = ["DRO", "SAA", "Deterministic"]
    data_by_model = {m: [row for row in benchmark_rows if row['model'] == m] for m in models}
    xs = np.arange(len(models))
    mean_dis = [100 * mean(float(row['dis']) for row in data_by_model[m]) for m in models]
    mean_shortage = [100 * mean(float(row['shortage_rate']) for row in data_by_model[m]) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    axes[0].bar(xs, mean_dis, color=["0.25", "0.65", "0.85"], edgecolor="black")
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(models)
    axes[0].set_ylabel("Mean DIS (%)", fontweight="bold")
    axes[0].set_title("(a) Mean disappointment", fontweight="bold", pad=10)
    axes[0].grid(True, axis="y", ls=":", alpha=0.3)

    axes[1].boxplot(
        [[100 * float(row['shortage_rate']) for row in data_by_model[m]] for m in models],
        tick_labels=models,
        patch_artist=True,
        showfliers=False,
        boxprops={"facecolor": "0.85", "edgecolor": "black"},
        medianprops={"color": "black"},
    )
    axes[1].set_ylabel("Shortage rate (%)", fontweight="bold")
    axes[1].set_title("(b) Leave-one-out shortage distribution", fontweight="bold", pad=10)
    axes[1].grid(True, axis="y", ls=":", alpha=0.3)

    fig.suptitle("UK Case Figure 2  Leave-One-Out Benchmarking", fontweight="bold", y=1.03)
    save_figure(fig, path)


def plot_value_of_3d(path: Path, value3d_rows):
    policies = ["Full 3D-enabled", "No 3D"]
    subset = {p: [row for row in value3d_rows if row['policy'] == p] for p in policies}
    metrics = {
        "Mean test cost (M)": [mean(float(row['test_total_M']) for row in subset[p]) for p in policies],
        "Mean shortage rate (%)": [100 * mean(float(row['shortage_rate']) for row in subset[p]) for p in policies],
        "Mean service level (%)": [100 * mean(float(row['service_level']) for row in subset[p]) for p in policies],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    shades = ["0.35", "0.82"]
    for ax, (title, vals) in zip(axes, metrics.items()):
        ax.bar(np.arange(len(policies)), vals, color=shades, edgecolor="black")
        ax.set_xticks(np.arange(len(policies)))
        ax.set_xticklabels(policies)
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(True, axis="y", ls=":", alpha=0.3)
    fig.suptitle("UK Case Figure 3  Value of 3D Printing", fontweight="bold", y=1.03)
    save_figure(fig, path)


def plot_targeted_scenarios(path: Path, scenario_rows):
    families = ['Budget', 'ConnectorDemandShock', 'PrinterFixedCost']
    titles = {
        'Budget': 'Budget scenarios',
        'ConnectorDemandShock': 'Connector-demand shock scenarios',
        'PrinterFixedCost': 'Printer fixed-cost scenarios',
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    for r, family in enumerate(families):
        subset = sorted(
            [row for row in scenario_rows if row['scenario_family'] == family],
            key=lambda row: float(row['scenario_value'])
        )
        xs = [float(row['scenario_value']) for row in subset]
        axes[r, 0].plot(xs, [row['sum_xk'] for row in subset], color='black', marker='o', lw=1.8)
        axes[r, 0].plot(xs, [row['sum_xi'] for row in subset], color='0.5', marker='s', lw=1.6)
        axes[r, 0].set_title(f"({chr(97 + 2*r)}) {titles[family]}", fontweight='bold', pad=10)
        axes[r, 0].set_ylabel("Deployment", fontweight='bold')
        axes[r, 0].grid(True, ls=':', alpha=0.3)
        axes[r, 0].legend(['Printer count', 'Inventory units'])

        axes[r, 1].plot(xs, [100 * row['aggregate_shortage_rate'] for row in subset], color='black', marker='o', lw=1.8)
        axes[r, 1].plot(xs, [row['mean_total_M'] for row in subset], color='0.6', marker='^', lw=1.6)
        axes[r, 1].set_title(f"({chr(98 + 2*r)}) {titles[family]} performance", fontweight='bold', pad=10)
        axes[r, 1].set_ylabel("Value", fontweight='bold')
        axes[r, 1].grid(True, ls=':', alpha=0.3)
        axes[r, 1].legend(['Shortage rate (%)', 'Mean total cost (M)'])

        for c in range(2):
            axes[r, c].set_xlabel("Scenario level", fontweight='bold')

    fig.suptitle("UK Case Figure 4  Targeted Policy Scenarios", fontweight='bold', y=1.01)
    save_figure(fig, path)


def plot_activation_scenarios(path: Path, scenario_rows):
    labels = [row['scenario_name'] for row in scenario_rows]
    xs = np.arange(len(labels))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    axes[0].bar(xs, [row['sum_xk'] for row in scenario_rows], color='0.3', edgecolor='black')
    axes[0].set_title("(a) Printer deployment", fontweight='bold', pad=10)
    axes[0].set_ylabel("Total printers", fontweight='bold')
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels(labels, rotation=35, ha='right')
    axes[0].grid(True, axis='y', ls=':', alpha=0.3)

    axes[1].bar(xs, [row['mean_total_M'] for row in scenario_rows], color='0.65', edgecolor='black')
    axes[1].set_title("(b) Mean total cost", fontweight='bold', pad=10)
    axes[1].set_ylabel("Mean total cost (M)", fontweight='bold')
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels(labels, rotation=35, ha='right')
    axes[1].grid(True, axis='y', ls=':', alpha=0.3)

    axes[2].bar(xs, [100 * row['mean_service_level'] for row in scenario_rows], color='0.85', edgecolor='black')
    axes[2].set_title("(c) Mean service level", fontweight='bold', pad=10)
    axes[2].set_ylabel("Service level (%)", fontweight='bold')
    axes[2].set_xticks(xs)
    axes[2].set_xticklabels(labels, rotation=35, ha='right')
    axes[2].grid(True, axis='y', ls=':', alpha=0.3)

    fig.suptitle("UK Case Figure 5  3D-Activation Scenarios", fontweight='bold', y=1.02)
    save_figure(fig, path)


def write_activation_summary(path: Path, scenario_rows):
    lines = ['# UK Real-Case Activation Scenario Summary', '']
    activated = [row for row in scenario_rows if int(row.get('sum_xk', 0)) > 0]
    if activated:
        lines.append('## Activated scenarios')
        for row in activated:
            lines.append(
                f"- {row['scenario_name']}: printers={int(row['sum_xk'])}, "
                f"mean_total={float(row['mean_total_M']):.3f} M, "
                f"service={100 * float(row['mean_service_level']):.2f}%"
            )
    else:
        lines.append('## Activated scenarios')
        lines.append('- None of the second-round activation scenarios triggered endogenous printer deployment.')

    lines.extend(['', '## All scenarios', ''])
    for row in scenario_rows:
        lines.append(
            f"- {row['scenario_name']}: printers={int(row['sum_xk'])}, "
            f"inventory={int(row['sum_xi'])}, mean_total={float(row['mean_total_M']):.3f} M, "
            f"shortage={100 * float(row['aggregate_shortage_rate']):.2f}%, "
            f"service={100 * float(row['mean_service_level']):.2f}%"
        )
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved: {path}')


def write_activation_value_summary(path: Path, rows):
    lines = ['# UK Activation Scenario Value-of-3D Summary', '']
    scenario_names = []
    for row in rows:
        if row['scenario_name'] not in scenario_names:
            scenario_names.append(row['scenario_name'])

    for name in scenario_names:
        subset = [row for row in rows if row['scenario_name'] == name]
        full = next(row for row in subset if row['policy'] == 'Full 3D-enabled')
        no3d = next(row for row in subset if row['policy'] == 'No 3D')
        cost_delta = float(no3d['mean_total_M']) - float(full['mean_total_M'])
        lines.append(f"## {name}")
        lines.append(
            f"- Full 3D-enabled: printers={int(full['sum_xk'])}, mean_total={float(full['mean_total_M']):.3f} M, "
            f"shortage={100 * float(full['aggregate_shortage_rate']):.2f}%, service={100 * float(full['mean_service_level']):.2f}%"
        )
        lines.append(
            f"- No 3D: printers={int(no3d['sum_xk'])}, mean_total={float(no3d['mean_total_M']):.3f} M, "
            f"shortage={100 * float(no3d['aggregate_shortage_rate']):.2f}%, service={100 * float(no3d['mean_service_level']):.2f}%"
        )
        lines.append(f"- Cost difference (No3D - Full3D): {cost_delta:.3f} M")
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved: {path}')


def plot_activation_value_comparison(path: Path, rows):
    scenario_names = []
    for row in rows:
        if row['scenario_name'] not in scenario_names:
            scenario_names.append(row['scenario_name'])
    policies = ['Full 3D-enabled', 'No 3D']
    width = 0.34
    xs = np.arange(len(scenario_names))

    def vals(metric, policy):
        out = []
        for name in scenario_names:
            row = next(r for r in rows if r['scenario_name'] == name and r['policy'] == policy)
            out.append(float(row[metric]))
        return out

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for idx, (metric, title, scale) in enumerate([
        ('mean_total_M', 'Mean total cost (M)', 1.0),
        ('aggregate_shortage_rate', 'Aggregate shortage rate (%)', 100.0),
        ('mean_service_level', 'Mean service level (%)', 100.0),
    ]):
        for off, policy, shade in [(-width/2, policies[0], '0.35'), (width/2, policies[1], '0.82')]:
            axes[idx].bar(xs + off, [scale * v for v in vals(metric, policy)], width=width,
                          color=shade, edgecolor='black', label=policy)
        axes[idx].set_title(title, fontweight='bold', pad=10)
        axes[idx].set_xticks(xs)
        axes[idx].set_xticklabels(scenario_names, rotation=20, ha='right')
        axes[idx].grid(True, axis='y', ls=':', alpha=0.3)
    axes[0].legend()
    fig.suptitle('UK Case Figure 6  Value of 3D Under Activated Medical-Emergency Scenarios', fontweight='bold', y=1.03)
    save_figure(fig, path)


def build_integrated_activation_rows(scenario_rows, activation_rows, activation_value_rows):
    rows = []

    def add_screening_row(row, family_label):
        rows.append({
            'analysis_stage': family_label,
            'scenario_name': row['scenario_name'],
            'policy': 'DRO',
            'sum_xk': row['sum_xk'],
            'sum_xi': row.get('sum_xi', ''),
            'mean_total_M': row['mean_total_M'],
            'aggregate_shortage_rate': row['aggregate_shortage_rate'],
            'mean_service_level': row['mean_service_level'],
        })

    targeted_keep = {
        ('Budget', '0.8'): 'Single-factor screening',
        ('Budget', '1.0'): 'Single-factor screening',
        ('Budget', '1.2'): 'Single-factor screening',
        ('ConnectorDemandShock', '10.0'): 'Single-factor screening',
        ('PrinterFixedCost', '0.25'): 'Single-factor screening',
    }
    for row in scenario_rows:
        key = (row['scenario_family'], str(row['scenario_value']))
        if key in targeted_keep:
            add_screening_row(row, targeted_keep[key])

    activation_keep = {
        'Connector shortage penalty x10.0',
        'Connector traditional cost x25.0',
        'Pre-existing printer network',
        'Connector inventory cap 20',
        'Combined medical emergency (moderate)',
        'Combined medical emergency (severe)',
    }
    for row in activation_rows:
        if row['scenario_name'] in activation_keep:
            add_screening_row(row, 'Activation trigger')

    for row in activation_value_rows:
        rows.append({
            'analysis_stage': 'Activated value comparison',
            'scenario_name': row['scenario_name'],
            'policy': row['policy'],
            'sum_xk': row['sum_xk'],
            'sum_xi': '',
            'mean_total_M': row['mean_total_M'],
            'aggregate_shortage_rate': row['aggregate_shortage_rate'],
            'mean_service_level': row['mean_service_level'],
        })
    return rows


def plot_integrated_activation_analysis(path: Path, scenario_rows, activation_rows, activation_value_rows):
    label_map = {
        'Budget x0.8': 'Budget x0.8',
        'Budget x1.0': 'Budget x1.0',
        'Budget x1.2': 'Budget x1.2',
        'Connector demand x10.0': 'Connector demand x10',
        'Printer fixed cost x0.25': 'Printer fixed cost x0.25',
        'Connector shortage penalty x10.0': 'Shortage penalty x10',
        'Connector traditional cost x25.0': 'Traditional cost x25',
        'Pre-existing printer network': 'Pre-existing network',
        'Connector inventory cap 20': 'Connector inventory cap 20',
        'Combined medical emergency (moderate)': 'Medical emergency moderate',
        'Combined medical emergency (severe)': 'Medical emergency severe',
    }
    selected_names = list(label_map.keys())
    selected_rows = []
    for name in selected_names:
        row = next((r for r in scenario_rows if r['scenario_name'] == name), None)
        if row is None:
            row = next((r for r in activation_rows if r['scenario_name'] == name), None)
        if row is not None:
            selected_rows.append(row)

    scenario_names = []
    for row in activation_value_rows:
        if row['scenario_name'] not in scenario_names:
            scenario_names.append(row['scenario_name'])
    policies = ['Full 3D-enabled', 'No 3D']

    def value_row(name, policy):
        return next(row for row in activation_value_rows
                    if row['scenario_name'] == name and row['policy'] == policy)

    ys = np.arange(len(selected_rows))
    value_xs = np.arange(len(scenario_names))
    width = 0.34

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 6.6),
        gridspec_kw={'width_ratios': [1.35, 1.0, 1.0]}
    )

    printer_vals = [float(row['sum_xk']) for row in selected_rows]
    bars = axes[0].barh(ys, printer_vals, color='0.35', edgecolor='black')
    axes[0].set_title('(a) Printer activation path', fontweight='bold', pad=10)
    axes[0].set_xlabel('Total printers', fontweight='bold')
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([label_map[row['scenario_name']] for row in selected_rows], fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, max(printer_vals) + 2)
    axes[0].grid(True, axis='x', ls=':', alpha=0.3)
    for bar in bars:
        axes[0].text(bar.get_width() + 0.25, bar.get_y() + bar.get_height() / 2,
                     f"{bar.get_width():.0f}", ha='left', va='center', fontsize=9)

    for off, policy, shade in [(-width/2, policies[0], '0.35'), (width/2, policies[1], '0.82')]:
        axes[1].bar(
            value_xs + off,
            [float(value_row(name, policy)['mean_total_M']) for name in scenario_names],
            width=width, color=shade, edgecolor='black', label=policy
        )
    axes[1].set_title('(b) Activated-scenario cost comparison', fontweight='bold', pad=10)
    axes[1].set_ylabel('Mean total cost (M)', fontweight='bold')
    axes[1].set_xticks(value_xs)
    axes[1].set_xticklabels([label_map.get(name, name) for name in scenario_names], rotation=20, ha='right')
    axes[1].grid(True, axis='y', ls=':', alpha=0.3)
    axes[1].legend()

    for off, policy, shade in [(-width/2, policies[0], '0.35'), (width/2, policies[1], '0.82')]:
        axes[2].bar(
            value_xs + off,
            [100.0 * float(value_row(name, policy)['mean_service_level']) for name in scenario_names],
            width=width, color=shade, edgecolor='black', label=policy
        )
    axes[2].set_title('(c) Activated-scenario service comparison', fontweight='bold', pad=10)
    axes[2].set_ylabel('Mean service level (%)', fontweight='bold')
    axes[2].set_xticks(value_xs)
    axes[2].set_xticklabels([label_map.get(name, name) for name in scenario_names], rotation=20, ha='right')
    axes[2].grid(True, axis='y', ls=':', alpha=0.3)

    fig.suptitle('UK Case Figure 4  Integrated 3D-Printing Activation and Value Analysis',
                 fontweight='bold', y=0.98)
    fig.subplots_adjust(wspace=0.35)
    save_figure(fig, path)


def run_activation_value_by_item(data, zeta, max_iters, tl_master, tl_sp):
    rows = []
    for sc in build_combined_activation_scenarios(data):
        full = exp.solve_ccg(
            sc['data'], zeta, max_iters=max_iters, tol=1e-3,
            tl_master=tl_master, tl_sp=tl_sp, **sc['solve_kwargs']
        )
        no3d = exp.solve_ccg(
            sc['data'], zeta, max_iters=max_iters, tol=1e-3,
            tl_master=tl_master, tl_sp=tl_sp, disable_3d=True
        )
        full_item = exp.evaluate_policy_on_dataset_by_item(sc['data'], full)
        no3d_item = exp.evaluate_policy_on_dataset_by_item(sc['data'], no3d)
        for item_name in sc['data'].I:
            rows.append({
                'scenario_name': sc['scenario_name'],
                'policy': 'Full 3D-enabled',
                'item': item_name,
                'sum_xk': full['sum_xk'],
                'demand_total': full_item[item_name]['demand_total'],
                'shortage_units_total': full_item[item_name]['shortage_units_total'],
                'aggregate_shortage_rate': full_item[item_name]['aggregate_shortage_rate'],
                'aggregate_service_level': full_item[item_name]['aggregate_service_level'],
                'print_units_total': full_item[item_name]['print_units_total'],
                'print_share_of_demand': full_item[item_name]['print_share_of_demand'],
            })
            rows.append({
                'scenario_name': sc['scenario_name'],
                'policy': 'No 3D',
                'item': item_name,
                'sum_xk': no3d['sum_xk'],
                'demand_total': no3d_item[item_name]['demand_total'],
                'shortage_units_total': no3d_item[item_name]['shortage_units_total'],
                'aggregate_shortage_rate': no3d_item[item_name]['aggregate_shortage_rate'],
                'aggregate_service_level': no3d_item[item_name]['aggregate_service_level'],
                'print_units_total': no3d_item[item_name]['print_units_total'],
                'print_share_of_demand': no3d_item[item_name]['print_share_of_demand'],
            })
        print(f"Activation item-level comparison completed: {sc['scenario_name']}")
    return rows


def write_activation_value_by_item_summary(path: Path, rows):
    lines = ['# UK Activation Scenario Item-Level Summary', '']
    scenario_names = []
    for row in rows:
        if row['scenario_name'] not in scenario_names:
            scenario_names.append(row['scenario_name'])

    for name in scenario_names:
        lines.append(f"## {name}")
        subset = [row for row in rows if row['scenario_name'] == name]
        for item_name in sorted({row['item'] for row in subset}):
            full = next(row for row in subset if row['item'] == item_name and row['policy'] == 'Full 3D-enabled')
            no3d = next(row for row in subset if row['item'] == item_name and row['policy'] == 'No 3D')
            lines.append(
                f"- {item_name}: Full3D shortage={100 * float(full['aggregate_shortage_rate']):.2f}%, "
                f"No3D shortage={100 * float(no3d['aggregate_shortage_rate']):.2f}%, "
                f"Full3D print share={100 * float(full['print_share_of_demand']):.2f}%"
            )
        lines.append('')
    path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'Saved: {path}')


def plot_activation_value_by_item(path: Path, rows):
    preferred_items = ['Drinking_Water_3Day_Kit', 'Family_Tent', 'Respiratory_Circuit_Connector']
    items = [item for item in preferred_items if item in {row['item'] for row in rows}]
    items.extend(sorted({row['item'] for row in rows if row['item'] not in items}))
    item_labels = {
        'Drinking_Water_3Day_Kit': 'Drinking water',
        'Family_Tent': 'Family tent',
        'Respiratory_Circuit_Connector': 'Respiratory connector',
    }
    item_colors = {
        'Drinking_Water_3Day_Kit': '#4C78A8',
        'Family_Tent': '#F58518',
        'Respiratory_Circuit_Connector': '#54A24B',
    }
    scenario_names = []
    for row in rows:
        if row['scenario_name'] not in scenario_names:
            scenario_names.append(row['scenario_name'])

    scenario_labels = {
        'Combined medical emergency (moderate)': 'Medical emergency: moderate',
        'Combined medical emergency (severe)': 'Medical emergency: severe',
    }
    fig, axes = plt.subplots(1, len(scenario_names), figsize=(7.4 * len(scenario_names), 5.6), sharey=True)
    if len(scenario_names) == 1:
        axes = np.array([axes])
    xs = np.arange(len(items))

    for c, scenario_name in enumerate(scenario_names):
        subset = [row for row in rows if row['scenario_name'] == scenario_name]
        full_shortage = [
            100 * float(next(r for r in subset if r['item'] == item_name and r['policy'] == 'Full 3D-enabled')['aggregate_shortage_rate'])
            for item_name in items
        ]
        no3d_shortage = [
            100 * float(next(r for r in subset if r['item'] == item_name and r['policy'] == 'No 3D')['aggregate_shortage_rate'])
            for item_name in items
        ]
        print_share = [
            100 * float(next(r for r in subset if r['item'] == item_name and r['policy'] == 'Full 3D-enabled')['print_share_of_demand'])
            for item_name in items
        ]

        ax = axes[c]
        bars = ax.bar(
            xs,
            full_shortage,
            width=0.55,
            color=[item_colors.get(item_name, '0.6') for item_name in items],
            edgecolor='black',
            alpha=0.85,
            label='Full 3D shortage'
        )
        ax.plot(xs, no3d_shortage, color='black', marker='o', lw=2.0,
                label='No 3D shortage')
        ax.plot(xs, print_share, color='black', marker='^', lw=1.8, ls='--',
                label='Full 3D print share')

        for bar, value in zip(bars, full_shortage):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 1.4,
                    f"{value:.1f}", ha='center', va='bottom', fontsize=8)

        ax.set_title(f"({chr(97 + c)}) {scenario_labels.get(scenario_name, scenario_name)}",
                     fontweight='bold', pad=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([item_labels.get(item_name, item_name) for item_name in items],
                           rotation=0, ha='center')
        ax.set_ylim(0, 108)
        ax.grid(True, axis='y', ls=':', alpha=0.3)
        ax.set_xlabel('Item type', fontweight='bold')

    axes[0].set_ylabel('Rate (%)', fontweight='bold')
    item_handles = [
        Patch(facecolor=item_colors.get(item_name, '0.6'), edgecolor='black',
              label=item_labels.get(item_name, item_name))
        for item_name in items
    ]
    line_handles = [
        Line2D([0], [0], color='black', marker='o', lw=2.0, label='No 3D shortage'),
        Line2D([0], [0], color='black', marker='^', lw=1.8, ls='--', label='Full 3D print share'),
    ]
    fig.legend(handles=item_handles + line_handles, loc='lower center',
               ncol=5, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle('UK Case Figure 7  Item-Level Shortage and Print-Share Effects of 3D Printing',
                 fontweight='bold', y=0.98)
    fig.subplots_adjust(bottom=0.20, top=0.86, wspace=0.18)
    save_figure(fig, path)


def write_chinese_case_draft(path: Path, data, baseline_rows, benchmark_summary, value3d_summary,
                             scenario_rows, activation_rows, activation_value_rows, activation_item_rows):
    def as_float(row, key, default=0.0):
        try:
            return float(row.get(key, default))
        except Exception:
            return float(default)

    def as_int(row, key, default=0):
        try:
            return int(round(float(row.get(key, default))))
        except Exception:
            return int(default)

    def pct(value):
        return f"{100.0 * float(value):.2f}%"

    def require_row(mapping, key):
        return mapping.get(key, {})

    baseline_map = {row['model']: row for row in baseline_rows}
    bench_map = {row['model']: row for row in benchmark_summary}
    value_map = {row['policy']: row for row in value3d_summary}
    scenario_map = {(row['scenario_family'], str(row['scenario_value'])): row for row in scenario_rows}
    activation_map = {row['scenario_name']: row for row in activation_rows}
    activation_value_map = {(row['scenario_name'], row['policy']): row for row in activation_value_rows}
    activation_item_map = {(row['scenario_name'], row['policy'], row['item']): row for row in activation_item_rows}

    dro_base = require_row(baseline_map, 'DRO')
    saa_base = require_row(baseline_map, 'SAA')
    no3d_base = require_row(baseline_map, 'No3D-DRO')
    dro_loo = require_row(bench_map, 'DRO')
    saa_loo = require_row(bench_map, 'SAA')
    det_loo = require_row(bench_map, 'Deterministic')
    full3d = require_row(value_map, 'Full 3D-enabled')
    no3d = require_row(value_map, 'No 3D')

    budget_low = require_row(scenario_map, ('Budget', '0.8'))
    budget_high = require_row(scenario_map, ('Budget', '1.2'))
    shock_high = require_row(scenario_map, ('ConnectorDemandShock', '10.0'))
    printer_lowcost = require_row(scenario_map, ('PrinterFixedCost', '0.25'))

    preexisting = require_row(activation_map, 'Pre-existing printer network')
    moderate = require_row(activation_map, 'Combined medical emergency (moderate)')
    severe = require_row(activation_map, 'Combined medical emergency (severe)')

    moderate_full = require_row(activation_value_map, ('Combined medical emergency (moderate)', 'Full 3D-enabled'))
    moderate_no3d = require_row(activation_value_map, ('Combined medical emergency (moderate)', 'No 3D'))
    severe_full = require_row(activation_value_map, ('Combined medical emergency (severe)', 'Full 3D-enabled'))
    severe_no3d = require_row(activation_value_map, ('Combined medical emergency (severe)', 'No 3D'))

    water_mod_full = require_row(activation_item_map, ('Combined medical emergency (moderate)', 'Full 3D-enabled', 'Drinking_Water_3Day_Kit'))
    water_mod_no3d = require_row(activation_item_map, ('Combined medical emergency (moderate)', 'No 3D', 'Drinking_Water_3Day_Kit'))
    tent_mod_full = require_row(activation_item_map, ('Combined medical emergency (moderate)', 'Full 3D-enabled', 'Family_Tent'))
    conn_mod_full = require_row(activation_item_map, ('Combined medical emergency (moderate)', 'Full 3D-enabled', 'Respiratory_Circuit_Connector'))
    conn_mod_no3d = require_row(activation_item_map, ('Combined medical emergency (moderate)', 'No 3D', 'Respiratory_Circuit_Connector'))
    water_sev_full = require_row(activation_item_map, ('Combined medical emergency (severe)', 'Full 3D-enabled', 'Drinking_Water_3Day_Kit'))
    water_sev_no3d = require_row(activation_item_map, ('Combined medical emergency (severe)', 'No 3D', 'Drinking_Water_3Day_Kit'))
    tent_sev_full = require_row(activation_item_map, ('Combined medical emergency (severe)', 'Full 3D-enabled', 'Family_Tent'))
    conn_sev_full = require_row(activation_item_map, ('Combined medical emergency (severe)', 'Full 3D-enabled', 'Respiratory_Circuit_Connector'))
    conn_sev_no3d = require_row(activation_item_map, ('Combined medical emergency (severe)', 'No 3D', 'Respiratory_Circuit_Connector'))

    dro_xk = as_int(dro_base, 'sum_xk')
    dro_mk = as_float(dro_base, 'avg_mk')
    baseline_3d_active = (dro_xk > 0) or (dro_mk > 1e-6)
    baseline_text = (
        '????????????????3D ?????????????????????????????????'
        if baseline_3d_active else
        '???????????????????????????????3D ??????????DRO ? No3D-DRO ??????????????????????????????????????????????'
    )

    dro_mean_cost = as_float(dro_loo, 'mean_test_total_M')
    saa_mean_cost = as_float(saa_loo, 'mean_test_total_M')
    det_mean_cost = as_float(det_loo, 'mean_test_total_M')
    dro_max_cost = as_float(dro_loo, 'max_test_total_M')
    det_max_cost = as_float(det_loo, 'max_test_total_M')
    dro_saa_text = (
        'DRO ? SAA ???????????????????????????????????????????????'
        if abs(dro_mean_cost - saa_mean_cost) <= 0.5 else
        'DRO ?? SAA ????????????????'
    )
    det_text = (
        f"???????????????????? {det_mean_cost:.3f} ????????????? {det_max_cost:.3f} ?????? DRO ? {dro_max_cost:.3f} ????"
        if det_mean_cost > dro_mean_cost else
        f"????????????? {det_mean_cost:.3f} ????"
    )

    full_cost = as_float(full3d, 'mean_test_total_M')
    no3d_cost = as_float(no3d, 'mean_test_total_M')
    full_short = as_float(full3d, 'mean_shortage_rate')
    no3d_short = as_float(no3d, 'mean_shortage_rate')
    value_text = (
        '???????????????? 3D ????????????????????????????????????????????????????????'
        if full_cost >= no3d_cost and full_short >= no3d_short - 1e-9 else
        '???3D ???????????????????????'
    )

    values = {
        'n_samples': len(data.N),
        'n_nodes': len(data.J),
        'A_max': float(getattr(data, 'A_max', 0.0)),
        'B_m': float(getattr(data, 'B', 0.0)) * float(getattr(data, 'money_scale', 10000)) / 1e6,
        'dro_sum_xi': as_int(dro_base, 'sum_xi'),
        'dro_sum_xk': as_int(dro_base, 'sum_xk'),
        'dro_avg_mk': as_float(dro_base, 'avg_mk'),
        'dro_obj': as_float(dro_base, 'obj_M'),
        'dro_service': pct(as_float(dro_base, 'mean_service_level')),
        'saa_obj': as_float(saa_base, 'obj_M'),
        'no3d_obj': as_float(no3d_base, 'obj_M'),
        'baseline_text': baseline_text,
        'dro_mean_cost': dro_mean_cost,
        'saa_mean_cost': saa_mean_cost,
        'det_mean_cost': det_mean_cost,
        'dro_max_cost': dro_max_cost,
        'det_max_cost': det_max_cost,
        'dro_dis': pct(as_float(dro_loo, 'mean_dis')),
        'saa_dis': pct(as_float(saa_loo, 'mean_dis')),
        'det_dis': pct(as_float(det_loo, 'mean_dis')),
        'dro_saa_text': dro_saa_text,
        'det_text': det_text,
        'full_cost': full_cost,
        'no3d_cost': no3d_cost,
        'full_short': pct(full_short),
        'no3d_short': pct(no3d_short),
        'full_service': pct(as_float(full3d, 'mean_service_level')),
        'no3d_service': pct(as_float(no3d, 'mean_service_level')),
        'value_text': value_text,
        'budget_low_xi': as_int(budget_low, 'sum_xi'),
        'budget_high_xi': as_int(budget_high, 'sum_xi'),
        'budget_low_cost': as_float(budget_low, 'mean_total_M'),
        'budget_high_cost': as_float(budget_high, 'mean_total_M'),
        'shock_high_cost': as_float(shock_high, 'mean_total_M'),
        'shock_high_xi': as_int(shock_high, 'sum_xi'),
        'shock_high_xk': as_int(shock_high, 'sum_xk'),
        'printer_lowcost_cost': as_float(printer_lowcost, 'mean_total_M'),
        'printer_lowcost_xk': as_int(printer_lowcost, 'sum_xk'),
        'preexisting_xk': as_int(preexisting, 'sum_xk'),
        'moderate_xk': as_int(moderate, 'sum_xk'),
        'severe_xk': as_int(severe, 'sum_xk'),
        'moderate_full_cost': as_float(moderate_full, 'mean_total_M'),
        'moderate_no3d_cost': as_float(moderate_no3d, 'mean_total_M'),
        'moderate_saving': as_float(moderate_no3d, 'mean_total_M') - as_float(moderate_full, 'mean_total_M'),
        'severe_full_cost': as_float(severe_full, 'mean_total_M'),
        'severe_no3d_cost': as_float(severe_no3d, 'mean_total_M'),
        'severe_saving': as_float(severe_no3d, 'mean_total_M') - as_float(severe_full, 'mean_total_M'),
        'conn_mod_no3d_short': pct(as_float(conn_mod_no3d, 'aggregate_shortage_rate')),
        'conn_mod_full_short': pct(as_float(conn_mod_full, 'aggregate_shortage_rate')),
        'conn_mod_print': pct(as_float(conn_mod_full, 'print_share_of_demand')),
        'water_mod_no3d_short': pct(as_float(water_mod_no3d, 'aggregate_shortage_rate')),
        'water_mod_full_short': pct(as_float(water_mod_full, 'aggregate_shortage_rate')),
        'tent_mod_short': pct(as_float(tent_mod_full, 'aggregate_shortage_rate')),
        'conn_sev_no3d_short': pct(as_float(conn_sev_no3d, 'aggregate_shortage_rate')),
        'conn_sev_full_short': pct(as_float(conn_sev_full, 'aggregate_shortage_rate')),
        'conn_sev_print': pct(as_float(conn_sev_full, 'print_share_of_demand')),
        'water_sev_no3d_short': pct(as_float(water_sev_no3d, 'aggregate_shortage_rate')),
        'water_sev_full_short': pct(as_float(water_sev_full, 'aggregate_shortage_rate')),
        'tent_sev_short': pct(as_float(tent_sev_full, 'aggregate_shortage_rate')),
    }

    template_path = BASE_DIR / 'uk_flood_case_draft' / 'uk_real_case_chapter_template_zh.md'
    template = template_path.read_text(encoding='utf-8-sig')
    rendered = template.format(**values)
    path.write_text(rendered, encoding='utf-8-sig')
    print(f'Saved: {path}')



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-path', default=str(DEFAULT_DATA))
    ap.add_argument('--zeta', type=float, default=10.0)
    ap.add_argument('--max-iters', type=int, default=120)
    ap.add_argument('--tl-master', type=int, default=300)
    ap.add_argument('--tl-sp', type=int, default=30)
    ap.add_argument('--output-dir', default=str(BASE_DIR / 'uk_flood_case_results'))
    ap.add_argument('--save-diagnostic-plots', action='store_true',
                    help='Also save the older separate targeted/activation/value diagnostic plots.')
    args = ap.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = exp.load_instance_from_json(args.data_path)
    baseline_rows = baseline_case(data, args.zeta, args.max_iters, args.tl_master, args.tl_sp)
    write_csv(output_dir / 'uk_baseline_summary.csv', baseline_rows,
              ['model','obj_M','fs_M','sum_xi','sum_xk','avg_mk','mean_total_M','aggregate_shortage_rate','mean_wait_share','mean_service_level','iters','wall_time'])

    benchmark_rows, value3d_rows = leave_one_out(data, args.zeta, args.max_iters, args.tl_master, args.tl_sp)
    write_csv(output_dir / 'uk_leave_one_out_benchmark.csv', benchmark_rows,
              ['holdout_sample','model','ref_total_M','test_total_M','dis','shortage_rate','wait_share','service_level'])
    write_csv(output_dir / 'uk_leave_one_out_3d_value.csv', value3d_rows,
              ['holdout_sample','policy','test_total_M','shortage_rate','wait_share','service_level','sum_xk_train'])

    benchmark_summary = summarize_rows(benchmark_rows, 'model', ['test_total_M','dis','shortage_rate','wait_share','service_level'])
    value3d_summary = summarize_rows(value3d_rows, 'policy', ['test_total_M','shortage_rate','wait_share','service_level'])
    write_csv(output_dir / 'uk_leave_one_out_benchmark_summary.csv', benchmark_summary,
              ['model','n','mean_test_total_M','max_test_total_M','min_test_total_M','mean_dis','max_dis','min_dis','mean_shortage_rate','max_shortage_rate','min_shortage_rate','mean_wait_share','max_wait_share','min_wait_share','mean_service_level','max_service_level','min_service_level'])
    write_csv(output_dir / 'uk_leave_one_out_3d_value_summary.csv', value3d_summary,
              ['policy','n','mean_test_total_M','max_test_total_M','min_test_total_M','mean_shortage_rate','max_shortage_rate','min_shortage_rate','mean_wait_share','max_wait_share','min_wait_share','mean_service_level','max_service_level','min_service_level'])
    scenario_rows = run_targeted_scenarios(data, args.zeta, args.max_iters, args.tl_master, args.tl_sp)
    write_csv(output_dir / 'uk_targeted_scenarios.csv', scenario_rows,
              ['scenario_family','scenario_name','scenario_value','obj_M','fs_M','sum_xi','sum_xk','avg_mk','mean_total_M','aggregate_shortage_rate','mean_wait_share','mean_service_level'])
    activation_rows = run_activation_scenarios(data, args.zeta, args.max_iters, args.tl_master, args.tl_sp)
    write_csv(output_dir / 'uk_activation_scenarios.csv', activation_rows,
              ['scenario_family','scenario_name','scenario_value','obj_M','fs_M','sum_xi','sum_xk','avg_mk','mean_total_M','aggregate_shortage_rate','mean_wait_share','mean_service_level'])
    activation_value_rows = run_activation_value_comparison(data, args.zeta, args.max_iters, args.tl_master, args.tl_sp)
    write_csv(output_dir / 'uk_activation_value_comparison.csv', activation_value_rows,
              ['scenario_name','policy','sum_xk','mean_total_M','aggregate_shortage_rate','mean_service_level'])
    activation_value_item_rows = run_activation_value_by_item(data, args.zeta, args.max_iters, args.tl_master, args.tl_sp)
    write_csv(output_dir / 'uk_activation_value_by_item.csv', activation_value_item_rows,
              ['scenario_name','policy','item','sum_xk','demand_total','shortage_units_total','aggregate_shortage_rate',
               'aggregate_service_level','print_units_total','print_share_of_demand'])
    integrated_activation_rows = build_integrated_activation_rows(
        scenario_rows, activation_rows, activation_value_rows
    )
    write_csv(output_dir / 'uk_3d_activation_integrated.csv', integrated_activation_rows,
              ['analysis_stage','scenario_name','policy','sum_xk','sum_xi',
               'mean_total_M','aggregate_shortage_rate','mean_service_level'])

    write_markdown(output_dir / 'uk_real_case_summary.md', baseline_rows, benchmark_summary, value3d_summary)
    write_activation_summary(output_dir / 'uk_activation_summary.md', activation_rows)
    write_activation_value_summary(output_dir / 'uk_activation_value_summary.md', activation_value_rows)
    write_activation_value_by_item_summary(output_dir / 'uk_activation_value_by_item_summary.md', activation_value_item_rows)
    plot_baseline_summary(output_dir / 'uk_case_baseline.png', baseline_rows)
    plot_leave_one_out_benchmark(output_dir / 'uk_case_leave_one_out_benchmark.png', benchmark_rows)
    plot_integrated_activation_analysis(output_dir / 'uk_case_3d_activation_integrated.png',
                                        scenario_rows, activation_rows, activation_value_rows)
    plot_activation_value_by_item(output_dir / 'uk_case_activation_value_by_item.png', activation_value_item_rows)
    if args.save_diagnostic_plots:
        plot_value_of_3d(output_dir / 'uk_case_value_of_3d.png', value3d_rows)
        plot_targeted_scenarios(output_dir / 'uk_case_targeted_scenarios.png', scenario_rows)
        plot_activation_scenarios(output_dir / 'uk_case_activation_scenarios.png', activation_rows)
        plot_activation_value_comparison(output_dir / 'uk_case_activation_value.png', activation_value_rows)
    write_chinese_case_draft(output_dir / 'uk_real_case_chapter_draft_zh.md',
                             data, baseline_rows, benchmark_summary, value3d_summary,
                             scenario_rows, activation_rows, activation_value_rows, activation_value_item_rows)


if __name__ == '__main__':
    main()
