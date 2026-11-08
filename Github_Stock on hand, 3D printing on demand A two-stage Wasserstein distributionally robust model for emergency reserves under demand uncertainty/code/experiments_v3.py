# -*- coding: utf-8 -*-
"""
================================================================================
Wasserstein-DRO Emergency Stockpile Model — Experiment Suite  v3
================================================================================
v3 Changes (two root-cause fixes):

  FIX A — Font:
      All axis labels / titles switched to English.
      Chinese characters caused "missing glyph" warnings on most systems because
      matplotlib's bundled fonts (DejaVu, etc.) have no CJK coverage.
      International journals (COR / OR) also require English captions.

  FIX B — Parameter calibration (the structural root cause):
      In v1/v2 the synthetic instance had:
        · A_max = 20000 × (n_nodes/2)  e.g. 100 000 m³ for Medium
        · B     = 5×10⁸ × (n_nodes/2) yuan  e.g. 2.5×10⁹ yuan for Medium
        · f_k   = 1.5–2.5 M yuan / printer
        · ρ_k   = 50 yuan / unit  (negligible)

      Consequence: traditional inventory x_i could always cover 100% of even the
      worst-case demand without hitting any space or budget constraint → optimal
      x_k = m_k = 0 for all experiments (3D printing never needed).

      Fix: recalibrate so traditional inventory can cover only ~65% of average
      total demand, forcing 3D printing to fill the gap:
        · A_max ← n_items × n_nodes × d̄ × h̄ × 0.65  (tight storage)
        · f_k   ← 100 K – 300 K yuan  (realistic equipment cost)
        · ρ_k   ← 200 – 500 yuan/unit  (meaningful humanitarian penalty)
        · B     ← tight enough to force trade-offs between inventory & printers

      The key calibration rule:
          A_max / (n_items × h̄) ≈ 0.65 × n_nodes × d̄
      meaning each item can stock only ~65% of average total demand,
      leaving 35%+ to be supplemented by 3D printing.
================================================================================
"""

from __future__ import annotations
import copy
import csv
import json
import math, time, traceback
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB

# ── Journal-style plot settings (English, Times New Roman) ───────────────────
matplotlib.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman"],
    "font.size":       11,
    "axes.titlesize":  12,
    "axes.labelsize":  11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "savefig.dpi":     300,
    "savefig.bbox":    "tight",
})

MONEY_SCALE  = 1e4     # solver unit = yuan / 10 000
MEAN_DEMAND  = 50.0    # mean demand per (item, node) per scenario
AVG_H        = 1.0     # approximate average h_i [m³/unit], used for A_max sizing
DEFAULT_MULTI_SEEDS = [42, 43, 44, 45, 46]

KeyIJ  = Tuple[str, str]
KeyKJ  = Tuple[str, str]
KeyNIJ = Tuple[int, str, str]


def _s(x: float) -> float:
    return float(x) / MONEY_SCALE


def _intify(value: float, name: str, tol: float = 1e-5) -> int:
    nearest = int(round(float(value)))
    if abs(float(value) - nearest) > tol:
        raise RuntimeError(f"{name}={value:.8f} violates integrality tolerance {tol}.")
    return nearest


# ══════════════════════════════════════════════════════════════════════════════
# §1  Instance generator  (calibrated)
# ══════════════════════════════════════════════════════════════════════════════

class ProblemData:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def generate_instance(
        n_items: int, n_nodes: int, n_printers: int, n_samples: int,
        seed: int   = 42,
        # sensitivity knobs
        T_val:      float = 72.0,
        delta:      float = 0.4,
        rho_scale:  float = 1.0,
        B_scale:    float = 1.0,
        Amax_scale: float = 1.0,
) -> ProblemData:
    """
    Generate a synthetic instance with CALIBRATED parameters so that:
      · Traditional inventory can satisfy only ~65 % of average demand
        (space constraint A_max is the binding factor).
      · Purchasing 3D printers is cost-effective to fill the remaining gap.
      · The hierarchy  c_ij < c_kj + ρ_k << M_ij  holds.

    Calibration derivation (Medium: 5×10×3, N=50)
    -----------------------------------------------
    Average demand per item across all nodes = n_nodes × MEAN_DEMAND = 500 units
    Target x_i                               = 0.65 × 500 = 325 units / item
    Space for inventory                      ≈ n_items × 325 × avg(h_i=1) = 1625 m³
    Space for printers                       = n_printers × H_k × 1.5 = 22.5 m³
    → A_max ≈ 1648 m³   (was 100 000 m³ in v2!)

    Worst-case demand per item (δ=0.4) ≈ 1.4×500 = 700 units
    Shortfall = 700 – 325 = 375 units → 3D printing must fill this gap.

    Printer speed v_k ∈ [2,5] units/h, T=72 h → capacity = v_k × T × x_k
    For one printer at 3.5 u/h: 3.5×72 = 252 units → need ~2 printers.

    f_k = 100K–300K yuan → 10–30 scaled units.
    Traditional cost: 325 × _s(1250) = 40.6 scaled per item.
    Printer roughly pays for itself by avoiding M_ij × shortfall penalty.
    """
    rng = np.random.default_rng(seed)
    I = [f"Item_{i+1}"  for i in range(n_items)]
    J = [f"Node_{j+1}"  for j in range(n_nodes)]
    K = [f"Print_{k+1}" for k in range(n_printers)]
    N = list(range(1, n_samples + 1))

    k_to_i, valid_ik = {}, []
    for k in K:
        ai = rng.choice(I)
        k_to_i[k] = ai
        valid_ik.append((ai, k))

    # ── Physical parameters ──────────────────────────────────────────────────
    h_i = {i: rng.uniform(0.5, 1.5) for i in I}   # m³ per unit
    H_k = {k: 5.0                   for k in K}    # m³ per printer
    rho_vol = {k: 1000.0            for k in K}    # consumable density kg/m³

    # ── FIX B-1: tight A_max (only 65% of average total demand can be stocked) ──
    A_max_base = (
        n_items * n_nodes * MEAN_DEMAND * AVG_H * 0.65   # inventory space
        + n_printers * 5.0 * 1.5                          # printer space (+50% margin)
    )
    A_max = A_max_base * Amax_scale

    # ── Cost parameters ──────────────────────────────────────────────────────
    c_i = {i: _s(rng.uniform(500, 2000)) for i in I}   # 500–2000 yuan / unit
    R_i = {i: 0.0                        for i in I}

    # FIX B-2: realistic printer cost 100K–300K yuan (was 1.5–2.5 M yuan)
    f_k = {k: _s(rng.uniform(1e5, 3e5)) for k in K}    # 100K–300K yuan / printer
    c_k = {k: _s(rng.uniform(50, 200))  for k in K}    # 50–200 yuan / kg consumable

    v_k     = {k: rng.uniform(2.0, 5.0) for k in K}    # units / hour
    alpha_k = {k: 0.1                   for k in K}    # wastage rate
    sigma_k = {k: 0.05                  for k in K}    # material per unit (kg)

    # FIX B-3: meaningful waiting cost 200–500 yuan/unit (was 50 yuan/unit)
    rho_wait = {k: _s(rng.uniform(200, 500)) * rho_scale for k in K}

    T = T_val

    # ── FIX B-4: realistic budget B ─────────────────────────────────────────
    # Budget covers ~65% of full inventory + ~1.5 printers on average.
    # This forces genuine trade-offs between buying more inventory vs. printers.
    max_xi_per_item = A_max_base / (n_items * AVG_H)   # space-limited x_i ceiling
    avg_ci  = _s(1250)    # midpoint of c_i range
    avg_fk  = _s(2e5)     # midpoint of f_k range
    avg_ck  = _s(125)     # midpoint of c_k range
    avg_mk  = 200.0       # target consumable per printer (kg)

    B_base  = (
        n_items    * max_xi_per_item * avg_ci     # full trad inventory at constraint
        + n_printers * avg_fk                     # all printers
        + n_printers * avg_mk * avg_ck            # some consumables
    ) * 0.75    # 75% of "buy everything" → genuine scarcity
    B = B_base * B_scale

    # ── Transport costs ──────────────────────────────────────────────────────
    trans = 0.5   # yuan per km per unit
    cJ = {j: (rng.uniform(0,100), rng.uniform(0,100)) for j in J}
    cK = {k: (rng.uniform(0,100), rng.uniform(0,100)) for k in K}
    cI = {i: (rng.uniform(0,100), rng.uniform(0,100)) for i in I}

    c_ij: Dict[KeyIJ, float] = {
        (i,j): _s(math.hypot(cI[i][0]-cJ[j][0], cI[i][1]-cJ[j][1]) * trans)
        for i in I for j in J}
    c_kj: Dict[KeyKJ, float] = {
        (k,j): _s(math.hypot(cK[k][0]-cJ[j][0], cK[k][1]-cJ[j][1]) * trans)
        for k in K for j in J}

    # ── Demand samples ───────────────────────────────────────────────────────
    d_hat: Dict[KeyNIJ, float] = {
        (n,i,j): float(rng.lognormal(math.log(MEAN_DEMAND), 0.4))
        for n in N for i in I for j in J}

    d_low: Dict[KeyIJ, float] = {}
    d_up:  Dict[KeyIJ, float] = {}
    for i in I:
        for j in J:
            vals = [d_hat[(n,i,j)] for n in N]
            avg  = sum(vals) / len(vals)
            d_low[(i,j)] = max(0.0, (1.0 - delta) * avg)
            d_up [(i,j)] =          (1.0 + delta)  * avg

    # ── Big-M: shortage penalty — must satisfy c_ij < c_kj+ρ_k << M_ij ─────
    max_transport = max(max(c_ij.values()), max(c_kj.values()))
    max_wait      = max(rho_wait.values())
    # Set M_ij = 50 × (max transport + max waiting), ensuring it's truly "last resort"
    M_val = 50.0 * (max_transport + max_wait)
    M_ij  = {(i,j): M_val for i in I for j in J}

    return ProblemData(
        I=I, J=J, K=K, N=N,
        k_to_i=k_to_i, valid_ik=valid_ik,
        A_max=A_max, B=B, T=T, delta=delta,
        c_i=c_i, h_i=h_i, R_i=R_i,
        f_k=f_k, c_k=c_k, H_k=H_k,
        rho_vol=rho_vol, v_k=v_k, alpha_k=alpha_k,
        sigma_k=sigma_k, rho_wait=rho_wait,
        c_ij=c_ij, c_kj=c_kj, M_ij=M_ij,
        d_hat=d_hat, d_low=d_low, d_up=d_up,
        money_scale=MONEY_SCALE,
    )


# ══════════════════════════════════════════════════════════════════════════════
# §2  Subproblem (all Bug Fixes retained from v2)
# ══════════════════════════════════════════════════════════════════════════════

def _clone_problem_data(data: ProblemData, **overrides) -> ProblemData:
    payload = copy.deepcopy(data.__dict__)
    payload.update(overrides)
    return ProblemData(**payload)


def _money_to_M(data: ProblemData, value_scaled: float) -> float:
    return float(value_scaled) * float(getattr(data, "money_scale", MONEY_SCALE)) / 1e6


def _sample_mean_demands(data: ProblemData) -> Dict[KeyIJ, float]:
    return {
        (i, j): float(np.mean([data.d_hat[(n, i, j)] for n in data.N]))
        for i in data.I for j in data.J
    }


def _scenario_demand(data: ProblemData, sample_id: int) -> Dict[KeyIJ, float]:
    return {(i, j): float(data.d_hat[(sample_id, i, j)]) for i in data.I for j in data.J}


def _extract_policy(policy_or_result: Dict) -> Dict:
    if "xi_detail" in policy_or_result:
        return {
            "xv": {i: int(round(v)) for i, v in policy_or_result["xi_detail"].items()},
            "xkv": {k: int(round(v)) for k, v in policy_or_result["xk_detail"].items()},
            "mkv": {k: float(v) for k, v in policy_or_result["mk_detail"].items()},
            "fs_M": float(policy_or_result.get("fs_M", 0.0)),
        }
    if all(k in policy_or_result for k in ("xv", "xkv", "mkv")):
        return {
            "xv": {i: int(round(v)) for i, v in policy_or_result["xv"].items()},
            "xkv": {k: int(round(v)) for k, v in policy_or_result["xkv"].items()},
            "mkv": {k: float(v) for k, v in policy_or_result["mkv"].items()},
            "fs_M": float(policy_or_result.get("fs_M", 0.0)),
        }
    raise ValueError("Unsupported policy/result object.")


def load_instance_from_json(path: str) -> ProblemData:
    """
    Load a complete problem instance from JSON.

    Monetary fields in the JSON are expected in raw yuan and converted
    internally to scaled solver units via money_scale.
    """
    with open(path, "r", encoding="utf-8-sig") as fh:
        raw = json.load(fh)

    scalars = raw["scalar_params"]
    sets = raw["sets"]
    money_scale = float(scalars.get("money_scale", MONEY_SCALE))

    def scale_money(v: float) -> float:
        return float(v) / money_scale

    I = list(sets["items"])
    J = list(sets["nodes"])
    K = list(sets["printers"])
    N = [int(v) for v in sets["samples"]]

    item_params = {row["item"]: row for row in raw["item_params"]}
    printer_params = {row["printer"]: row for row in raw["printer_params"]}

    k_to_i = {k: printer_params[k]["item"] for k in K}
    valid_ik = [(k_to_i[k], k) for k in K]

    c_i = {i: scale_money(item_params[i]["c_i_yuan"]) for i in I}
    h_i = {i: float(item_params[i]["h_i"]) for i in I}
    R_i = {i: float(item_params[i].get("R_i", 0.0)) for i in I}

    f_k = {k: scale_money(printer_params[k]["f_k_yuan"]) for k in K}
    c_k = {k: scale_money(printer_params[k]["c_k_yuan_per_kg"]) for k in K}
    H_k = {k: float(printer_params[k]["H_k"]) for k in K}
    rho_vol = {k: float(printer_params[k]["rho_vol"]) for k in K}
    v_k = {k: float(printer_params[k]["v_k"]) for k in K}
    alpha_k = {k: float(printer_params[k]["alpha_k"]) for k in K}
    sigma_k = {k: float(printer_params[k]["sigma_k"]) for k in K}
    rho_wait = {k: scale_money(printer_params[k]["rho_wait_yuan_per_unit"]) for k in K}

    c_ij = {}
    M_ij = {}
    for row in raw["transport_item_to_node"]:
        key = (row["item"], row["node"])
        c_ij[key] = scale_money(row["c_ij_yuan"])
        M_ij[key] = scale_money(row["M_ij_yuan"])

    c_kj = {}
    for row in raw["transport_printer_to_node"]:
        c_kj[(row["printer"], row["node"])] = scale_money(row["c_kj_yuan"])

    d_hat = {}
    for row in raw["demand_samples"]:
        d_hat[(int(row["sample"]), row["item"], row["node"])] = float(row["demand"])

    delta = float(scalars.get("delta", 0.4))
    d_low = {}
    d_up = {}
    for i in I:
        for j in J:
            vals = [d_hat[(n, i, j)] for n in N]
            avg = sum(vals) / len(vals)
            d_low[(i, j)] = max(0.0, (1.0 - delta) * avg)
            d_up[(i, j)] = (1.0 + delta) * avg

    return ProblemData(
        I=I, J=J, K=K, N=N,
        k_to_i=k_to_i, valid_ik=valid_ik,
        A_max=float(scalars["A_max"]),
        B=scale_money(scalars["B_yuan"]),
        T=float(scalars["T"]),
        delta=delta,
        c_i=c_i, h_i=h_i, R_i=R_i,
        f_k=f_k, c_k=c_k, H_k=H_k,
        rho_vol=rho_vol, v_k=v_k, alpha_k=alpha_k,
        sigma_k=sigma_k, rho_wait=rho_wait,
        c_ij=c_ij, c_kj=c_kj, M_ij=M_ij,
        d_hat=d_hat, d_low=d_low, d_up=d_up,
        money_scale=money_scale,
        data_path=path,
    )


def _base_instance(data_path: Optional[str] = None,
                   n_items: int = 5, n_nodes: int = 10, n_printers: int = 3,
                   n_samples: int = 50, seed: int = 42) -> ProblemData:
    if data_path:
        return load_instance_from_json(data_path)
    return generate_instance(n_items, n_nodes, n_printers, n_samples, seed=seed)


def resample_demands_from_base(base_data: ProblemData, n_samples: int, seed: int,
                               demand_kappa: float = 1.0,
                               log_sigma: float = 0.4,
                               delta: Optional[float] = None) -> ProblemData:
    """
    Keep the physical network unchanged and resample only demand scenarios.
    """
    rng = np.random.default_rng(seed)
    means = _sample_mean_demands(base_data)
    delta_use = float(getattr(base_data, "delta", 0.4) if delta is None else delta)
    N = list(range(1, n_samples + 1))

    d_hat = {}
    for n in N:
        for i in base_data.I:
            for j in base_data.J:
                target_mean = max(1e-6, demand_kappa * means[(i, j)])
                mu_log = math.log(target_mean) - 0.5 * (log_sigma ** 2)
                d_hat[(n, i, j)] = float(rng.lognormal(mu_log, log_sigma))

    d_low = {}
    d_up = {}
    for i in base_data.I:
        for j in base_data.J:
            vals = [d_hat[(n, i, j)] for n in N]
            avg = sum(vals) / len(vals)
            d_low[(i, j)] = max(0.0, (1.0 - delta_use) * avg)
            d_up[(i, j)] = (1.0 + delta_use) * avg

    return _clone_problem_data(base_data, N=N, d_hat=d_hat, d_low=d_low, d_up=d_up, delta=delta_use)


def build_deterministic_instance_from_training(data_train: ProblemData) -> ProblemData:
    means = _sample_mean_demands(data_train)
    N = [1]
    d_hat = {(1, i, j): means[(i, j)] for i in data_train.I for j in data_train.J}
    d_low = {(i, j): means[(i, j)] for i in data_train.I for j in data_train.J}
    d_up = {(i, j): means[(i, j)] for i in data_train.I for j in data_train.J}
    return _clone_problem_data(data_train, N=N, d_hat=d_hat, d_low=d_low, d_up=d_up, delta=0.0)


def _solve_sp(data, n, xv, xkv, mkv, lamv, tl=None):
    I, J, K = data.I, data.J, data.K
    sp = gp.Model(f"SP_{n}"); sp.Params.OutputFlag = 0
    if tl: sp.Params.TimeLimit = tl

    pi = sp.addVars(I, J, lb=0.0,
                    ub={(i,j): data.M_ij[i,j] for i in I for j in J}, name="pi")
    mu = sp.addVars(I, lb=0.0, name="mu")
    nu = sp.addVars(K, lb=0.0, name="nu")
    om = sp.addVars(K, lb=0.0, name="om")
    z  = sp.addVars(I, J, vtype=GRB.BINARY, name="z")
    u  = sp.addVars(I, J, lb=0.0, name="u")

    for i in I:
        for j in J:
            sp.addConstr(pi[i,j] - mu[i] <= data.c_ij[i,j])
    for (i,k) in data.valid_ik:
        for j in J:
            sp.addConstr(pi[i,j]
                         - data.sigma_k[k]*(1+data.alpha_k[k])*nu[k]
                         - (1/data.v_k[k])*om[k]
                         <= data.c_kj[k,j] + data.rho_wait[k])
    for i in I:
        for j in J:
            M = data.M_ij[i,j]
            sp.addConstr(u[i,j] <= M*z[i,j])
            sp.addConstr(u[i,j] <= pi[i,j])
            sp.addConstr(u[i,j] >= pi[i,j] - M*(1-z[i,j]))

    obj_dp = gp.quicksum(
        data.d_low[i,j]*pi[i,j] + (data.d_up[i,j]-data.d_low[i,j])*u[i,j]
        for i in I for j in J)
    obj_res = (gp.quicksum(xv[i]*mu[i] for i in I)
              +gp.quicksum(mkv[k]*nu[k] for k in K)
              +gp.quicksum(data.T*xkv[k]*om[k] for k in K))
    obj_dist = gp.quicksum(
        float(abs(data.d_low[i,j]-data.d_hat[n,i,j]))
        + float(abs(data.d_up[i,j]-data.d_hat[n,i,j])
               -abs(data.d_low[i,j]-data.d_hat[n,i,j]))*z[i,j]
        for i in I for j in J)

    sp.setObjective(obj_dp - obj_res - lamv*obj_dist, GRB.MAXIMIZE)
    sp.optimize()

    bo = sp.ObjVal
    bz  = {(i,j): z[i,j].X  for i in I for j in J}
    piv = {(i,j): pi[i,j].X for i in I for j in J}
    muv = {i: mu[i].X for i in I}
    nuv = {k: nu[k].X for k in K}
    omv = {k: om[k].X for k in K}

    res_val = (sum(xv[i]*muv[i] for i in I)
              +sum(mkv[k]*nuv[k] for k in K)
              +sum(data.T*xkv[k]*omv[k] for k in K))

    net_hat = sum(data.d_hat[n,i,j]*piv[i,j] for i in I for j in J) - res_val
    if net_hat > bo + 1e-5:
        actual = net_hat
        ud = {(i,j): data.d_hat[n,i,j] for i in I for j in J}
        udist = 0.0
    else:
        actual, ud, udist = bo, {}, 0.0
        for i in I:
            for j in J:
                td = data.d_up[i,j] if bz[i,j]>0.5 else data.d_low[i,j]
                ud[(i,j)] = td
                udist += float(abs(td-data.d_hat[n,i,j]))

    return {
        "sp_obj":      actual,
        "obj_res":     res_val,
        "const":       sum(ud[i,j]*piv[i,j] for i in I for j in J),
        "coef_x":      {i: -muv[i]           for i in I},
        "coef_mk":     {k: -nuv[k]           for k in K},
        "coef_xk":     {k: -(data.T*omv[k])  for k in K},
        "coef_lambda": -udist,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §3  CCG master solver
# ══════════════════════════════════════════════════════════════════════════════

def solve_ccg(data, zeta, max_iters=300, tol=1e-3,
              tl_master=None, tl_sp=None, verbose=False,
              disable_3d: bool = False,
              item_ub_map: Optional[Dict[str, float]] = None,
              printer_lb_map: Optional[Dict[str, float]] = None,
              printer_ub_map: Optional[Dict[str, float]] = None) -> Dict:
    t0 = time.time()
    m  = gp.Model("Master"); m.Params.OutputFlag = 0
    m.Params.NumericFocus=3; m.Params.FeasibilityTol=1e-9
    m.Params.OptimalityTol=1e-9; m.Params.Method=1
    if tl_master: m.Params.TimeLimit = tl_master

    I, K, N = data.I, data.K, data.N
    x   = m.addVars(I, vtype=GRB.INTEGER,    lb=0.0, name="x")
    xk  = m.addVars(K, vtype=GRB.INTEGER,    lb=0.0, name="xk")
    mk  = m.addVars(K, vtype=GRB.CONTINUOUS, lb=0.0, name="mk")
    lam = m.addVar(lb=0.0, name="lam")
    eta = m.addVars(N, lb=0.0, name="eta")

    for i in I:
        ub_i = data.A_max / data.h_i[i]
        if item_ub_map and i in item_ub_map:
            ub_i = min(ub_i, float(item_ub_map[i]))
        m.addConstr(x[i] <= ub_i)
    for k in K:
        ub_k = data.A_max / data.H_k[k]
        if printer_ub_map and k in printer_ub_map:
            ub_k = min(ub_k, float(printer_ub_map[k]))
        m.addConstr(xk[k] <= ub_k)
    for i in I:
        m.addConstr(x[i] >= data.R_i[i])
    if printer_lb_map:
        for k in K:
            if k in printer_lb_map:
                m.addConstr(xk[k] >= float(printer_lb_map[k]))
    for k in K:
        # Consumables are meaningful only when at least one printer is deployed.
        m.addConstr(mk[k] <= data.rho_vol[k] * data.A_max * xk[k])
    if disable_3d:
        for k in K:
            m.addConstr(xk[k] == 0)
            m.addConstr(mk[k] == 0)
    m.addConstr(gp.quicksum(data.h_i[i]*x[i] for i in I)
               +gp.quicksum(data.H_k[k]*xk[k]+mk[k]/data.rho_vol[k] for k in K)
               <= data.A_max)
    m.addConstr(gp.quicksum(data.c_i[i]*x[i] for i in I)
               +gp.quicksum(data.f_k[k]*xk[k]+data.c_k[k]*mk[k] for k in K)
               <= data.B)

    fs = (gp.quicksum(data.c_i[i]*x[i] for i in I)
         +gp.quicksum(data.f_k[k]*xk[k]+data.c_k[k]*mk[k] for k in K))
    m.setObjective(fs + zeta*lam + (1/len(N))*gp.quicksum(eta[n] for n in N),
                   GRB.MINIMIZE)

    LB=-1e20; UB=1e20; history={"iter":[],"lb":[],"ub":[]}
    xv={}; xkv={}; mkv={}; val_fs=0.0; gap=1.0

    for it in range(1, max_iters+1):
        m.optimize()
        if m.Status==GRB.INFEASIBLE: raise RuntimeError("Master INFEASIBLE")
        LB = max(LB, m.ObjVal)
        xv   = {i: _intify(x[i].X,  f"x[{i}]")  for i in I}
        xkv  = {k: _intify(xk[k].X, f"xk[{k}]") for k in K}
        mkv  = {k: mk[k].X for k in K}
        lamv = lam.X
        val_fs = (sum(data.c_i[i]*xv[i] for i in I)
                 +sum(data.f_k[k]*xkv[k]+data.c_k[k]*mkv[k] for k in K))

        viol=0; sf=0.0
        for n in N:
            cut  = _solve_sp(data, n, xv, xkv, mkv, lamv, tl_sp)
            etav = eta[n].X
            sf  += cut["sp_obj"] + cut["obj_res"]
            if cut["sp_obj"] > etav + tol:
                viol += 1
                expr = (cut["const"]
                       +gp.quicksum(cut["coef_x"][i]*x[i]   for i in I)
                       +gp.quicksum(cut["coef_mk"][k]*mk[k]  for k in K)
                       +gp.quicksum(cut["coef_xk"][k]*xk[k] for k in K)
                       +cut["coef_lambda"]*lam)
                m.addConstr(eta[n] >= expr)

        cur = val_fs + zeta*lamv + sf/len(N)
        if cur < LB-1e-5: cur = LB
        UB = min(UB, cur)
        history["iter"].append(it); history["lb"].append(LB); history["ub"].append(UB)
        gap = (UB-LB)/abs(UB) if abs(UB)>1e-9 else 0.0
        if verbose:
            print(f"    it={it:3d}  LB={LB:.4f}  UB={UB:.4f}"
                  f"  gap={gap:.2%}  viol={viol}")
        if gap < 0.01 or viol == 0: break

    S = float(getattr(data, "money_scale", MONEY_SCALE)) / 1e6
    return {
        "obj_M":     LB*S,
        "fs_M":      val_fs*S,
        "sum_xi":    int(sum(xv.values()))               if xv  else 0,
        "avg_xi":    float(np.mean(list(xv.values())))  if xv  else 0.0,
        "avg_xk":    float(np.mean(list(xkv.values()))) if xkv else 0.0,
        "sum_xk":    int(sum(xkv.values()))             if xkv else 0,
        "avg_mk":    float(np.mean(list(mkv.values()))) if mkv else 0.0,
        "xi_detail": dict(xv),
        "xk_detail": dict(xkv),
        "mk_detail": dict(mkv),
        "disable_3d": disable_3d,
        "iters":     it,
        "wall_time": time.time()-t0,
        "gap":       gap,
        "history":   history,
    }


# ══════════════════════════════════════════════════════════════════════════════
# §4  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def eval_second_stage_detail(data: ProblemData, demand: Dict[KeyIJ, float],
                             xv: Dict[str, int], xkv: Dict[str, int], mkv: Dict[str, float],
                             tl: Optional[float] = None) -> Dict:
    """
    Evaluate a fixed first-stage policy under one realized demand scenario.
    """
    I, J, K = data.I, data.J, data.K
    item_to_printers = {
        i: [k for (ii, k) in data.valid_ik if ii == i]
        for i in I
    }

    sp = gp.Model("EvalSecondStage")
    sp.Params.OutputFlag = 0
    if tl:
        sp.Params.TimeLimit = tl

    y_inv = sp.addVars(I, J, lb=0.0, name="y_inv")
    y_prn = sp.addVars(K, J, lb=0.0, name="y_prn")
    short = sp.addVars(I, J, lb=0.0, name="short")

    for i in I:
        sp.addConstr(gp.quicksum(y_inv[i, j] for j in J) <= float(xv.get(i, 0)))
    for k in K:
        sp.addConstr(
            gp.quicksum(data.sigma_k[k] * (1.0 + data.alpha_k[k]) * y_prn[k, j] for j in J)
            <= float(mkv.get(k, 0.0))
        )
        sp.addConstr(
            gp.quicksum((1.0 / data.v_k[k]) * y_prn[k, j] for j in J)
            <= data.T * float(xkv.get(k, 0))
        )
    for i in I:
        printers_i = item_to_printers[i]
        for j in J:
            sp.addConstr(
                y_inv[i, j]
                + gp.quicksum(y_prn[k, j] for k in printers_i)
                + short[i, j]
                >= float(demand[(i, j)])
            )

    inv_cost = gp.quicksum(data.c_ij[i, j] * y_inv[i, j] for i in I for j in J)
    print_trans_cost = gp.quicksum(data.c_kj[k, j] * y_prn[k, j] for k in K for j in J)
    wait_cost = gp.quicksum(data.rho_wait[k] * y_prn[k, j] for k in K for j in J)
    shortage_cost = gp.quicksum(data.M_ij[i, j] * short[i, j] for i in I for j in J)
    sp.setObjective(inv_cost + print_trans_cost + wait_cost + shortage_cost, GRB.MINIMIZE)
    sp.optimize()

    if sp.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Second-stage evaluation failed with status {sp.Status}.")

    inv_cost_v = sum(data.c_ij[i, j] * y_inv[i, j].X for i in I for j in J)
    print_trans_v = sum(data.c_kj[k, j] * y_prn[k, j].X for k in K for j in J)
    wait_cost_v = sum(data.rho_wait[k] * y_prn[k, j].X for k in K for j in J)
    shortage_cost_v = sum(data.M_ij[i, j] * short[i, j].X for i in I for j in J)
    shortage_units = sum(short[i, j].X for i in I for j in J)
    total_demand = sum(float(demand[(i, j)]) for i in I for j in J)
    print_units = sum(y_prn[k, j].X for k in K for j in J)
    delivered_units = total_demand - shortage_units
    total_recourse = inv_cost_v + print_trans_v + wait_cost_v + shortage_cost_v
    demand_by_item = {i: float(sum(demand[(i, j)] for j in J)) for i in I}
    shortage_by_item = {i: float(sum(short[i, j].X for j in J)) for i in I}
    delivered_by_item = {i: float(demand_by_item[i] - shortage_by_item[i]) for i in I}
    print_by_item = {
        i: float(sum(y_prn[k, j].X for k in item_to_printers[i] for j in J))
        for i in I
    }

    return {
        "inv_cost_M": _money_to_M(data, inv_cost_v),
        "print_trans_cost_M": _money_to_M(data, print_trans_v),
        "wait_cost_M": _money_to_M(data, wait_cost_v),
        "shortage_cost_M": _money_to_M(data, shortage_cost_v),
        "recourse_M": _money_to_M(data, total_recourse),
        "shortage_units": float(shortage_units),
        "total_demand": float(total_demand),
        "print_units": float(print_units),
        "delivered_units": float(delivered_units),
        "shortage_rate": float(shortage_units / total_demand) if total_demand > 1e-9 else 0.0,
        "service_level": float(delivered_units / total_demand) if total_demand > 1e-9 else 1.0,
        "wait_share": float(wait_cost_v / total_recourse) if total_recourse > 1e-9 else 0.0,
        "demand_by_item": demand_by_item,
        "shortage_by_item": shortage_by_item,
        "delivered_by_item": delivered_by_item,
        "print_by_item": print_by_item,
    }


def eval_second_stage(data: ProblemData, demand: Dict[KeyIJ, float],
                      xv: Dict[str, int], xkv: Dict[str, int], mkv: Dict[str, float],
                      tl: Optional[float] = None) -> float:
    return eval_second_stage_detail(data, demand, xv, xkv, mkv, tl=tl)["recourse_M"]


def evaluate_policy_on_dataset(data: ProblemData, policy_or_result: Dict,
                               ref_total_M: Optional[float] = None,
                               tl: Optional[float] = None) -> Dict:
    policy = _extract_policy(policy_or_result)
    totals, recourses, shortage_rates, wait_shares, service_levels = [], [], [], [], []
    shortage_units_all, total_demands_all = [], []

    for n in data.N:
        detail = eval_second_stage_detail(
            data, _scenario_demand(data, n),
            policy["xv"], policy["xkv"], policy["mkv"], tl=tl,
        )
        total_cost_M = policy["fs_M"] + detail["recourse_M"]
        totals.append(total_cost_M)
        recourses.append(detail["recourse_M"])
        shortage_rates.append(detail["shortage_rate"])
        wait_shares.append(detail["wait_share"])
        service_levels.append(detail["service_level"])
        shortage_units_all.append(detail["shortage_units"])
        total_demands_all.append(detail["total_demand"])

    out = {
        "costs_M": totals,
        "recourse_costs_M": recourses,
        "mean_total_M": float(np.mean(totals)),
        "max_total_M": float(np.max(totals)),
        "min_total_M": float(np.min(totals)),
        "p90_total_M": float(np.percentile(totals, 90)),
        "p95_total_M": float(np.percentile(totals, 95)),
        "mean_recourse_M": float(np.mean(recourses)),
        "mean_shortage_rate": float(np.mean(shortage_rates)),
        "aggregate_shortage_rate": float(sum(shortage_units_all) / sum(total_demands_all))
        if sum(total_demands_all) > 1e-9 else 0.0,
        "mean_wait_share": float(np.mean(wait_shares)),
        "mean_service_level": float(np.mean(service_levels)),
        "shortage_units_total": float(sum(shortage_units_all)),
        "demand_total": float(sum(total_demands_all)),
    }
    if ref_total_M is not None:
        dis = [max((c - ref_total_M) / ref_total_M, 0.0) for c in totals]
        out.update({
            "ref_total_M": float(ref_total_M),
            "dis": dis,
            "mean_dis": float(np.mean(dis)),
            "max_dis": float(np.max(dis)),
            "min_dis": float(np.min(dis)),
            "p90_dis": float(np.percentile(dis, 90)),
            "p95_dis": float(np.percentile(dis, 95)),
        })
    return out


def evaluate_policy_on_dataset_by_item(data: ProblemData, policy_or_result: Dict,
                                       tl: Optional[float] = None) -> Dict[str, Dict[str, float]]:
    policy = _extract_policy(policy_or_result)
    by_item = {
        i: {
            "demand_total": 0.0,
            "shortage_units_total": 0.0,
            "delivered_units_total": 0.0,
            "print_units_total": 0.0,
        }
        for i in data.I
    }

    for n in data.N:
        detail = eval_second_stage_detail(
            data, _scenario_demand(data, n),
            policy["xv"], policy["xkv"], policy["mkv"], tl=tl,
        )
        for i in data.I:
            by_item[i]["demand_total"] += float(detail["demand_by_item"][i])
            by_item[i]["shortage_units_total"] += float(detail["shortage_by_item"][i])
            by_item[i]["delivered_units_total"] += float(detail["delivered_by_item"][i])
            by_item[i]["print_units_total"] += float(detail["print_by_item"][i])

    for i in data.I:
        demand_total = by_item[i]["demand_total"]
        shortage_total = by_item[i]["shortage_units_total"]
        delivered_total = by_item[i]["delivered_units_total"]
        by_item[i]["aggregate_shortage_rate"] = float(shortage_total / demand_total) if demand_total > 1e-9 else 0.0
        by_item[i]["aggregate_service_level"] = float(delivered_total / demand_total) if demand_total > 1e-9 else 1.0
        by_item[i]["print_share_of_demand"] = float(by_item[i]["print_units_total"] / demand_total) if demand_total > 1e-9 else 0.0

    return by_item


def _run(label, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"  !! [{label}] failed: {e}"); traceback.print_exc(); return None

def _save(fig, fname):
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    print(f"  >> Saved: {fname}"); plt.close(fig)

# Shared line styles (black-and-white printable)
_LS = [
    dict(color="black", ls="-",  marker="o", ms=5, mfc="white", lw=1.8),
    dict(color="black", ls="--", marker="s", ms=5, mfc="black",  lw=1.6),
    dict(color="black", ls=":",  marker="^", ms=5, mfc="white",  lw=1.5),
]

def _plot_series(ax, xs, ydata, sty, xlog=False, yerr=None):
    xs = np.asarray(xs, dtype=float)
    y = np.asarray(ydata, dtype=float)
    ax.plot(xs, y, **sty)
    if yerr is not None:
        err = np.asarray(yerr, dtype=float)
        ax.fill_between(
            xs,
            np.maximum(0.0, y - err),
            y + err,
            color="0.75",
            alpha=0.25,
            linewidth=0,
        )
    if xlog:
        ax.set_xscale("log")


def _series_stats(values):
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_csv_rows(path: str, fieldnames: List[str], rows: List[Dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"  >> Saved: {path}")


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"  >> Saved: {path}")


def _write_latex_table(path: str, headers: List[str], rows: List[List[str]],
                       caption: str, label: str) -> None:
    cols = "l" + "r" * (len(headers) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        " & ".join(headers) + " \\\\",
        "\\hline",
    ]
    lines.extend(" & ".join(row) + " \\\\" for row in rows)
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    _write_text(path, "\n".join(lines))


def _plot_exp5_oos(results: Dict, kappa_list: List[float]) -> None:
    fig, axes = plt.subplots(1, len(kappa_list), figsize=(5.2 * len(kappa_list), 4.2), squeeze=False)
    for ax, kappa in zip(axes[0], kappa_list):
        dro = results[kappa]["DRO"]["costs_M"]
        saa = results[kappa]["SAA"]["costs_M"]
        ax.hist(dro, bins=25, density=True, alpha=0.45, color="0.25", label="DRO", edgecolor="black")
        ax.hist(saa, bins=25, density=True, alpha=0.35, color="0.75", label="SAA", edgecolor="black")
        ax.set_title(rf"$\kappa={kappa}$", fontweight="bold")
        ax.set_xlabel("Out-of-sample total cost (M¥)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend()
    plt.suptitle("Figure 5a  Out-of-Sample Cost Distributions", fontweight="bold", y=1.02)
    _save(fig, "exp5_oos_cost_histograms.png")

    fig2, ax2 = plt.subplots(figsize=(9, 4.8))
    box_data, positions, labels = [], [], []
    pos = 1
    for kappa in kappa_list:
        for model in ("DRO", "SAA"):
            box_data.append(results[kappa][model]["dis"])
            positions.append(pos)
            labels.append(f"{model}\nκ={kappa}")
            pos += 1
        pos += 0.5
    bp = ax2.boxplot(box_data, positions=positions, patch_artist=True, widths=0.65, showfliers=False)
    shades = ["0.35", "0.8"] * len(kappa_list)
    for patch, shade in zip(bp["boxes"], shades):
        patch.set_facecolor(shade)
        patch.set_edgecolor("black")
    for med in bp["medians"]:
        med.set_color("black")
        med.set_linewidth(1.4)
    ax2.set_xticks(positions)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Disappointment value DIS", fontweight="bold")
    ax2.set_title("Figure 5b  Out-of-Sample DIS Comparison", fontweight="bold", pad=10)
    ax2.grid(True, axis="y", ls=":", alpha=0.3)
    _save(fig2, "exp5_oos_dis_boxplot.png")


def _plot_exp6_value_3d(rows: List[Dict], kappa_list: List[float]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    width = 0.34
    xs = np.arange(len(kappa_list))
    for idx, policy in enumerate(("Full 3D-enabled", "No 3D")):
        subset = [r for r in rows if r["policy"] == policy]
        offset = (idx - 0.5) * width
        axes[0].bar(xs + offset, [r["mean_total_M"] for r in subset], width=width,
                    color="0.3" if idx == 0 else "0.8", edgecolor="black", label=policy)
        axes[1].bar(xs + offset, [r["aggregate_shortage_rate"] * 100 for r in subset], width=width,
                    color="0.3" if idx == 0 else "0.8", edgecolor="black", label=policy)

    for ax, title, ylabel in zip(
        axes,
        ["(a) Mean out-of-sample total cost", "(b) Aggregate shortage rate"],
        ["Mean cost (M¥)", "Shortage rate (%)"],
    ):
        ax.set_xticks(xs)
        ax.set_xticklabels([str(k) for k in kappa_list])
        ax.set_xlabel(r"Distribution-shift factor $\kappa$", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(True, axis="y", ls=":", alpha=0.3)
        ax.legend()
    plt.suptitle("Figure 6  Value of 3D Printing Capacity", fontweight="bold", y=1.02)
    _save(fig, "exp6_value_of_3d_printing.png")


def _plot_exp7_benchmark(rows: List[Dict], kappa_list: List[float], model_names: List[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = {
        model_names[0]: _LS[0],
        model_names[1]: _LS[1],
        model_names[2]: _LS[2],
    }
    for model in model_names:
        subset = [r for r in rows if r["model"] == model]
        xs = [r["kappa"] for r in subset]
        axes[0].plot(xs, [100 * r["mean_dis"] for r in subset], **styles[model], label=model)
        axes[1].plot(xs, [100 * r["p95_dis"] for r in subset], **styles[model], label=model)
    for ax, title, ylabel in zip(
        axes,
        ["(a) Average disappointment", "(b) 95th percentile disappointment"],
        ["Mean DIS (%)", "P95 DIS (%)"],
    ):
        ax.set_xlabel(r"Distribution-shift factor $\kappa$", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=10)
        ax.grid(True, ls=":", alpha=0.3)
        ax.legend()
    plt.suptitle("Figure 7  Model Benchmarking under Distribution Shift", fontweight="bold", y=1.02)
    _save(fig, "exp7_model_benchmarking.png")


def _summarize_seed_runs(records, param_name, param_value):
    summary = {param_name: param_value, "runs": list(records), "n_runs": len(records)}
    for metric in ("sum_xi", "avg_xi", "avg_xk", "sum_xk", "avg_mk", "obj_M", "iters"):
        mean, std = _series_stats([r[metric] for r in records])
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = std
    return summary


def _multi_seed_sweep(label, param_name, param_value, seeds, build_data: Callable[[int], ProblemData],
                      zeta, **solve_kw):
    records = []
    for seed in seeds:
        data = build_data(seed)
        res = _run(f"{label}, seed={seed}", solve_ccg, data, zeta, **solve_kw)
        if res:
            records.append({"seed": seed, **res})
            print(f"       seed={seed:>3}  xi_tot={res['sum_xi']}"
                  f"  xk_tot={res['sum_xk']:.0f}  mk={res['avg_mk']:.1f}")
    if not records:
        return None

    summary = _summarize_seed_runs(records, param_name, param_value)
    print(f"     mean+/-std  xi_tot={summary['sum_xi_mean']:.2f}+/-{summary['sum_xi_std']:.2f}"
          f"  xk_tot={summary['sum_xk_mean']:.2f}+/-{summary['sum_xk_std']:.2f}"
          f"  mk={summary['avg_mk_mean']:.1f}+/-{summary['avg_mk_std']:.1f}")
    return summary


def _three_panel(axes_row, xs, xi_list, xk_list, mk_list, xlabel, xlog=False,
                 yerrs=(None, None, None),
                 xi_ylabel=r"Total traditional stock $\sum_i x_i$ (units)",
                 xi_title=r"(i) Total traditional stock $\sum_i x_i$",
                 xk_ylabel=r"Total printer count $\sum_k x_k$ (units)",
                 xk_title=r"(ii) Total printer count $\sum_k x_k$"):
    """Draw total stock / total printer / consumable panels, with optional +/-1 std shading."""
    rows = [
        (xi_list, yerrs[0], xi_ylabel, xi_title),
        (xk_list, yerrs[1], xk_ylabel, xk_title),
        (mk_list, yerrs[2], r"Avg. consumable $\bar{m}_k$ (kg)",
         r"(iii) Consumable stock $\bar{m}_k$"),
    ]
    for ax, (ydata, yerr, ylabel, title), sty in zip(axes_row, rows, _LS):
        _plot_series(ax, xs, ydata, sty, xlog=xlog, yerr=yerr)
        ax.set_xlabel(xlabel, fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=9)
        ax.grid(True, ls=":", alpha=0.35)


# ══════════════════════════════════════════════════════════════════════════════
# §5  Experiment 1 — Scalability benchmark
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 1)
To validate the computational feasibility of the proposed Wasserstein-DRO model
and CCG algorithm across different problem scales, we construct a benchmark test
set comprising three instance sizes (Small / Medium / Large) and three historical
sample sizes (N = 50 / 100 / 200), yielding 9 experimental cases in total.
Parameter details are summarised in Table X.  Network topology is generated
randomly within a 100 × 100 km grid; inter-node transport costs are proportional
to Euclidean distance.  Historical demand follows a log-normal distribution
(mean = 50, log-std = 0.4) to capture the heavy-tailed nature of disaster demand.
Both budget B and capacity A_max scale proportionally with the number of nodes
to ensure feasibility across all scales.  All experiments use Gurobi 13.0 with
Wasserstein radius ζ = 10 and convergence tolerance ε = 1 %.
"""

def exp1_scalability(zeta=10.0):
    print("\n" + "═"*60 + "\nExp 1  Scalability Benchmark\n" + "═"*60)
    scales = [("Small",2,5,2), ("Medium",5,10,3), ("Large",10,20,5)]
    N_list = [50, 100, 200]

    time_mat = np.full((3,3), np.nan)
    iter_mat = np.full((3,3), np.nan)
    conv_res = None

    for si,(sn,ni,nj,nk) in enumerate(scales):
        for ci,nn in enumerate(N_list):
            lbl = f"{sn}(N={nn})"
            print(f"\n  ▶ {lbl}  [{ni}×{nj}×{nk}×{nn}]")
            data = generate_instance(ni,nj,nk,nn,seed=42)
            res  = _run(lbl, solve_ccg, data, zeta,
                        max_iters=300, tol=1e-3, tl_master=1200, tl_sp=120)
            if not res: continue
            time_mat[si,ci] = res["wall_time"]
            iter_mat[si,ci] = res["iters"]
            print(f"     xi_tot={res['sum_xi']}  xk_avg={res['avg_xk']:.2f}"
                  f"  xk_tot={res['sum_xk']:.0f}"
                  f"  mk={res['avg_mk']:.1f}  obj={res['obj_M']:.3f} M¥"
                  f"  t={res['wall_time']:.1f}s  iter={res['iters']}")
            if sn=="Large" and nn==100: conv_res = res

    snames = [s[0] for s in scales]

    # ── Figure 1: heatmaps ────────────────────────────────────────────────
    fig, axes = plt.subplots(1,2, figsize=(13,4.5))
    def _hm(ax, mat, title, cbar_lbl, fmt):
        im = ax.imshow(mat, cmap="Greys", aspect="auto",
                       vmin=0, vmax=np.nanmax(mat)*1.05)
        ax.set_xticks(range(3)); ax.set_xticklabels(N_list)
        ax.set_yticks(range(3)); ax.set_yticklabels(snames)
        ax.set_xlabel("Sample size $N$", fontweight="bold")
        ax.set_ylabel("Instance scale", fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=11)
        plt.colorbar(im, ax=ax, label=cbar_lbl, shrink=0.88)
        vmax = np.nanmax(mat)
        for r in range(3):
            for c in range(3):
                v=mat[r,c]
                if not np.isnan(v):
                    ax.text(c,r,f"{v:{fmt}}",ha="center",va="center",
                            color="white" if v>vmax*0.55 else "black",
                            fontsize=9, fontweight="bold")
    _hm(axes[0], time_mat, "(a) CPU Time (seconds)", "CPU Time (s)", ".1f")
    _hm(axes[1], iter_mat, "(b) CCG Iterations",     "Iterations",   ".0f")
    plt.suptitle("Figure 1  CCG Scalability Benchmark",
                 fontweight="bold", y=1.02)
    _save(fig, "exp1_scalability_heatmap.png")

    # ── Figure 1c: convergence curve (Large, N=100) ───────────────────────
    if conv_res:
        h = conv_res["history"]
        S = MONEY_SCALE/1e6
        lbs = [v*S for v in h["lb"]]; ubs = [v*S for v in h["ub"]]
        fig2, ax = plt.subplots(figsize=(7,4))
        ax.plot(h["iter"],lbs,**_LS[0],label="Lower Bound (LB)")
        ax.plot(h["iter"],ubs,**_LS[1],label="Upper Bound (UB)")
        ax.set_yscale("log")
        ax.set_xlabel("Iteration", fontweight="bold")
        ax.set_ylabel("Objective (M¥, log scale)", fontweight="bold")
        ax.set_title("Figure 1c  Convergence Profile — Large (N=100)",
                     fontweight="bold", pad=11)
        ax.legend(); ax.grid(True,which="both",ls=":",alpha=0.35)
        _save(fig2, "exp1_convergence_large.png")


# ══════════════════════════════════════════════════════════════════════════════
# §6  Experiment 2 — Wasserstein radius ζ sensitivity
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 2)
The Wasserstein radius ζ governs the conservatism of the DRO model.  At ζ = 0
the model reduces to SAA; as ζ grows, the ambiguity set F expands to accommodate
larger distributional shifts from the empirical measure, yielding more
conservative pre-positioning decisions at the cost of higher first-stage
expenditure (robustness premium).  We sweep ζ ∈ {0, 1, 2, 5, 10, 20, 50, 100}
on the Medium instance (|I|=5, |J|=10, |K|=3, N=50) and report changes in
traditional stock x̄_i, printer count x̄_k, consumable stock m̄_k, total
objective, robustness premium, and CCG iterations.
"""

def exp2_zeta_sensitivity():
    print("\n" + "═"*60 + "\nExp 2  Wasserstein Radius ζ Sensitivity\n" + "═"*60)
    zeta_list = [0,1,2,5,10,20,50,100]
    data = generate_instance(5,10,3,50,seed=42)
    records = []
    for z in zeta_list:
        print(f"  ζ={z:>5}")
        res = _run(f"ζ={z}", solve_ccg, data, z,
                   max_iters=300, tol=1e-3, tl_master=600, tl_sp=60)
        if res:
            records.append({"zeta":z, **res})
            print(f"     xi_tot={res['sum_xi']} xk_avg={res['avg_xk']:.2f}"
                  f" xk_tot={res['sum_xk']:.0f}"
                  f" mk={res['avg_mk']:.1f} obj={res['obj_M']:.3f}")
    if not records: return records

    xs  = [r["zeta"]   for r in records]
    xi  = [r["sum_xi"] for r in records]
    xk  = [r["sum_xk"] for r in records]
    mk  = [r["avg_mk"] for r in records]
    obj = [r["obj_M"]  for r in records]
    itr = [r["iters"]  for r in records]
    base= obj[0] if obj[0]!=0 else 1.0
    prm = [(v-obj[0])/base*100 for v in obj]

    fig, axes = plt.subplots(2,3, figsize=(15,8))
    plt.subplots_adjust(hspace=0.42, wspace=0.32)
    _three_panel(axes[0], xs, xi, xk, mk,
                 xlabel=r"Wasserstein radius $\zeta$ (log scale)", xlog=True)

    def _pl(ax, y, title, ylabel):
        ax.plot(xs, y, **_LS[0]); ax.set_xscale("log")
        ax.set_xlabel(r"Wasserstein radius $\zeta$ (log scale)", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(title, fontweight="bold", pad=9); ax.grid(True,ls=":",alpha=0.35)

    _pl(axes[1,0], obj, "(iv) Total objective value",     "Objective (M¥)")
    _pl(axes[1,1], prm, "(v)  Robustness premium vs. SAA","Premium (%)")
    axes[1,1].axhline(0,color="gray",ls=":",lw=1)
    _pl(axes[1,2], itr, "(vi) CCG iterations",            "Iterations")

    for ax in axes.flat:
        ax.text(0.03,0.95,"SAA\n(ζ=0)",transform=ax.transAxes,
                fontsize=7,color="gray",va="top")

    plt.suptitle(r"Figure 2  Sensitivity to Wasserstein Radius $\zeta$",
                 fontweight="bold", y=1.01)
    _save(fig, "exp2_zeta_sensitivity.png")
    return records


# ══════════════════════════════════════════════════════════════════════════════
# §7  Experiment 3 — Budget B × Capacity A_max
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 3)
Government emergency budget B and central depot capacity A_max are the two most
policy-relevant resource constraints.  We perform a 5×5 factorial experiment
over B_scale, A_max_scale ∈ {60%, 80%, 100%, 120%, 150%} on the Medium instance
(ζ = 10).  Each cell reports the optimal values of x̄_i, x̄_k, and m̄_k,
revealing the substitution dynamics between traditional stockpiling and 3D-print
capacity as resource constraints tighten or relax.
"""

def exp3_budget_capacity(zeta=10.0):
    print("\n" + "═"*60 + "\nExp 3  Budget B × Capacity A_max\n" + "═"*60)
    B_sc  = [0.6,0.8,1.0,1.2,1.5]
    A_sc  = [0.6,0.8,1.0,1.2,1.5]
    xi_g  = np.full((5,5),np.nan)
    xk_g  = np.full((5,5),np.nan)
    mk_g  = np.full((5,5),np.nan)

    for bi,bs in enumerate(B_sc):
        for ai,as_ in enumerate(A_sc):
            lbl = f"B×{bs:.1f}_A×{as_:.1f}"
            print(f"  ▶ {lbl}")
            data = generate_instance(5,10,3,50,seed=42,B_scale=bs,Amax_scale=as_)
            res  = _run(lbl, solve_ccg, data, zeta,
                        max_iters=200, tol=1e-3, tl_master=600, tl_sp=60)
            if res:
                xi_g[bi,ai]=res["sum_xi"]; xk_g[bi,ai]=res["sum_xk"]
                mk_g[bi,ai]=res["avg_mk"]
                print(f"     xi_tot={res['sum_xi']} xk_avg={res['avg_xk']:.2f}"
                      f" xk_tot={res['sum_xk']:.0f}"
                      f" mk={res['avg_mk']:.1f}")

    Bl=[f"{int(s*100)}%" for s in B_sc]; Al=[f"{int(s*100)}%" for s in A_sc]
    fig, axes = plt.subplots(1,3, figsize=(16,5)); plt.subplots_adjust(wspace=0.32)
    grids  = [xi_g, xk_g, mk_g]
    titles = [
        r"(a) Total traditional stock $\sum_i x_i$ (units)",
        r"(b) Total printer count $\sum_k x_k$ (units)",
        r"(c) Consumable stock $\bar{m}_k$ (kg)",
    ]
    for ax,grid,title in zip(axes,grids,titles):
        im=ax.imshow(grid,cmap="Greys",aspect="auto")
        ax.set_xticks(range(5)); ax.set_xticklabels(Al,fontsize=9)
        ax.set_yticks(range(5)); ax.set_yticklabels(Bl,fontsize=9)
        ax.set_xlabel(r"Capacity $A_{\max}$ (relative to base)",fontweight="bold")
        ax.set_ylabel("Budget $B$ (relative to base)",           fontweight="bold")
        ax.set_title(title,fontweight="bold",pad=10)
        plt.colorbar(im,ax=ax,shrink=0.82)
        vmax=np.nanmax(grid) if not np.all(np.isnan(grid)) else 1.0
        for bi2 in range(5):
            for ai2 in range(5):
                v=grid[bi2,ai2]
                if not np.isnan(v):
                    ax.text(ai2,bi2,f"{v:.1f}",ha="center",va="center",
                            fontsize=8,fontweight="bold",
                            color="white" if v>vmax*0.6 else "black")

    plt.suptitle(r"Figure 3  Sensitivity to Budget $B$ and Capacity $A_{\max}$",
                 fontweight="bold",y=1.02)
    _save(fig,"exp3_budget_capacity.png")
    return xi_g,xk_g,mk_g


# ══════════════════════════════════════════════════════════════════════════════
# §8  Experiment 4a — Rescue time window T
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 4a)
The golden rescue time window T limits the total printing output per printer:
Σ_j (1/v_k) y_{k(i)j} ≤ T·x_k.  A shorter T reduces the effective 3D-print
capacity, shifting the optimal mix towards higher traditional pre-positioning.
We vary T ∈ {24, 36, 48, 72, 96, 120} hours and record the resulting changes
in x̄_i, x̄_k, and m̄_k.
"""

def exp4a_T_sensitivity(zeta=10.0, seeds: Optional[List[int]] = None):
    print("\n" + "═"*60 + "\nExp 4a  Rescue Time Window T\n" + "═"*60)
    T_list = [24,36,48,72,96,120]
    seeds = DEFAULT_MULTI_SEEDS if seeds is None else list(seeds)
    recs = []
    for T_val in T_list:
        lbl = f"T={T_val}h"
        print(f"  ▶ {lbl}")
        summary = _multi_seed_sweep(
            lbl, "T", T_val, seeds,
            lambda seed, T_val=T_val: generate_instance(
                5, 10, 3, 50, seed=seed, T_val=float(T_val)),
            zeta, max_iters=200, tol=1e-3, tl_master=600, tl_sp=60,
        )
        if summary:
            recs.append(summary)
    if not recs:
        return recs

    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    plt.subplots_adjust(wspace=0.32, bottom=0.18)
    _three_panel(axes,[r["T"] for r in recs],
                 [r["sum_xi_mean"] for r in recs],
                 [r["sum_xk_mean"] for r in recs],
                 [r["avg_mk_mean"] for r in recs],
                 yerrs=([r["sum_xi_std"] for r in recs],
                        [r["sum_xk_std"] for r in recs],
                        [r["avg_mk_std"] for r in recs]),
                 xlabel="Golden rescue window $T$ (hours)")
    fig.text(0.5, 0.04,
             "Across random seeds; shaded band shows +/- 1 std.",
             ha="center", fontsize=9)
    plt.suptitle("Figure 4a  Sensitivity to Rescue Time Window $T$",
                 fontweight="bold",y=1.02)
    _save(fig,"exp4a_T_sensitivity.png")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# §9  Experiment 4b — Demand uncertainty width δ
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 4b)
The support-set half-width δ controls how far demand can deviate from its
empirical mean: d̲_{ij} = (1−δ)·d̄_{ij},  d̄_{ij} = (1+δ)·d̄_{ij}.
A larger δ raises the worst-case demand, increasing the required stockpile.
We sweep δ ∈ {0.1, 0.2, 0.3, 0.4, 0.5, 0.6} and report x̄_i, x̄_k, m̄_k.
"""

def exp4b_delta_sensitivity(zeta=10.0):
    print("\n" + "═"*60 + "\nExp 4b  Demand Uncertainty Width δ\n" + "═"*60)
    delta_list=[0.1,0.2,0.3,0.4,0.5,0.6]
    recs=[]
    for d in delta_list:
        lbl=f"δ={d:.1f}"; print(f"  ▶ {lbl}")
        data=generate_instance(5,10,3,50,seed=42,delta=d)
        res=_run(lbl,solve_ccg,data,zeta,max_iters=200,tol=1e-3,
                 tl_master=600,tl_sp=60)
        if res:
            recs.append({"delta":d,**res})
            print(f"     xi_tot={res['sum_xi']} xk_avg={res['avg_xk']:.2f}"
                  f" xk_tot={res['sum_xk']:.0f}"
                  f" mk={res['avg_mk']:.1f}")
    if not recs: return recs
    fig,axes=plt.subplots(1,3,figsize=(14,4.5)); plt.subplots_adjust(wspace=0.32)
    _three_panel(axes,[r["delta"] for r in recs],
                 [r["sum_xi"] for r in recs],
                 [r["sum_xk"] for r in recs],
                 [r["avg_mk"] for r in recs],
                 xlabel=r"Demand uncertainty half-width $\delta$")
    plt.suptitle(r"Figure 4b  Sensitivity to Uncertainty Width $\delta$",
                 fontweight="bold",y=1.02)
    _save(fig,"exp4b_delta_sensitivity.png")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# §10  Experiment 4c — Waiting-penalty coefficient ρ_k
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 4c)
The waiting-discomfort coefficient ρ_k represents the linearised humanitarian
cost of delayed 3D-print delivery (per unit printed).  When ρ_k is small, 3D
printing remains cost-competitive and the model relies on printers to fill
inventory gaps.  As ρ_k grows, the effective cost of printing approaches the
shortage penalty M_ij, at which point it becomes preferable to pre-stock more
traditional inventory instead.  We scale ρ_k globally by factors
ρ_scale ∈ {0.25, 0.5, 1.0, 2.0, 4.0, 8.0} and analyse the resulting
substitution between x̄_i, x̄_k, and m̄_k.
"""

def exp4c_rho_sensitivity(zeta=10.0, seeds: Optional[List[int]] = None):
    print("\n" + "═"*60 + "\nExp 4c  Waiting-Penalty ρ_k Sensitivity\n" + "═"*60)
    rho_list = [0.25,0.5,1.0,2.0,4.0,8.0]
    seeds = DEFAULT_MULTI_SEEDS if seeds is None else list(seeds)
    recs = []
    for rs in rho_list:
        lbl = f"ρ×{rs}"
        print(f"  ▶ {lbl}")
        summary = _multi_seed_sweep(
            lbl, "rho_scale", rs, seeds,
            lambda seed, rs=rs: generate_instance(
                5, 10, 3, 50, seed=seed, rho_scale=rs),
            zeta, max_iters=200, tol=1e-3, tl_master=600, tl_sp=60,
        )
        if summary:
            recs.append(summary)
    if not recs:
        return recs

    fig, axes = plt.subplots(1,3,figsize=(14,4.5))
    plt.subplots_adjust(wspace=0.32, bottom=0.18)
    _three_panel(axes,[r["rho_scale"] for r in recs],
                 [r["sum_xi_mean"] for r in recs],
                 [r["sum_xk_mean"] for r in recs],
                 [r["avg_mk_mean"] for r in recs],
                 yerrs=([r["sum_xi_std"] for r in recs],
                        [r["sum_xk_std"] for r in recs],
                        [r["avg_mk_std"] for r in recs]),
                 xlabel=r"Waiting-penalty scale factor $\rho_{\mathrm{scale}}$",
                 xlog=True)
    fig.text(0.5, 0.04,
             "Across random seeds; shaded band shows +/- 1 std.",
             ha="center", fontsize=9)
    plt.suptitle(r"Figure 4c  Sensitivity to Waiting-Penalty Coefficient $\rho_k$",
                 fontweight="bold",y=1.02)
    _save(fig,"exp4c_rho_sensitivity.png")
    return recs


# ══════════════════════════════════════════════════════════════════════════════
# §11  Experiment 4d — Joint T × δ heatmap
# ══════════════════════════════════════════════════════════════════════════════
"""
PAPER TEXT (Exp 4d)
Both T and δ independently push decisions towards higher pre-positioning; their
interaction may be non-additive.  We run a 3×3 factorial (T ∈ {36,72,120} h,
δ ∈ {0.2,0.4,0.6}) and display the joint effect on x̄_i, x̄_k, and m̄_k.
"""

def exp4d_T_delta_joint(zeta=10.0):
    print("\n" + "═"*60 + "\nExp 4d  T × δ Joint Heatmap\n" + "═"*60)
    T_l=[36,72,120]; D_l=[0.2,0.4,0.6]
    xi_g=np.full((3,3),np.nan); xk_g=np.full((3,3),np.nan); mk_g=np.full((3,3),np.nan)
    for ti,tv in enumerate(T_l):
        for di,dv in enumerate(D_l):
            lbl=f"T={tv}_δ={dv:.1f}"; print(f"  ▶ {lbl}")
            data=generate_instance(5,10,3,50,seed=42,T_val=float(tv),delta=dv)
            res=_run(lbl,solve_ccg,data,zeta,max_iters=200,tol=1e-3,
                     tl_master=600,tl_sp=60)
            if res:
                xi_g[ti,di]=res["sum_xi"]; xk_g[ti,di]=res["sum_xk"]
                mk_g[ti,di]=res["avg_mk"]
                print(f"     xi_tot={res['sum_xi']} xk_avg={res['avg_xk']:.2f}"
                      f" xk_tot={res['sum_xk']:.0f}"
                      f" mk={res['avg_mk']:.1f}")

    Tl=[f"{t}h" for t in T_l]; Dl=[f"{d:.1f}" for d in D_l]
    fig,axes=plt.subplots(1,3,figsize=(15,4.5)); plt.subplots_adjust(wspace=0.35)
    grids=[xi_g,xk_g,mk_g]
    titles=[
        r"(a) $\sum_i x_i$ — Total traditional stock (units)",
        r"(b) $\sum_k x_k$ — Total printer count (units)",
        r"(c) $\bar{m}_k$ — Consumable stock (kg)",
    ]
    for ax,grid,title in zip(axes,grids,titles):
        im=ax.imshow(grid,cmap="Greys",aspect="auto")
        ax.set_xticks(range(3)); ax.set_xticklabels(Dl)
        ax.set_yticks(range(3)); ax.set_yticklabels(Tl)
        ax.set_xlabel(r"Uncertainty half-width $\delta$", fontweight="bold")
        ax.set_ylabel("Rescue window $T$",                fontweight="bold")
        ax.set_title(title,fontweight="bold",pad=10)
        plt.colorbar(im,ax=ax,shrink=0.82)
        vmax=np.nanmax(grid) if not np.all(np.isnan(grid)) else 1.0
        for ti2 in range(3):
            for di2 in range(3):
                v=grid[ti2,di2]
                if not np.isnan(v):
                    ax.text(di2,ti2,f"{v:.1f}",ha="center",va="center",
                            fontsize=9,fontweight="bold",
                            color="white" if v>vmax*0.6 else "black")

    plt.suptitle(r"Figure 4d  Joint Sensitivity: $T \times \delta$",
                 fontweight="bold",y=1.02)
    _save(fig,"exp4d_T_delta_joint.png")
    return xi_g,xk_g,mk_g


# ══════════════════════════════════════════════════════════════════════════════
# §12  Main
# ══════════════════════════════════════════════════════════════════════════════

def exp5_out_of_sample(zeta: float = 10.0, n_oos: int = 300,
                       kappa_list: Optional[List[float]] = None,
                       data_path: Optional[str] = None,
                       train_seed: int = 42) -> Dict:
    """
    Out-of-sample analysis comparing DRO and SAA under demand-distribution shift.
    """
    print("\n" + "="*60 + "\nExp 5  Out-of-Sample Robustness Analysis\n" + "="*60)
    if kappa_list is None:
        kappa_list = [1.0, 1.5, 2.0]

    data_train = _base_instance(data_path=data_path, n_samples=50, seed=train_seed)
    res_dro = _run("Train DRO", solve_ccg, data_train, zeta,
                   max_iters=300, tol=1e-3, tl_master=900, tl_sp=90)
    res_saa = _run("Train SAA", solve_ccg, data_train, 0.0,
                   max_iters=300, tol=1e-3, tl_master=900, tl_sp=90)
    if not (res_dro and res_saa):
        return {}

    ref_dro = evaluate_policy_on_dataset(data_train, res_dro)["mean_total_M"]
    ref_saa = evaluate_policy_on_dataset(data_train, res_saa)["mean_total_M"]

    results = {}
    rows = []
    for idx, kappa in enumerate(kappa_list):
        print(f"  kappa={kappa}")
        data_oos = resample_demands_from_base(
            data_train, n_samples=n_oos, seed=1000 + idx, demand_kappa=kappa
        )
        dro_eval = evaluate_policy_on_dataset(data_oos, res_dro, ref_total_M=ref_dro)
        saa_eval = evaluate_policy_on_dataset(data_oos, res_saa, ref_total_M=ref_saa)
        results[kappa] = {"DRO": dro_eval, "SAA": saa_eval}

        for model_name, summary in (("DRO", dro_eval), ("SAA", saa_eval)):
            rows.append({
                "kappa": kappa,
                "model": model_name,
                "ref_total_M": summary["ref_total_M"],
                "mean_total_M": summary["mean_total_M"],
                "max_total_M": summary["max_total_M"],
                "min_total_M": summary["min_total_M"],
                "p90_total_M": summary["p90_total_M"],
                "p95_total_M": summary["p95_total_M"],
                "mean_dis": summary["mean_dis"],
                "max_dis": summary["max_dis"],
                "min_dis": summary["min_dis"],
                "p90_dis": summary["p90_dis"],
                "p95_dis": summary["p95_dis"],
            })
            print(f"    {model_name:<4} mean DIS={summary['mean_dis']:.2%}"
                  f"  p95 DIS={summary['p95_dis']:.2%}"
                  f"  mean cost={summary['mean_total_M']:.3f} M¥")

    _write_csv_rows(
        "exp5_out_of_sample_summary.csv",
        ["kappa", "model", "ref_total_M", "mean_total_M", "max_total_M", "min_total_M",
         "p90_total_M", "p95_total_M", "mean_dis", "max_dis", "min_dis", "p90_dis", "p95_dis"],
        rows,
    )
    _plot_exp5_oos(results, kappa_list)
    return {"train_data": data_train, "DRO": res_dro, "SAA": res_saa, "rows": rows, "results": results}


def exp6_value_of_3d_printing(zeta: float = 10.0, n_oos: int = 300,
                              kappa_list: Optional[List[float]] = None,
                              data_path: Optional[str] = None,
                              train_seed: int = 42) -> Dict:
    """
    Compare the full model with a no-3D benchmark under out-of-sample scenarios.
    """
    print("\n" + "="*60 + "\nExp 6  Value of 3D Printing Capacity\n" + "="*60)
    if kappa_list is None:
        kappa_list = [1.0, 1.5, 2.0]

    data_train = _base_instance(data_path=data_path, n_samples=50, seed=train_seed)
    res_full = _run("Full DRO", solve_ccg, data_train, zeta,
                    max_iters=300, tol=1e-3, tl_master=900, tl_sp=90)
    res_no3d = _run("No3D DRO", solve_ccg, data_train, zeta,
                    max_iters=300, tol=1e-3, tl_master=900, tl_sp=90, disable_3d=True)
    if not (res_full and res_no3d):
        return {}

    rows = []
    results = {}
    for idx, kappa in enumerate(kappa_list):
        print(f"  kappa={kappa}")
        data_oos = resample_demands_from_base(
            data_train, n_samples=n_oos, seed=2000 + idx, demand_kappa=kappa
        )
        full_eval = evaluate_policy_on_dataset(data_oos, res_full)
        no3d_eval = evaluate_policy_on_dataset(data_oos, res_no3d)
        results[kappa] = {"Full 3D-enabled": full_eval, "No 3D": no3d_eval}

        for policy_name, summary in (("Full 3D-enabled", full_eval), ("No 3D", no3d_eval)):
            rows.append({
                "kappa": kappa,
                "policy": policy_name,
                "mean_total_M": summary["mean_total_M"],
                "max_total_M": summary["max_total_M"],
                "min_total_M": summary["min_total_M"],
                "aggregate_shortage_rate": summary["aggregate_shortage_rate"],
                "mean_wait_share": summary["mean_wait_share"],
                "mean_service_level": summary["mean_service_level"],
            })

        cost_reduction = (
            (no3d_eval["mean_total_M"] - full_eval["mean_total_M"]) / no3d_eval["mean_total_M"]
            if no3d_eval["mean_total_M"] > 1e-9 else 0.0
        )
        shortage_reduction = (
            (no3d_eval["aggregate_shortage_rate"] - full_eval["aggregate_shortage_rate"])
            / no3d_eval["aggregate_shortage_rate"]
            if no3d_eval["aggregate_shortage_rate"] > 1e-9 else 0.0
        )
        print(f"    Full 3D-enabled mean cost={full_eval['mean_total_M']:.3f} M¥,"
              f" shortage={full_eval['aggregate_shortage_rate']:.2%}")
        print(f"    No 3D          mean cost={no3d_eval['mean_total_M']:.3f} M¥,"
              f" shortage={no3d_eval['aggregate_shortage_rate']:.2%}")
        print(f"    cost reduction={cost_reduction:.2%}, shortage reduction={shortage_reduction:.2%}")

    _write_csv_rows(
        "exp6_value_of_3d_printing_summary.csv",
        ["kappa", "policy", "mean_total_M", "max_total_M", "min_total_M",
         "aggregate_shortage_rate", "mean_wait_share", "mean_service_level"],
        rows,
    )
    _plot_exp6_value_3d(rows, kappa_list)
    return {"train_data": data_train, "full_model": res_full, "no3d_model": res_no3d,
            "rows": rows, "results": results}


def exp7_model_benchmarking(N_train: int = 50, n_oos: int = 300,
                            kappa_list: Optional[List[float]] = None,
                            data_path: Optional[str] = None,
                            train_seed: int = 42) -> Dict:
    """
    Benchmark DRO, SAA, and deterministic models under distribution shift.
    """
    print("\n" + "="*60 + "\nExp 7  Model Benchmarking\n" + "="*60)
    if kappa_list is None:
        kappa_list = [1.0, 1.5, 2.0]

    data_train = _base_instance(data_path=data_path, n_samples=N_train, seed=train_seed)
    data_det = build_deterministic_instance_from_training(data_train)

    models = {
        "DRO (zeta=10)": _run("Benchmark DRO", solve_ccg, data_train, 10.0,
                              max_iters=300, tol=1e-3, tl_master=900, tl_sp=90),
        "SAA (zeta=0)": _run("Benchmark SAA", solve_ccg, data_train, 0.0,
                             max_iters=300, tol=1e-3, tl_master=900, tl_sp=90),
        "Deterministic": _run("Benchmark DET", solve_ccg, data_det, 0.0,
                              max_iters=300, tol=1e-3, tl_master=900, tl_sp=90),
    }
    models = {k: v for k, v in models.items() if v is not None}
    if len(models) != 3:
        return {}

    ref_costs = {
        name: evaluate_policy_on_dataset(data_train, res)["mean_total_M"]
        for name, res in models.items()
    }

    rows = []
    latex_rows = []
    for idx, kappa in enumerate(kappa_list):
        print(f"\n--- kappa = {kappa} ---")
        print(f"{'Model':<18} {'Max DIS':>10} {'Min DIS':>10} {'Avg DIS':>10} {'P90':>10} {'P95':>10}")
        data_oos = resample_demands_from_base(
            data_train, n_samples=n_oos, seed=3000 + idx, demand_kappa=kappa
        )
        for name, res in models.items():
            summary = evaluate_policy_on_dataset(data_oos, res, ref_total_M=ref_costs[name])
            row = {
                "kappa": kappa,
                "model": name,
                "max_dis": summary["max_dis"],
                "min_dis": summary["min_dis"],
                "mean_dis": summary["mean_dis"],
                "p90_dis": summary["p90_dis"],
                "p95_dis": summary["p95_dis"],
            }
            rows.append(row)
            latex_rows.append([
                f"{kappa}",
                name,
                f"{summary['max_dis']:.2%}",
                f"{summary['min_dis']:.2%}",
                f"{summary['mean_dis']:.2%}",
                f"{summary['p90_dis']:.2%}",
                f"{summary['p95_dis']:.2%}",
            ])
            print(f"{name:<18} {summary['max_dis']:>9.2%} {summary['min_dis']:>9.2%}"
                  f" {summary['mean_dis']:>9.2%} {summary['p90_dis']:>9.2%} {summary['p95_dis']:>9.2%}")

    _write_csv_rows(
        "exp7_model_benchmarking_summary.csv",
        ["kappa", "model", "max_dis", "min_dis", "mean_dis", "p90_dis", "p95_dis"],
        rows,
    )
    _write_latex_table(
        "exp7_model_benchmarking_table.tex",
        ["Kappa", "Model", "Max DIS", "Min DIS", "Avg DIS", "P90", "P95"],
        latex_rows,
        "Benchmarking of DRO, SAA, and deterministic models under distribution shift.",
        "tab:benchmarking_oos",
    )
    _plot_exp7_benchmark(rows, kappa_list, list(models.keys()))
    return {"train_data": data_train, "models": models, "rows": rows}


if __name__ == "__main__":
    print("="*60)
    print("Wasserstein-DRO Emergency Stockpile — Full Experiments v3")
    print("="*60)

    exp1_scalability(zeta=10.0)
    exp2_zeta_sensitivity()
    exp3_budget_capacity(zeta=10.0)
    exp4a_T_sensitivity(zeta=10.0)
    exp4b_delta_sensitivity(zeta=10.0)
    exp4c_rho_sensitivity(zeta=10.0)
    exp4d_T_delta_joint(zeta=10.0)

    # Heavy out-of-sample / benchmarking experiments.
    # Uncomment as needed after verifying the solver environment and runtime budget.
    # exp5_out_of_sample(zeta=10.0, n_oos=300)
    # exp6_value_of_3d_printing(zeta=10.0, n_oos=300)
    # exp7_model_benchmarking(N_train=50, n_oos=300)

    print("\n" + "="*60)
    print("All experiments complete.  Figures saved to current directory.")
    print("="*60)
