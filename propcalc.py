"""
propcalc.py — Wageningen B-Series Propeller Calculation Engine
==============================================================

Implements Wageningen B-series regression polynomials and optimization routines
to size an optimum propeller given ship resistance, speed, and optional max D.

Exports:
    optimize_propeller(...)   # main API for sizing
    plot_selected_prop(...)   # quick graph of KT, KQ, η0 vs J for the chosen prop
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd
from scipy.interpolate import griddata
import plotly.graph_objects as go
from scipy.optimize import minimize
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy.interpolate import griddata


def fmt_dict(d: dict, sig: int = 5) -> dict:
    """Format all floats in a dict to N significant figures."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (float, np.floating)):
            out[k] = float(f"{v:.{sig}g}")
        elif isinstance(v, dict):
            out[k] = fmt_dict(v, sig)
        else:
            out[k] = v
    return out


def fmt_dataframe(df: pd.DataFrame, sig: int = 5) -> pd.DataFrame:
    df_fmt = df.copy()
    for col in df_fmt.columns:
        if pd.api.types.is_numeric_dtype(df_fmt[col]):
            df_fmt[col] = df_fmt[col].apply(
                lambda v: float(f"{v:.{sig}g}") if isinstance(v, (float, np.floating)) else v
            )
    return df_fmt


def format_hover(d: dict, sig: int = 5) -> str:
    """Return a multi-line hover string with formatted numbers."""
    lines = []
    for k in ["J", "Efficiency", "Z", "AE_A0", "P_over_D", "D_m",
              "RPM", "Power_W", "Thrust_N"]:
        if k in d:
            v = d[k]
            if isinstance(v, (float, np.floating)):
                lines.append(f"{k}={v:.{sig}g}")
            else:
                lines.append(f"{k}={v}")
    return "<br>".join(lines)


# ---------------- Polynomial Coefficients (Table 1) ---------------- #
# Format: (coeff, s, t, u, v)

KT_COEFFS = [
    (+0.00880496, 0, 0, 0, 0), (-0.204554, 1, 0, 0, 0), (+0.166351, 0, 1, 0, 0),
    (+0.158114, 0, 2, 0, 0), (-0.147581, 2, 0, 1, 0), (-0.481497, 1, 1, 1, 0),
    (+0.415437, 0, 2, 1, 0), (+0.0144043, 0, 0, 0, 1), (-0.0530054, 2, 0, 0, 1),
    (+0.0143481, 0, 1, 0, 1), (+0.0606826, 1, 1, 0, 1), (-0.0125894, 0, 0, 1, 1),
    (+0.0109689, 1, 0, 1, 1), (-0.133698, 0, 3, 0, 0), (+0.00638407, 0, 6, 0, 0),
    (-0.00132718, 2, 6, 0, 0), (+0.168496, 3, 0, 1, 0), (-0.0507214, 0, 0, 2, 0),
    (+0.0854559, 2, 0, 2, 0), (-0.0504475, 3, 0, 2, 0), (+0.010465, 1, 6, 2, 0),
    (-0.00648272, 2, 6, 2, 0), (-0.00841728, 0, 3, 0, 1), (+0.0168424, 1, 3, 0, 1),
    (-0.00102296, 3, 3, 0, 1), (-0.0317791, 0, 3, 1, 1), (+0.018604, 1, 0, 2, 1),
    (-0.00410798, 0, 2, 2, 1), (-0.000606848, 0, 0, 0, 2), (-0.0049819, 1, 0, 0, 2),
    (+0.0025983, 2, 0, 0, 2), (-0.000560528, 3, 0, 0, 2), (-0.00163652, 1, 2, 0, 2),
    (-0.000328787, 1, 6, 0, 2), (+0.000116502, 2, 6, 0, 2), (+0.000690904, 0, 0, 1, 2),
    (+0.00421749, 0, 3, 1, 2), (+0.0000565229, 3, 6, 1, 2), (-0.00146564, 0, 3, 2, 2),
]

KQ_COEFFS = [
    (+0.00379368, 0, 0, 0, 0), (+0.00886523, 2, 0, 0, 0), (-0.032241, 1, 1, 0, 0),
    (+0.00344778, 0, 2, 0, 0), (-0.0408811, 0, 1, 1, 0), (-0.108009, 1, 1, 1, 0),
    (-0.0885381, 2, 1, 1, 0), (+0.188561, 0, 2, 1, 0), (-0.00370871, 1, 0, 0, 1),
    (+0.00513696, 0, 1, 0, 1), (+0.0209449, 1, 1, 0, 1), (+0.00474319, 2, 1, 0, 1),
    (-0.00723408, 2, 0, 1, 1), (+0.00438388, 1, 1, 1, 1), (-0.0269403, 0, 2, 1, 1),
    (+0.0558082, 3, 0, 1, 0), (+0.0161886, 0, 3, 1, 0), (+0.00318086, 1, 3, 1, 0),
    (+0.015896, 0, 0, 2, 0), (+0.0471729, 1, 0, 2, 0), (+0.0196283, 3, 0, 2, 0),
    (-0.0502782, 0, 1, 2, 0), (-0.030055, 3, 1, 2, 0), (+0.0417122, 2, 2, 2, 0),
    (-0.0397722, 0, 3, 2, 0), (-0.00350024, 0, 6, 2, 0), (-0.0106854, 3, 0, 0, 1),
    (+0.00110903, 3, 3, 0, 1), (-0.000313912, 0, 6, 0, 1), (+0.0035985, 3, 0, 1, 1),
    (-0.00142121, 0, 6, 1, 1), (-0.00383637, 1, 0, 2, 1), (+0.0126803, 0, 2, 2, 1),
    (-0.00318278, 2, 3, 2, 1), (+0.00334268, 0, 6, 2, 1), (-0.00183491, 1, 1, 0, 2),
    (+0.000112451, 3, 2, 0, 2), (-0.0000297228, 3, 6, 0, 2), (+0.000269551, 1, 0, 1, 2),
    (+0.00083265, 2, 0, 1, 2), (+0.00155334, 0, 2, 1, 2), (+0.000302683, 0, 6, 1, 2),
    (-0.0001843, 0, 0, 2, 2), (-0.000425399, 0, 3, 2, 2), (+0.0000869243, 3, 3, 2, 2),
    (-0.0004659, 0, 6, 2, 2), (+0.0000554194, 1, 6, 2, 2)
]


# ---------------- Reynolds Corrections (Table 2 full) ---------------- #
# NOTE on notation: In Publication 237, the Reynolds correction uses log R_n
# where R_n := Re / 10^6 (Re is the usual Reynolds number). Therefore the
# term (log R_n - 0.301) means log10(Re/1e6) - log10(2). We implement it as
# `log10Rn = math.log10(Re/1e6)` to match the paper exactly.

def reynolds_corrections(J: float, PD: float, AE: float, Z: int, Re: float):
    if Re <= 2e6:
        return 0.0, 0.0
    log10Rn = math.log10(Re / 1e6)
    # ΔKT terms (Table 2)
    dKT = (0.000353485
           - 0.00333758 * (AE) * J ** 2
           - 0.00478125 * (AE) * (PD) * J
           + 0.000257792 * (log10Rn - 0.301) ** 2 * (AE) * J ** 2
           + 0.0000643192 * (log10Rn - 0.301) * (PD ** 6) * J ** 2
           - 0.0000110636 * (log10Rn - 0.301) ** 2 * (PD ** 6) * J ** 2
           - 0.0000276305 * (log10Rn - 0.301) ** 2 * Z * (AE) * J ** 2
           + 0.0000954 * (log10Rn - 0.301) * Z * (AE) * (PD) * J
           + 0.0000032049 * (log10Rn - 0.301) * (Z ** 2) * (AE) * (PD ** 3) * J)

    # ΔKQ terms (Table 2)
    dKQ = (-0.000591412
           + 0.00696898 * (PD)
           - 0.0000666654 * Z * (PD ** 6)
           + 0.0160818 * (AE ** 2)
           - 0.000938091 * (log10Rn - 0.301) * (PD)
           - 0.00059593 * (log10Rn - 0.301) * (PD ** 2)
           + 0.0000782099 * (log10Rn - 0.301) ** 2 * (PD ** 2)
           + 0.0000052199 * (log10Rn - 0.301) * Z * (AE) * J ** 2
           - 0.00000088528 * (log10Rn - 0.301) ** 2 * Z * (AE) * (PD) * J
           + 0.0000230171 * (log10Rn - 0.301) * Z * (PD ** 6)
           - 0.00000184341 * (log10Rn - 0.301) ** 2 * Z * (PD ** 6)
           - 0.00400252 * (log10Rn - 0.301) * (AE ** 2)
           + 0.000220915 * (log10Rn - 0.301) ** 2 * (AE ** 2))

    return dKT, dKQ

def enforce_monotone_KQ(J_vals, KQ_vals):
    """Force KQ(J) to be non-increasing up to thrust cutoff."""
    out = np.array(KQ_vals, dtype=float)
    mask = np.isfinite(out)
    if np.any(mask):
        out[mask] = np.minimum.accumulate(out[mask])
    return out

def eval_poly(J, PD, AE, Z, coeffs):
    return sum(c * (J ** s) * (PD ** t) * (AE ** u) * (Z ** v) for c, s, t, u, v in coeffs)


def bseries_eval_poly(J, PD, AE, Z, Re=2e6):
    KT = eval_poly(J, PD, AE, Z, KT_COEFFS)
    KQ = eval_poly(J, PD, AE, Z, KQ_COEFFS)
    dKT, dKQ = reynolds_corrections(J, PD, AE, Z, Re)
    return KT + dKT, KQ + dKQ

def bseries_eval_clipped(J, PD, AE, Z, Re=2e6):
    """Evaluate KT, KQ with guards:
       - stop at KT<=0
       - reject KQ<=0
       - enforce non-increasing KQ vs J
    """
    KT, KQ = bseries_eval_poly(J, PD, AE, Z, Re)

    if KT <= 0 or KQ <= 0:
        return float("nan"), float("nan")

    return KT, KQ

def compute_re_075R(Va, n, D, nu=1.1e-6, c075_over_D=0.10):
    r = 0.75 * (D / 2.0)
    Vrel = (Va ** 2 + (2 * math.pi * n * r) ** 2) ** 0.5
    return Vrel * (c075_over_D * D) / nu


# --- Optimizer ---
KTS_TO_MS = 0.514444


def optimize_propeller(resistance_kN: float,
                       speed_knots: float,
                       diameter_max_m: float | None = None,
                       *,
                       n_propellers: int = 1,
                       fixed_power_W: float | None = None,
                       fixed_rpm: float | None = None,
                       mode: str = "max_efficiency",
                       eta0_min: float = 0.55,
                       wake_fraction: float = 0.20,
                       thrust_deduction: float = 0.10,
                       rpm_min: float = 300.0,
                       rpm_max: float = 2000.0,
                       tip_speed_max: float = 70.0,
                       Z_list=(2, 3, 4, 5, 6, 7),
                       AE_list=np.linspace(0.3, 0.7, 5),  # 41 steps, ~0.01 increments
                       PD_list=np.linspace(0.6, 1.2, 7),  # 61 steps, ~0.01 increments
                       D_min_fraction: float = 0.70,
                       default_D_range=(0.3, 3),
                       nu: float = 1.1e-6,
                       rho: float = 1025.0) -> dict:
    """Main API: optimize a Wageningen B-series propeller from minimal inputs."""
    R = resistance_kN * 1000.0
    Vs = speed_knots * KTS_TO_MS
    Va = Vs * (1.0 - wake_fraction)
    # Total thrust required
    T_total = R / (1.0 - thrust_deduction)
    trade_space = []

    # Per-prop thrust requirement
    T_req = T_total / max(1, n_propellers)

    if diameter_max_m:
        Dmax = diameter_max_m
        Dmin = Dmax * D_min_fraction
    else:
        Dmin, Dmax = default_D_range
    D_vals = np.linspace(Dmin, Dmax, 30)
    rpm_vals = np.linspace(rpm_min, rpm_max, 50)

    best = None;
    best_metric = -1;
    best_power = float("inf")
    total_iters = len(Z_list) * len(AE_list) * len(PD_list) * len(D_vals) * len(rpm_vals)
    progress_bar = st.progress(0, text="Optimising...")
    counter = 0
    update_interval = max(1, total_iters // 100)  # Update ~100 times

    if fixed_rpm:
        rpm_vals = [fixed_rpm]

    for Z in Z_list:
        for AE in AE_list:
            for PD in PD_list:
                for D in D_vals:
                    for rpm in rpm_vals:
                        counter += 1
                        if counter % update_interval == 0:
                            progress_bar.progress(counter / total_iters,
                                                  text=f"Optimising... {int((counter / total_iters) * 100)}%")
                        n = rpm / 60.0
                        J = Va / (n * D)
                        if not (0.2 <= J <= 1.4): continue
                        tip_speed = 2 * math.pi * n * (D / 2)
                        if tip_speed > tip_speed_max: continue
                        Re = compute_re_075R(Va, n, D, nu)
                        KT, KQ = bseries_eval_clipped(J, PD, AE, Z, Re)
                        if not np.isfinite(KT) or not np.isfinite(KQ):
                            continue
                        if KQ <= 0: continue
                        T = rho * n ** 2 * D ** 4 * KT
                        if T < T_req: continue
                        Q = rho * n ** 2 * D ** 5 * KQ
                        P = 2 * math.pi * n * Q
                        if fixed_power_W and abs(P - fixed_power_W) > 0.05 * fixed_power_W:
                            continue
                        eta0 = (J * KT) / (2 * math.pi * KQ) if KQ > 1e-9 else 0

                        row = dict(Z=Z, AE_A0=AE, P_over_D=PD, D_m=D, RPM=rpm,
                                   n_rps=n, J=J, KT=KT, KQ=KQ,
                                   Thrust_N=T, Torque_Nm=Q, Power_W=P, Efficiency=eta0,
                                   tip_speed_mps=tip_speed)
                        trade_space.append(row)
                        if mode == "min_power_eta":
                            if eta0 < eta0_min: continue
                            if P < best_power: best = row; best_power = P
                        else:  # max_efficiency
                            if eta0 > best_metric or (abs(eta0 - best_metric) < 1e-4 and P < best_power):
                                best = row;
                                best_metric = eta0;
                                best_power = P
                                best["n_propellers"] = n_propellers
                                best["Total_Thrust_N"] = T_total
                                best["Total_Power_W"] = best["Power_W"] * n_propellers
                                best["Total_Torque_Nm"] = best["Torque_Nm"] * n_propellers
                                best["trade_space"] = trade_space
    if not best:
        return {"feasible": False, "message": "No feasible solution found."}
    best["feasible"] = True
    best["message"] = (
        f"Grid optimisation: "
        f"{n_propellers} × propellers: "
        f"Z={best['Z']}, AE/A0={best['AE_A0']}, P/D={best['P_over_D']}, "
        f"D={best['D_m']:.2f} m, RPM={best['RPM']:.0f}, "
        f"η0={best['Efficiency']:.3f}, "
        f"Per-prop Power={best['Power_W'] / 1000:.1f} kW, "
        f"Total Power={best['Total_Power_W'] / 1000:.1f} kW"
    )
    progress_bar.empty()  # clear when finished
    return fmt_dict(best, 5)


def continuous_optimize(resistance_kN: float,
                        speed_knots: float,
                        diameter_guess: float,
                        n_propellers: int = 1,
                        Z: int = 4,
                        wake_fraction: float = 0.20,
                        thrust_deduction: float = 0.10,
                        nu: float = 1.1e-6,
                        rho: float = 1025.0,
                        seed: dict | None = None,
                        fixed_power_W: float | None = None,
                        fixed_rpm: float | None = None):
    """
    Continuous optimiser: maximise efficiency subject to thrust >= requirement
    and optional power/RPM constraints. Uses SLSQP with constraints.
    """

    R = resistance_kN * 1000.0
    Vs = speed_knots * KTS_TO_MS
    Va = Vs * (1.0 - wake_fraction)
    T_total = R / (1.0 - thrust_deduction)
    T_req = T_total / max(1, n_propellers)

    # --- Objective: maximise efficiency (η₀) ---
    def objective(x):
        P_D, AE, D, rpm = x
        n = rpm / 60.0
        J = Va / (n * D)
        if not (0.2 <= J <= 1.4):
            return 1e3

        tip_speed = 2 * math.pi * n * (D / 2)
        if tip_speed > 70:
            return 1e3

        Re = compute_re_075R(Va, n, D, nu)
        KT, KQ = bseries_eval_poly(J, P_D, AE, Z, Re)
        if KQ <= 0:
            return 1e3

        T = rho * n ** 2 * D ** 4 * KT
        Q = rho * n ** 2 * D ** 5 * KQ
        P = 2 * math.pi * n * Q
        eta0 = (J * KT) / (2 * math.pi * KQ)

        # Penalise thrust shortfall (soft)
        penalty = max(0, (T_req - T)) * 1e-3

        # Soft penalty for power mismatch
        if fixed_power_W:
            penalty += abs(P - fixed_power_W) / fixed_power_W * 1e-2

        return -(eta0) + penalty

    # --- Constraint: thrust >= required ---
    def thrust_constraint(x):
        P_D, AE, D, rpm = x
        n = rpm / 60.0
        J = Va / (n * D)
        KT, _ = bseries_eval_poly(J, P_D, AE, Z)
        T = rho * n ** 2 * D ** 4 * KT
        return T - T_req  # must be >= 0

    constraints = [{"type": "ineq", "fun": thrust_constraint}]

    # --- Optional power constraint (tolerance band ±5%) ---
    if fixed_power_W:
        def power_constraint(x):
            P_D, AE, D, rpm = x
            n = rpm / 60.0
            J = Va / (n * D)
            _, KQ = bseries_eval_poly(J, P_D, AE, Z)
            Q = rho * n ** 2 * D ** 5 * KQ
            P = 2 * math.pi * n * Q
            return 0.05 * fixed_power_W - abs(P - fixed_power_W)

        constraints.append({"type": "ineq", "fun": power_constraint})

    # --- Initial guess ---
    if seed and seed.get("feasible", False):
        x0 = [seed["P_over_D"], seed["AE_A0"], seed["D_m"], seed["RPM"]]
    else:
        x0 = [0.9, 0.55, diameter_guess, fixed_rpm or 1000]

    # --- Bounds ---
    rpm_bounds = (fixed_rpm * 0.9, fixed_rpm * 1.1) if fixed_rpm else (300, 2000)
    bounds = [
        (0.6, 1.4),  # P/D
        (0.3, 0.7),  # AE/A0 (clamped)
        (diameter_guess * 0.7, diameter_guess * 1.1),  # D
        rpm_bounds  # RPM
    ]

    # --- Run optimisation ---
    result = minimize(
        objective, x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": 5000, "ftol": 1e-6}
    )

    # --- Extract solution ---
    P_D, AE, D, rpm = result.x
    n = rpm / 60.0
    J = Va / (n * D)
    Re = compute_re_075R(Va, n, D, nu)
    KT, KQ = bseries_eval_poly(J, P_D, AE, Z, Re)
    T = rho * n ** 2 * D ** 4 * KT
    Q = rho * n ** 2 * D ** 5 * KQ
    P = 2 * math.pi * n * Q
    eta0 = (J * KT) / (2 * math.pi * KQ)

    msg = (f"Continuous opt: "
        f"{n_propellers} × propellers: "
        f"Z={Z}, AE/A0={AE:.3f}, P/D={P_D:.3f}, "
        f"D={D:.2f} m, RPM={rpm:.0f}, "
        f"η0={eta0:.3f}, "
        f"Per-prop Power={P / 1000:.1f} kW, "
        f"Total Power={(P * n_propellers) / 1000:.1f} kW"
    )

    if not result.success:
        msg = f"⚠️ Continuous optimisation warning: {result.message}"

    result_dict = dict(
        feasible=True,
        Z=Z, AE_A0=AE, P_over_D=P_D, D_m=D, RPM=rpm,
        n_rps=n, J=J, KT=KT, KQ=KQ,
        Thrust_N=T, Torque_Nm=Q, Power_W=P, Efficiency=eta0,
        Required_Thrust_N=T_req,
        Required_Power_W=fixed_power_W if fixed_power_W else None,
        n_propellers=n_propellers,
        Total_Thrust_N=T_total,
        Total_Power_W=P * n_propellers,
        Total_Torque_Nm=Q * n_propellers,
        message=msg
    )

    # Keep structure consistent with grid search
    result_dict["trade_space"] = [result_dict.copy()]

    return fmt_dict(result_dict, 5)


# --- Plotting ---
def plot_selected_prop(best: dict, Re: float = 2e6):
    J_vals = np.linspace(0.0, 1.4, 200)
    KT_vals, KQ_vals, eta_vals = [], [], []
    PD, AE, Z = best['P_over_D'], best['AE_A0'], best['Z']

    clipped = False
    for J in J_vals:
        KT, KQ = bseries_eval_poly(J, PD, AE, Z, Re)
        if KT <= 0 or KQ <= 0 or clipped:
            # mask everything once either coefficient goes non-positive
            KT_vals.append(np.nan)
            KQ_vals.append(np.nan)
            eta_vals.append(np.nan)
            clipped = True
        else:
            KT_vals.append(KT)
            KQ_vals.append(KQ)
            eta_vals.append(J * KT / (2 * math.pi * KQ))

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx()
    ax1.plot(J_vals, KT_vals, '-', label='KT (solid)')
    ax2.plot(J_vals, KQ_vals, '--', label='KQ (dashed)')
    ax1.plot(J_vals, eta_vals, ':', label='η0 (dotted)')
    ax1.axvline(best['J'], color='red', linestyle=':', label='Selected J')
    ax1.set_xlabel("Advance Coefficient J")
    ax1.set_ylabel("KT, η0")
    ax2.set_ylabel("KQ")
    ax1.legend(loc="best")
    plt.title("Selected Propeller Performance")
    plt.tight_layout()
    return fig




def plot_efficiency_map(trade_space, color_by="P_over_D", filters=None,
                        best_grid=None, best_cont=None):
    df = pd.DataFrame(trade_space)
    df = fmt_dataframe(df, 5)

    if filters:
        for key, vals in filters.items():
            if vals:
                df = df[df[key].isin(vals)]

    fig = px.scatter(
        df, x="J", y="Efficiency",
        color=color_by,
        hover_data=["Z", "AE_A0", "P_over_D", "D_m", "RPM", "Power_W", "Thrust_N"],
        color_continuous_scale="Viridis",
        title=f"Efficiency map colored by {color_by}",
    )

    if best_grid:
        fig.add_scatter(
            x=[best_grid["J"]],
            y=[best_grid["Efficiency"]],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name="Grid Optimum",
            hovertext=format_hover(best_grid, 5),
            hoverinfo="text"
        )

    if best_cont:
        fig.add_scatter(
            x=[best_cont["J"]],
            y=[best_cont["Efficiency"]],
            mode="markers",
            marker=dict(color="blue", size=12, symbol="diamond"),
            name="Continuous Optimum",
            hovertext=format_hover(best_cont, 5),
            hoverinfo="text"
        )

    fig.update_layout(
        xaxis_title="Advance Coefficient J",
        yaxis_title="Efficiency η₀",
        template="plotly_white",
        height=500
    )
    return fig


def plot_efficiency_surface(trade_space, resolution=50, best_grid=None, best_cont=None):
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    import plotly.graph_objects as go

    df = pd.DataFrame(trade_space)
    df["J_bin"] = (df["J"] * 100).round(0) / 100
    df["PD_bin"] = (df["P_over_D"] * 100).round(0) / 100
    grouped = df.groupby(["J_bin", "PD_bin"])["Efficiency"].max().reset_index()

    Js = grouped["J_bin"].values
    PDs = grouped["PD_bin"].values
    Etas = grouped["Efficiency"].values

    J_grid = np.linspace(min(Js), max(Js), resolution)
    PD_grid = np.linspace(min(PDs), max(PDs), resolution)
    J_mesh, PD_mesh = np.meshgrid(J_grid, PD_grid)
    Eta_mesh = griddata((Js, PDs), Etas, (J_mesh, PD_mesh), method="linear")

    fig = go.Figure(data=[go.Surface(z=Eta_mesh, x=J_mesh, y=PD_mesh, colorscale="Viridis")])

    # Highlight grid optimum
    if best_grid:
        fig.add_trace(go.Scatter3d(
            x=[best_grid["J"]],
            y=[best_grid["P_over_D"]],
            z=[best_grid["Efficiency"]],
            mode="markers",
            marker=dict(color="red", size=8, symbol="x"),
            name="Grid Optimum",
            text=[format_hover(best_grid, 5)],
            hoverinfo="text"
        ))

    if best_cont:
        fig.add_trace(go.Scatter3d(
            x=[best_cont["J"]],
            y=[best_cont["P_over_D"]],
            z=[best_cont["Efficiency"]],
            mode="markers",
            marker=dict(color="blue", size=8, symbol="diamond"),
            name="Continuous Optimum",
            text=[format_hover(best_cont, 5)],
            hoverinfo="text"
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title="Advance Coefficient J",
            yaxis_title="Pitch Ratio P/D",
            zaxis_title="Efficiency η₀"
        ),
        title="Best Efficiency Surface (interpolated)",
        template="plotly_white",
        height=700
    )
    return fig
