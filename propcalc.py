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

# ---------------- Polynomial Coefficients (Table 1) ---------------- #
# Format: (coeff, s, t, u, v)

KT_COEFFS = [
    (+0.00880496,0,0,0,0), (-0.204554,1,0,0,0), (+0.166351,0,1,0,0),
    (+0.158114,0,2,0,0), (-0.147581,2,0,1,0), (-0.481497,1,1,1,0),
    (+0.415437,0,2,1,0), (+0.0144043,0,0,0,1), (-0.0530054,2,0,0,1),
    (+0.0143481,0,1,0,1), (+0.0606826,1,1,0,1), (-0.0125894,0,0,1,1),
    (+0.0109689,1,0,1,1), (-0.133698,0,3,0,0), (+0.00638407,0,6,0,0),
    (-0.00132718,2,6,0,0), (+0.168496,3,0,1,0), (-0.0507214,0,0,2,0),
    (+0.0854559,2,0,2,0), (-0.0504475,3,0,2,0), (+0.010465,1,6,2,0),
    (-0.00648272,2,6,2,0), (-0.00841728,0,3,0,1), (+0.0168424,1,3,0,1),
    (-0.00102296,3,3,0,1), (-0.0317791,0,3,1,1), (+0.018604,1,0,2,1),
    (-0.00410798,0,2,2,1), (-0.000606848,0,0,0,2), (-0.0049819,1,0,0,2),
    (+0.0025983,2,0,0,2), (-0.000560528,3,0,0,2), (-0.00163652,1,2,0,2),
    (-0.000328787,1,6,0,2), (+0.000116502,2,6,0,2), (+0.000690904,0,0,1,2),
    (+0.00421749,0,3,1,2), (+0.0000565229,3,6,1,2), (-0.00146564,0,3,2,2),
]

KQ_COEFFS = [
    (+0.00379368,0,0,0,0), (+0.00886523,2,0,0,0), (-0.032241,1,1,0,0),
    (+0.00344778,0,2,0,0), (-0.0408811,0,1,1,0), (-0.108009,1,1,1,0),
    (-0.0885381,2,1,1,0), (+0.188561,0,2,1,0), (-0.00370871,1,0,0,1),
    (+0.00513696,0,1,0,1), (+0.0209449,1,1,0,1), (+0.00474319,2,1,0,1),
    (-0.00723408,2,0,1,1), (+0.00438388,1,1,1,1), (-0.0269403,0,2,1,1),
    (+0.0558082,3,0,1,0), (+0.0161886,0,3,1,0), (+0.00318086,1,3,1,0),
    (+0.015896,0,0,2,0), (+0.0471729,1,0,2,0), (+0.0196283,3,0,2,0),
    (-0.0502782,0,1,2,0), (-0.030055,3,1,2,0), (+0.0417122,2,2,2,0),
    (-0.0397722,0,3,2,0), (-0.00350024,0,6,2,0), (-0.0106854,3,0,0,1),
    (+0.00110903,3,3,0,1), (-0.000313912,0,6,0,1), (+0.0035985,3,0,1,1),
    (-0.00142121,0,6,1,1), (-0.00383637,1,0,2,1), (+0.0126803,0,2,2,1),
    (-0.00318278,2,3,2,1), (+0.00334268,0,6,2,1), (-0.00183491,1,1,0,2),
    (+0.000112451,3,2,0,2), (-0.0000297228,3,6,0,2), (+0.000269551,1,0,1,2),
    (+0.00083265,2,0,1,2), (+0.00155334,0,2,1,2), (+0.000302683,0,6,1,2),
    (-0.0001843,0,0,2,2), (-0.000425399,0,3,2,2), (+0.0000869243,3,3,2,2),
    (-0.0004659,0,6,2,2), (+0.0000554194,1,6,2,2)
]


# ---------------- Reynolds Corrections (Table 2 full) ---------------- #
# NOTE on notation: In Publication 237, the Reynolds correction uses log R_n
# where R_n := Re / 10^6 (Re is the usual Reynolds number). Therefore the
# term (log R_n - 0.301) means log10(Re/1e6) - log10(2). We implement it as
# `log10Rn = math.log10(Re/1e6)` to match the paper exactly.

def reynolds_corrections(J: float, PD: float, AE: float, Z: int, Re: float):
    if Re <= 2e6:
        return 0.0, 0.0
    log10Rn = math.log10(Re/1e6)
    # ΔKT terms (Table 2)
    dKT = (0.000353485
           -0.00333758*(AE)*J**2
           -0.00478125*(AE)*(PD)*J
           +0.000257792*(log10Rn-0.301)**2*(AE)*J**2
           +0.0000643192*(log10Rn-0.301)*(PD**6)*J**2
           -0.0000110636*(log10Rn-0.301)**2*(PD**6)*J**2
           -0.0000276305*(log10Rn-0.301)**2*Z*(AE)*J**2
           +0.0000954*(log10Rn-0.301)*Z*(AE)*(PD)*J
           +0.0000032049*(log10Rn-0.301)*(Z**2)*(AE)*(PD**3)*J)

    # ΔKQ terms (Table 2)
    dKQ = (-0.000591412
           +0.00696898*(PD)
           -0.0000666654*Z*(PD**6)
           +0.0160818*(AE**2)
           -0.000938091*(log10Rn-0.301)*(PD)
           -0.00059593*(log10Rn-0.301)*(PD**2)
           +0.0000782099*(log10Rn-0.301)**2*(PD**2)
           +0.0000052199*(log10Rn-0.301)*Z*(AE)*J**2
           -0.00000088528*(log10Rn-0.301)**2*Z*(AE)*(PD)*J
           +0.0000230171*(log10Rn-0.301)*Z*(PD**6)
           -0.00000184341*(log10Rn-0.301)**2*Z*(PD**6)
           -0.00400252*(log10Rn-0.301)*(AE**2)
           +0.000220915*(log10Rn-0.301)**2*(AE**2))

    return dKT, dKQ

def eval_poly(J, PD, AE, Z, coeffs):
    return sum(c * (J**s) * (PD**t) * (AE**u) * (Z**v) for c,s,t,u,v in coeffs)

def bseries_eval_poly(J, PD, AE, Z, Re=2e6):
    KT = eval_poly(J, PD, AE, Z, KT_COEFFS)
    KQ = eval_poly(J, PD, AE, Z, KQ_COEFFS)
    dKT, dKQ = reynolds_corrections(J, PD, AE, Z, Re)
    return KT+dKT, KQ+dKQ

def compute_re_075R(Va, n, D, nu=1.1e-6, c075_over_D=0.10):
    r = 0.75 * (D/2.0)
    Vrel = (Va**2 + (2*math.pi*n*r)**2)**0.5
    return Vrel * (c075_over_D*D) / nu

# --- Optimizer ---
KTS_TO_MS = 0.514444

def optimize_propeller(resistance_kN: float,
                       speed_knots: float,
                       diameter_max_m: float|None=None,
                       *,
                       n_propellers: int = 1,
                       mode: str="max_efficiency",
                       eta0_min: float=0.55,
                       wake_fraction: float=0.20,
                       thrust_deduction: float=0.10,
                       rpm_min: float=300.0,
                       rpm_max: float=2000.0,
                       tip_speed_max: float=70.0,
                       Z_list=(2,3,4,5,6,7),
                       AE_list=(0.3,0.4,.5,.6,.7,.8,.9,1),
                       PD_list=(0.5,0.6,0.7,0.8,0.9,1,1.1,1.2,1.3,1.4),
                       D_min_fraction: float=0.70,
                       default_D_range=(0.3,3),
                       nu: float=1.1e-6,
                       rho: float=1025.0) -> dict:
    """Main API: optimize a Wageningen B-series propeller from minimal inputs."""
    R = resistance_kN*1000.0
    Vs = speed_knots*KTS_TO_MS
    Va = Vs*(1.0-wake_fraction)
    # Total thrust required
    T_total = R / (1.0 - thrust_deduction)
    trade_space = []

    # Per-prop thrust requirement
    T_req = T_total / max(1, n_propellers)

    if diameter_max_m:
        Dmax = diameter_max_m
        Dmin = Dmax*D_min_fraction
    else:
        Dmin,Dmax = default_D_range
    D_vals = np.linspace(Dmin,Dmax,30)
    rpm_vals = np.linspace(rpm_min,rpm_max,50)

    best=None; best_metric=-1; best_power=float("inf")

    for Z in Z_list:
        for AE in AE_list:
            for PD in PD_list:
                for D in D_vals:
                    for rpm in rpm_vals:
                        n=rpm/60.0
                        J=Va/(n*D)
                        if not (0.2<=J<=1.4): continue
                        tip_speed=2*math.pi*n*(D/2)
                        if tip_speed>tip_speed_max: continue
                        Re=compute_re_075R(Va,n,D,nu)
                        KT,KQ=bseries_eval_poly(J,PD,AE,Z,Re)
                        if KQ<=0: continue
                        T=rho*n**2*D**4*KT
                        if T<T_req: continue
                        Q=rho*n**2*D**5*KQ
                        P=2*math.pi*n*Q
                        eta0=(J*KT)/(2*math.pi*KQ)

                        row=dict(Z=Z,AE_A0=AE,P_over_D=PD,D_m=D,RPM=rpm,
                                 n_rps=n,J=J,KT=KT,KQ=KQ,
                                 Thrust_N=T,Torque_Nm=Q,Power_W=P,Efficiency=eta0,
                                 tip_speed_mps=tip_speed)
                        trade_space.append(row)
                        if mode=="min_power_eta":
                            if eta0<eta0_min: continue
                            if P<best_power: best=row; best_power=P
                        else: # max_efficiency
                            if eta0>best_metric or (abs(eta0-best_metric)<1e-4 and P<best_power):
                                best=row; best_metric=eta0; best_power=P
                                best["n_propellers"] = n_propellers
                                best["Total_Thrust_N"] = T_total
                                best["Total_Power_W"] = best["Power_W"] * n_propellers
                                best["Total_Torque_Nm"] = best["Torque_Nm"] * n_propellers
                                best["trade_space"]=trade_space
    if not best:
        return {"feasible":False,"message":"No feasible solution found."}
    best["feasible"]=True
    best["message"] = (
        f"{n_propellers} × propellers: "
        f"Z={best['Z']}, AE/A0={best['AE_A0']}, P/D={best['P_over_D']}, "
        f"D={best['D_m']:.2f} m, RPM={best['RPM']:.0f}, "
        f"η0={best['Efficiency']:.3f}, "
        f"Per-prop Power={best['Power_W'] / 1000:.1f} kW, "
        f"Total Power={best['Total_Power_W'] / 1000:.1f} kW"
    )

    return best

# --- Plotting ---
def plot_selected_prop(best: dict, Re: float=2e6):
    J_vals=np.linspace(0.0,1.4,200)
    KT_vals,KQ_vals,eta_vals=[],[],[]
    PD,AE,Z=best['P_over_D'],best['AE_A0'],best['Z']
    for J in J_vals:
        KT,KQ=bseries_eval_poly(J,PD,AE,Z,Re)
        KT_vals.append(KT); KQ_vals.append(KQ)
        eta_vals.append(J*KT/(2*math.pi*KQ) if KQ>1e-9 else math.nan)
    fig,ax1=plt.subplots(figsize=(8,6),dpi=150)
    ax2=ax1.twinx()
    ax1.plot(J_vals,KT_vals,'-',label='KT (solid)')
    ax2.plot(J_vals,KQ_vals,'--',label='KQ (dashed)')
    ax1.plot(J_vals,eta_vals,':',label='η0 (dotted)')
    ax1.axvline(best['J'],color='red',linestyle=':',label='Selected J')
    ax1.set_xlabel("Advance Coefficient J")
    ax1.set_ylabel("KT, η0")
    ax2.set_ylabel("KQ")
    ax1.legend(loc="best")
    plt.title("Selected Propeller Performance")
    plt.tight_layout()
    return fig

def plot_efficiency_map(trade_space, color_by="P_over_D", filters=None, best=None):
    import pandas as pd
    import plotly.express as px

    df = pd.DataFrame(trade_space)

    # Apply filters if provided
    if filters:
        for key, vals in filters.items():
            if vals:
                df = df[df[key].isin(vals)]

    fig = px.scatter(
        df, x="J", y="Efficiency",
        color=color_by,
        hover_data=["Z","AE_A0","P_over_D","D_m","RPM","Power_W","Thrust_N"],
        color_continuous_scale="Viridis",
        title=f"Efficiency map colored by {color_by}",
    )

    # Highlight the selected prop if passed
    if best:
        fig.add_scatter(
            x=[best["J"]],
            y=[best["Efficiency"]],
            mode="markers",
            marker=dict(color="red", size=12, symbol="x"),
            name="Selected"
        )

    fig.update_layout(
        xaxis_title="Advance Coefficient J",
        yaxis_title="Efficiency η₀",
        template="plotly_white",
        height=500
    )
    return fig



def plot_efficiency_surface(trade_space, resolution=50, best=None):
    import pandas as pd
    import numpy as np
    from scipy.interpolate import griddata
    import plotly.graph_objects as go

    df = pd.DataFrame(trade_space)
    df["J_bin"] = (df["J"]*100).round(0)/100
    df["PD_bin"] = (df["P_over_D"]*100).round(0)/100
    grouped = df.groupby(["J_bin","PD_bin"])["Efficiency"].max().reset_index()

    Js = grouped["J_bin"].values
    PDs = grouped["PD_bin"].values
    Etas = grouped["Efficiency"].values

    J_grid = np.linspace(min(Js), max(Js), resolution)
    PD_grid = np.linspace(min(PDs), max(PDs), resolution)
    J_mesh, PD_mesh = np.meshgrid(J_grid, PD_grid)
    Eta_mesh = griddata((Js, PDs), Etas, (J_mesh, PD_mesh), method="linear")

    fig = go.Figure(data=[go.Surface(z=Eta_mesh, x=J_mesh, y=PD_mesh, colorscale="Viridis")])

    # Highlight best prop if provided
    if best:
        fig.add_trace(go.Scatter3d(
            x=[best["J"]],
            y=[best["P_over_D"]],
            z=[best["Efficiency"]],
            mode="markers",
            marker=dict(color="red", size=6, symbol="diamond"),
            name="Selected"
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
