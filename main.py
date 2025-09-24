import streamlit as st
import plotly.graph_objects as go
from propcalc import (
    optimize_propeller,
    plot_selected_prop,
    plot_efficiency_map,
    plot_efficiency_surface,
    continuous_optimize,
    fmt_dict
)

with st.expander("ℹ️ What does this app do?"):
    st.markdown(
        """
        This app helps you **size an optimum Wageningen B-Series propeller**  
        based on your vessel’s resistance, speed, and optional diameter limits.

        **What it does:**
        - Calculates thrust and power requirements  
        - Optimises propeller parameters (Diameter, RPM, Blade count, AE/A0, P/D)  
        - Finds the most efficient feasible propeller using **grid search**  
        - Refines the result with **continuous optimisation** for higher accuracy  
        - Allows optional inputs for **fixed shaft power** and/or **fixed RPM**  
        - Reports per-prop and total power for multi-propeller setups  

        **Outputs include:**
        - Summary of the chosen propeller(s)  
        - Full optimisation results (expandable)  
        - Interactive plots:
            - KT, KQ, and η₀ vs. J curves  
            - Efficiency trade-space maps (scatter & 3D surface)

        ---
        **How to use:**
        1. Enter your vessel’s **resistance** (kN) and **speed** (knots).  
        2. Optionally provide maximum diameter, shaft power, or fixed RPM.  
        3. Choose the **number of propellers** and **blades** (or let optimiser decide).  
        4. Click **Calculate** to run both grid and continuous optimisation.  
        5. Review the summary, expand results, and open plots as needed.
        """
    )

st.title("Marine Propeller Sizing Tool")

col1, col2 = st.columns(2)

with col1:
    resistance = st.number_input("Ship Resistance (kN)", min_value=0.0, value=25.0)
    speed = st.number_input("Ship Speed (knots)", min_value=0.0, value=12.0)
    dmax = st.number_input("Max Propeller Diameter (m, 0 = ignore)", min_value=0.0, value=0.0)
    nprop = st.number_input("Number of Propellers", min_value=1, max_value=4, value=1, step=1)

with col2:
    power = st.number_input("Shaft Power (kW, 0 = ignore)", min_value=0.0, value=0.0)
    rpm_fixed = st.number_input("Fixed RPM (0 = ignore)", min_value=0.0, value=0.0)
    n_blades = st.selectbox(
        "Number of Blades (Z, 0 = let optimiser choose)",
        options=[0, 2, 3, 4, 5, 6, 7],
        index=0
    )

power_val = None if power <= 0 else power * 1000.0
rpm_val = None if rpm_fixed <= 0 else rpm_fixed

if st.button("Calculate"):
    dmax_val = None if dmax <= 0 else dmax

    # Always run grid search (gives trade space for plotting)
    Z_list = list(range(2, 8)) if n_blades == 0 else [n_blades]

    best_grid = optimize_propeller(
        resistance, speed,
        diameter_max_m=dmax_val,
        n_propellers=nprop,
        Z_list=Z_list,
        fixed_power_W=power_val,
        fixed_rpm=rpm_val
    )
    # Run continuous optimisation, seeded with grid search result
    with st.spinner("Running continuous optimiser..."):
        best_cont = continuous_optimize(
            resistance, speed,
            diameter_guess=dmax_val or best_grid.get("D_m", 1.0),
            Z=(best_grid.get("Z", 4) if n_blades == 0 else n_blades),
            n_propellers=nprop,
            seed=best_grid if best_grid.get("feasible") else None,
            fixed_power_W=power_val,
            fixed_rpm=rpm_val
        )
    st.success("Continuous optimisation complete")

    st.session_state["best_grid"] = best_grid
    st.session_state["best_cont"] = best_cont


# --- Results Display ---
if "best_grid" in st.session_state:
    best_grid = st.session_state["best_grid"]
    best_cont = st.session_state.get("best_cont", {})

    st.subheader("Optimisation Results")

    if best_grid.get("feasible", False):
        # Show summary only
        st.success(best_grid["message"])
        # Collapse the full JSON
        with st.expander("See full grid search result"):
            summary = {k: v for k, v in best_grid.items() if k != "trade_space"}
            st.json(fmt_dict(summary, 5))
    else:
        st.error(best_grid["message"])

    if best_cont.get("feasible", False):
        st.success(best_cont["message"])
        with st.expander("See full continuous optimisation result"):
            summary = {k: v for k, v in best_cont.items() if k != "trade_space"}
            st.json(fmt_dict(summary, 5))
    else:
        st.error(best_cont.get("message", "Continuous optimisation failed"))

    # --- Curves, hidden unless requested ---
    if best_grid.get("feasible", False):
        with st.expander("Show grid-selected propeller curves"):
            fig = plot_selected_prop(best_grid)
            st.pyplot(fig)

    if best_cont.get("feasible", False):
        with st.expander("Show continuous-selected propeller curves"):
            fig = plot_selected_prop(best_cont)
            st.pyplot(fig)

    # --- Efficiency Map ---
    with st.expander("Show efficiency map of trade space"):
        view_mode = st.radio(
            "Choose view:",
            ["Scatter (raw candidates)", "3D Surface (best efficiency)"]
        )

        if view_mode == "Scatter (raw candidates)":
            color_by = st.selectbox(
                "Color points by:", ["P_over_D", "AE_A0", "Z", "D_m", "RPM"]
            )

            st.write("Filter candidates:")
            fZ = st.multiselect(
                "Blades (Z)",
                sorted(set(c["Z"] for c in best_grid.get("trade_space", []))),
                default=[best_grid.get("Z")] if best_grid.get("feasible") else []
            )
            fPD = st.multiselect(
                "Pitch/Diameter",
                sorted(set(c["P_over_D"] for c in best_grid.get("trade_space", [])))
            )
            fAE = st.multiselect(
                "AE/A0",
                sorted(set(c["AE_A0"] for c in best_grid.get("trade_space", [])))
            )

            filters = {"Z": fZ, "P_over_D": fPD, "AE_A0": fAE}

            if "trade_space" in best_grid:
                fig_map = plot_efficiency_map(
                    best_grid["trade_space"],
                    color_by=color_by,
                    filters=filters,
                    best_grid=best_grid if best_grid.get("feasible") else None,
                    best_cont=best_cont if best_cont.get("feasible") else None
                )
                st.plotly_chart(fig_map, use_container_width=True)

        else:
            if "trade_space" in best_grid:
                fig_surface = plot_efficiency_surface(
                    best_grid["trade_space"],
                    best_grid=best_grid if best_grid.get("feasible") else None,
                    best_cont=best_cont if best_cont.get("feasible") else None
                )
                st.plotly_chart(fig_surface, use_container_width=True)
