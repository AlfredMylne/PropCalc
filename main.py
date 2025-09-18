import streamlit as st
from propcalc import optimize_propeller, plot_selected_prop, plot_efficiency_map, plot_efficiency_surface

st.title("Wageningen B-Series Propeller Sizing")

resistance = st.number_input("Ship Resistance (kN)", min_value=0.0, value=25.0)
speed = st.number_input("Ship Speed (knots)", min_value=0.0, value=12.0)
dmax = st.number_input("Max Propeller Diameter (m, optional, 0=ignore)", min_value=0.0, value=0.0)
nprop = st.number_input("Number of Propellers", min_value=1, max_value=4, value=1, step=1)

if st.button("Calculate"):
    dmax_val = None if dmax <= 0 else dmax
    best = optimize_propeller(resistance, speed,
                              diameter_max_m=dmax_val,
                              n_propellers=nprop)
    st.session_state["best_result"] = best

if "best_result" in st.session_state:
    best = st.session_state["best_result"]

    if not best.get("feasible", False):
        st.error(best["message"])
    else:
        summary = {k: v for k, v in best.items() if k != "trade_space"}
        st.success(best["message"])
        st.json(summary)

        # Chosen prop curves
        fig = plot_selected_prop(best)
        st.pyplot(fig)

        # Efficiency map controls (placed just above plot)
        st.subheader("Efficiency Map of Trade Space")

        view_mode = st.radio(
            "Choose view:",
            ["Scatter (raw candidates)", "3D Surface (best efficiency)"]
        )

        if view_mode == "Scatter (raw candidates)":
            # First set up controls
            color_by = st.selectbox(
                "Color points by:", ["P_over_D", "AE_A0", "Z", "D_m", "RPM"]
            )

            st.write("Filter candidates:")
            fZ = st.multiselect("Blades (Z)", sorted(set(c["Z"] for c in best["trade_space"])))
            fPD = st.multiselect("Pitch/Diameter", sorted(set(c["P_over_D"] for c in best["trade_space"])))
            fAE = st.multiselect("AE/A0", sorted(set(c["AE_A0"] for c in best["trade_space"])))

            filters = {"Z": fZ, "P_over_D": fPD, "AE_A0": fAE}

            # Now call plotting
            fig_map = plot_efficiency_map(best["trade_space"], color_by=color_by, filters=filters, best=best)
            st.plotly_chart(fig_map, use_container_width=True)

        else:
            fig_surface = plot_efficiency_surface(best["trade_space"], best=best)
            st.plotly_chart(fig_surface, use_container_width=True)
