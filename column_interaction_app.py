import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import brentq

# --- Page config ---
st.set_page_config(page_title="Column M-N Interaction Diagram", page_icon="🏛️", layout="wide")
st.title("Concrete Column — M-N Interaction Diagram")
st.caption("Rectangular section, symmetric reinforcement")

# ── Sidebar inputs ────────────────────────────────────────────────────────────
st.sidebar.header("Concrete Properties")
concrete_grades = {
    "C20/25 — f_cu = 25 MPa": 25,
    "C25/30 — f_cu = 30 MPa": 30,
    "C30/37 — f_cu = 37 MPa": 37,
    "C35/45 — f_cu = 45 MPa": 45,
    "C40/50 — f_cu = 50 MPa": 50,
}
concrete_label  = st.sidebar.selectbox("Concrete Grade", list(concrete_grades.keys()), index=1)
f_cu            = concrete_grades[concrete_label]
epsilon_cu      = st.sidebar.number_input("Ultimate strain ε_cu", value=0.0035, format="%.4f")

st.sidebar.header("Reinforcement Properties")
rebar_grades = {
    "f_y = 500 MPa": 500,
    "f_y = 450 MPa": 450,
    "f_y = 250 MPa": 250,
}
rebar_label = st.sidebar.selectbox("Reinforcement Grade", list(rebar_grades.keys()), index=1)
f_y         = rebar_grades[rebar_label]
E_s         = st.sidebar.number_input("E_s (MPa)", value=200000, step=1000)

st.sidebar.header("Section Geometry")
h = st.sidebar.number_input("Section depth h (mm)",      min_value=100, max_value=2000, value=450, step=25)
b = st.sidebar.number_input("Section width b (mm)",      min_value=100, max_value=1000, value=350, step=25)
c = st.sidebar.number_input("Cover to bar centre (mm)",  min_value=20,  max_value=150,  value=60,  step=5)

st.sidebar.header("Reinforcement Areas")
As       = st.sidebar.number_input("Tension steel Aₛ (mm²)",       min_value=100, max_value=20000, value=1608, step=50)
As_prime = st.sidebar.number_input("Compression steel Aₛ' (mm²)",  min_value=100, max_value=20000, value=1608, step=50)

# ── Core calculations ─────────────────────────────────────────────────────────
def run_interaction(f_cu, epsilon_cu, f_y, E_s, h, b, c, As, As_prime):
    d       = h - c
    d_prime = c

    epsilon_y  = (0.87 * f_y) / E_s
    f_yc       = f_y / (1.15 + f_y / 2000)
    epsilon_yc = f_yc / E_s
    f_yc_mod   = f_yc - 0.45 * f_cu

    def concrete_force(x):
        s = min(0.9 * x, h)
        return 0.45 * f_cu * b * s

    def concrete_moment(x):
        s = min(0.9 * x, h)
        return 0.45 * f_cu * b * s * (h / 2 - s / 2)

    def tension_steel_stress(x):
        eps = -((d - x) / x) * epsilon_cu
        return -0.87 * f_y if eps < -epsilon_y else E_s * eps

    def comp_steel_stress(x):
        eps = ((x - d_prime) / x) * epsilon_cu
        return f_yc_mod if eps >= epsilon_yc else (eps * E_s - 0.45 * f_cu)

    def tension_steel_strain(x):
        return -((d - x) / x) * epsilon_cu

    def comp_steel_strain(x):
        return ((x - d_prime) / x) * epsilon_cu

    points = {}

    # Point 1 — NA at compression steel
    x1   = d_prime
    fst1 = tension_steel_stress(x1)
    fsc1 = 0.0
    N1   = concrete_force(x1) + fsc1 * As_prime + fst1 * As
    M1   = concrete_moment(x1) + fsc1 * As_prime * (h/2 - d_prime) - fst1 * As * (d - h/2)
    points[1] = {"label": "NA at comp. steel level", "x": x1,
                 "eps_st": tension_steel_strain(x1), "eps_sc": 0.0,
                 "f_st": fst1, "f_sc": fsc1, "N": N1, "M": M1}

    # Point 2 — Pure flexure (N = 0)
    def net_force_p2(x):
        if x <= 0:
            return 1e9
        fst = -0.87 * f_y
        eps_sc = ((x - d_prime) / x) * epsilon_cu
        fsc = (eps_sc * E_s - 0.45 * f_cu) if eps_sc < epsilon_yc else f_yc_mod
        return concrete_force(x) + fsc * As_prime + fst * As

    try:
        x2 = brentq(net_force_p2, 1, h * 2)
    except Exception:
        x2 = d_prime
    fst2 = tension_steel_stress(x2)
    fsc2 = comp_steel_stress(x2)
    N2   = 0.0
    M2   = concrete_moment(x2) + fsc2 * As_prime * (h/2 - d_prime) - fst2 * As * (d - h/2)
    points[2] = {"label": "Pure flexure (N = 0)", "x": x2,
                 "eps_st": tension_steel_strain(x2), "eps_sc": comp_steel_strain(x2),
                 "f_st": fst2, "f_sc": fsc2, "N": N2, "M": M2}

    # Point 3 — Compression steel just yields
    x3   = (epsilon_cu / (epsilon_cu - epsilon_yc)) * d_prime
    fst3 = tension_steel_stress(x3)
    fsc3 = f_yc_mod
    N3   = concrete_force(x3) + fsc3 * As_prime + fst3 * As
    M3   = concrete_moment(x3) + fsc3 * As_prime * (h/2 - d_prime) - fst3 * As * (d - h/2)
    points[3] = {"label": "Comp. steel yields", "x": x3,
                 "eps_st": tension_steel_strain(x3), "eps_sc": epsilon_yc,
                 "f_st": fst3, "f_sc": fsc3, "N": N3, "M": M3}

    # Point 4 — Balanced failure
    x4   = (epsilon_cu / (epsilon_cu + epsilon_y)) * d
    fst4 = -0.87 * f_y
    fsc4 = comp_steel_stress(x4)
    N4   = concrete_force(x4) + fsc4 * As_prime + fst4 * As
    M4   = concrete_moment(x4) + fsc4 * As_prime * (h/2 - d_prime) - fst4 * As * (d - h/2)
    points[4] = {"label": "Balanced failure", "x": x4,
                 "eps_st": -epsilon_y, "eps_sc": comp_steel_strain(x4),
                 "f_st": fst4, "f_sc": fsc4, "N": N4, "M": M4}

    # Point 5 — NA at tension steel
    x5   = d
    fst5 = 0.0
    fsc5 = comp_steel_stress(x5)
    N5   = concrete_force(x5) + fsc5 * As_prime + fst5 * As
    M5   = concrete_moment(x5) + fsc5 * As_prime * (h/2 - d_prime) - fst5 * As * (d - h/2)
    points[5] = {"label": "NA at tension steel", "x": x5,
                 "eps_st": 0.0, "eps_sc": comp_steel_strain(x5),
                 "f_st": fst5, "f_sc": fsc5, "N": N5, "M": M5}

    # Point 6 — Full compression
    x6   = (epsilon_cu / (epsilon_cu - epsilon_yc)) * d
    fst6 = f_yc_mod
    eps_sc6 = ((x6 - d_prime) / x6) * epsilon_cu
    fsc6 = f_yc_mod if eps_sc6 >= epsilon_yc else (eps_sc6 * E_s - 0.45 * f_cu)
    N6   = 0.45 * f_cu * b * h + fsc6 * As_prime + fst6 * As
    M6   = 0.45 * f_cu * b * h * (h/2 - h/2) + fsc6 * As_prime * (h/2 - d_prime) - fst6 * As * (d - h/2)
    points[6] = {"label": "Full compression", "x": x6,
                 "eps_st": epsilon_yc, "eps_sc": eps_sc6,
                 "f_st": fst6, "f_sc": fsc6, "N": N6, "M": M6}

    return points

# ── Run & display ─────────────────────────────────────────────────────────────
if st.sidebar.button("▶  Generate Diagram", type="primary"):

    points = run_interaction(f_cu, epsilon_cu, f_y, E_s, h, b, c, As, As_prime)

    M_vals = [points[i]["M"] / 1e6 for i in range(1, 7)]
    N_vals = [points[i]["N"] / 1e3 for i in range(1, 7)]

    col1, col2 = st.columns([1.2, 1])

    # ── Interaction diagram ───────────────────────────────────────────────────
    with col1:
        st.subheader("Interaction diagram")
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(M_vals, N_vals, "o-", color="#1D9E75", linewidth=2,
                markersize=7, label="M-N envelope")
        for i in range(1, 7):
            ax.annotate(
                f" {i}. {points[i]['label']}",
                (M_vals[i - 1], N_vals[i - 1]),
                fontsize=8, color="#444441",
            )
        ax.axhline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
        ax.axvline(0, color="#B4B2A9", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Moment M (kN·m)", fontsize=11)
        ax.set_ylabel("Axial Force N (kN)", fontsize=11)
        ax.set_title("M-N Interaction Diagram", fontsize=12, fontweight="normal")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(fontsize=9)
        fig.tight_layout()
        st.pyplot(fig)

    # ── Results table ─────────────────────────────────────────────────────────
    with col2:
        st.subheader("Results — all 6 points")
        rows = []
        for i in range(1, 7):
            p = points[i]
            rows.append({
                "Pt":              i,
                "Description":     p["label"],
                "x (mm)":         f"{p['x']:.1f}",
                "ε_st":           f"{p['eps_st']:.5f}",
                "ε_sc":           f"{p['eps_sc']:.5f}",
                "f_st (MPa)":     f"{p['f_st']:.1f}",
                "f_sc (MPa)":     f"{p['f_sc']:.1f}",
                "N (kN)":         f"{p['N']/1e3:.1f}",
                "M (kN·m)":       f"{p['M']/1e6:.1f}",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("**Section & material summary**")
        st.markdown(f"""
        | Parameter | Value |
        |---|---|
        | f_cu | {f_cu} MPa |
        | f_y | {f_y} MPa |
        | h × b | {h} × {b} mm |
        | Cover | {c} mm |
        | Aₛ | {As} mm² |
        | Aₛ' | {As_prime} mm² |
        """)