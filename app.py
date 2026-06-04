"""
Logistics Bullseye - MFC Suitability Framework
AHP-weighted WASPAS / TOPSIS site-selection model with what-if sensitivity.

Author: Muhammad Tayseer Wadiwala (MS25GL109)
IBR Term 2 - SP Jain School of Global Management

Run locally:   streamlit run app.py
Deploy:        push this folder to GitHub, then deploy app.py on share.streamlit.io
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# Page config
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Logistics Bullseye - MFC Suitability Model",
                   page_icon="🎯", layout="wide")

NAVY = "#1E2761"; GOLD = "#E8A33D"; RED = "#C0392B"; GREEN = "#2E8B57"; GREY = "#6B7280"

# ----------------------------------------------------------------------------
# Default data (illustrative - mirrors the report demonstration)
# ----------------------------------------------------------------------------
CLUSTERS = ["A: Demand Density", "B: Time-Dependent Travel",
            "C: Zoning & Regulatory", "D: Energy & Sustainability"]

# Saaty 1-9 pairwise comparison (rows vs columns), justified by interview ranking
DEFAULT_PAIRWISE = pd.DataFrame(
    [[1, 2, 3, 5],
     [1/2, 1, 1, 4],
     [1/3, 1, 1, 4],
     [1/5, 1/4, 1/4, 1]],
    index=CLUSTERS, columns=CLUSTERS)

# sub-variable shares within each cluster (from thematic emphasis)
SUBS = {
    "A": [("A1 Population density", 0.25), ("A2 Household count", 0.30), ("A3 Order generation (FTA)", 0.45)],
    "B": [("B1 Peak-time volatility", 0.45), ("B2 Average traffic speed", 0.25), ("B3 Corridor/resupply access", 0.30)],
    "C": [("C1 Land-use permissibility", 0.30), ("C2 Zoning/charging feasibility", 0.25), ("C3 Urban density mix", 0.45)],
    "D": [("D1 Grid-emission cleanliness", 0.35), ("D2 Low-emission vehicle feasibility", 0.40), ("D3 Energy intensity / order", 0.25)],
}
CRIT = [name for key in ["A", "B", "C", "D"] for name, _ in SUBS[key]]
# +1 = benefit (higher better), -1 = cost (higher worse)
DIRECTION = np.array([+1, +1, +1, -1, +1, +1, +1, +1, +1, +1, +1, -1])

SITES = ["Dubai Marina", "JLT", "Business Bay", "Downtown", "JVC",
         "Discovery Gardens", "Deira", "Mirdif", "Al Quoz", "Dubai South"]
X_DEFAULT = np.array([
    [90, 85, 92, 85, 40, 55, 50, 40, 88, 55, 45, 75],
    [88, 82, 90, 80, 45, 70, 55, 45, 90, 55, 48, 72],
    [80, 70, 85, 78, 50, 75, 60, 50, 85, 58, 55, 70],
    [78, 72, 80, 82, 42, 65, 52, 45, 87, 58, 52, 72],
    [75, 80, 70, 60, 60, 68, 58, 55, 70, 60, 58, 65],
    [82, 78, 68, 58, 62, 72, 55, 50, 68, 60, 55, 63],
    [85, 80, 60, 88, 35, 50, 60, 40, 75, 45, 40, 78],
    [45, 55, 50, 45, 70, 65, 55, 55, 50, 60, 60, 60],
    [25, 20, 30, 40, 75, 85, 90, 88, 45, 70, 85, 55],
    [35, 40, 38, 30, 85, 90, 85, 90, 55, 80, 90, 50],
], dtype=float)
RI = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32}

# ----------------------------------------------------------------------------
# Core computations
# ----------------------------------------------------------------------------
def ahp_weights(M):
    M = np.array(M, dtype=float)
    n = M.shape[0]
    gm = np.prod(M, axis=1) ** (1 / n)
    w = gm / gm.sum()
    lam_max = (M @ w / w).mean()
    CI = (lam_max - n) / (n - 1) if n > 1 else 0.0
    CR = CI / RI.get(n, 1.12) if RI.get(n, 1.12) else 0.0
    return w, lam_max, CR

def global_weights(cluster_w, shares):
    out = []
    for i, key in enumerate(["A", "B", "C", "D"]):
        for (_, share) in shares[key]:
            out.append(cluster_w[i] * share)
    return np.array(out)

def waspas(X, w, direction, lam=0.5):
    R = np.zeros_like(X, dtype=float)
    for j in range(X.shape[1]):
        col = X[:, j]
        R[:, j] = col / col.max() if direction[j] > 0 else col.min() / col
    WSM = (R * w).sum(axis=1)
    WPM = np.prod(R ** w, axis=1)
    return lam * WSM + (1 - lam) * WPM

def topsis(X, w, direction):
    norm = X / np.sqrt((X ** 2).sum(axis=0))
    V = norm * w
    best = np.where(direction > 0, V.max(axis=0), V.min(axis=0))
    worst = np.where(direction > 0, V.min(axis=0), V.max(axis=0))
    d_best = np.sqrt(((V - best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)

def ranks(scores):
    order = np.argsort(-scores)
    r = np.empty_like(order); r[order] = np.arange(1, len(scores) + 1)
    return r

# ----------------------------------------------------------------------------
# Session state init
# ----------------------------------------------------------------------------
if "pairwise" not in st.session_state:
    st.session_state.pairwise = DEFAULT_PAIRWISE.copy()
if "shares" not in st.session_state:
    st.session_state.shares = {k: list(v) for k, v in SUBS.items()}
if "sites_df" not in st.session_state:
    st.session_state.sites_df = pd.DataFrame(X_DEFAULT, index=SITES, columns=CRIT)

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    f"<div style='background:{NAVY};padding:18px 24px;border-radius:10px'>"
    f"<span style='color:{GOLD};font-size:13px;letter-spacing:2px;font-weight:700'>LOGISTICS BULLSEYE &nbsp;·&nbsp; IBR TERM 2</span><br>"
    f"<span style='color:white;font-size:28px;font-weight:800'>MFC Suitability Framework</span><br>"
    f"<span style='color:#CADCFC;font-size:15px'>AHP-weighted WASPAS / TOPSIS site selection with what-if sensitivity &nbsp;·&nbsp; Dubai</span>"
    f"</div>", unsafe_allow_html=True)

st.caption("⚠️ Illustrative proxy data. This demonstrates the framework end-to-end; "
           "live deployment requires proprietary order-flow, travel-time, grid-capacity and land-permission data.")

cluster_w, lam_max, CR = ahp_weights(st.session_state.pairwise.values)
gw = global_weights(cluster_w, st.session_state.shares)
gw = gw / gw.sum()

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["① AHP Weights", "② Candidate Sites", "③ Ranking", "④ What-If Sensitivity", "ℹ️ About"])

# ============================================================ TAB 1 ==========
with tab1:
    st.subheader("Step 1 — Criterion weights from expert judgment (AHP)")
    c1, c2 = st.columns([1.15, 1])
    with c1:
        st.markdown("**Cluster pairwise comparison** (Saaty 1–9; row vs column). Edit to re-weight.")
        edited = st.data_editor(st.session_state.pairwise, use_container_width=True,
                                key="pw_editor", num_rows="fixed")
        st.session_state.pairwise = edited
        cluster_w, lam_max, CR = ahp_weights(edited.values)
        gw = global_weights(cluster_w, st.session_state.shares)
        gw = gw / gw.sum()
        cr_ok = CR < 0.10
        st.metric("Consistency Ratio (CR)", f"{CR:.4f}",
                  "consistent (<0.10)" if cr_ok else "inconsistent — revise",
                  delta_color="normal" if cr_ok else "inverse")
        st.caption(f"λmax = {lam_max:.4f}.  CR below 0.10 means the pairwise judgments are internally consistent.")
    with c2:
        fig = go.Figure(go.Bar(
            x=cluster_w, y=CLUSTERS, orientation="h",
            marker_color=[NAVY, NAVY, NAVY, GOLD],
            text=[f"{v:.3f}" for v in cluster_w], textposition="outside"))
        fig.update_layout(title="Cluster weights", height=320,
                          margin=dict(l=10, r=10, t=40, b=10),
                          xaxis_range=[0, max(cluster_w) * 1.25], yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**Sub-variable shares within each cluster** (normalised per cluster).")
    cols = st.columns(4)
    new_shares = {}
    for i, key in enumerate(["A", "B", "C", "D"]):
        with cols[i]:
            st.markdown(f"**Cluster {key}**")
            rows = []
            for name, share in st.session_state.shares[key]:
                val = st.number_input(name, 0.0, 1.0, float(share), 0.05, key=f"sh_{name}")
                rows.append((name, val))
            tot = sum(v for _, v in rows) or 1.0
            new_shares[key] = [(n, v / tot) for n, v in rows]
    st.session_state.shares = new_shares
    gw = global_weights(cluster_w, st.session_state.shares); gw = gw / gw.sum()

    st.markdown("**Resulting global criterion weights (12)**")
    gwdf = pd.DataFrame({"Criterion": CRIT,
                         "Direction": ["Benefit" if d > 0 else "Cost" for d in DIRECTION],
                         "Global weight": np.round(gw, 4)})
    st.dataframe(gwdf, use_container_width=True, hide_index=True)

# ============================================================ TAB 2 ==========
with tab2:
    st.subheader("Step 2 — Candidate sites & decision matrix")
    st.markdown("Scores 0–100 per criterion. **B1 Peak-time volatility** and **D3 Energy intensity** are "
                "cost criteria (higher = worse). Edit any value, or add/remove rows.")
    edited_sites = st.data_editor(st.session_state.sites_df, use_container_width=True,
                                  num_rows="dynamic", key="sites_editor")
    st.session_state.sites_df = edited_sites
    st.caption("Tip: keep values on a 0–100 scale. Direction (benefit/cost) is fixed per criterion — see the About tab.")

# ============================================================ TAB 3 ==========
with tab3:
    st.subheader("Step 3 — Site ranking (WASPAS & TOPSIS)")
    lam = st.slider("WASPAS λ  (0 = pure product model, 1 = pure sum model)", 0.0, 1.0, 0.5, 0.05)
    df = st.session_state.sites_df.dropna()
    X = df.values.astype(float)
    site_names = list(df.index)
    if X.shape[1] != len(CRIT):
        st.error("The site matrix must have exactly 12 criteria columns.")
    else:
        Q = waspas(X, gw, DIRECTION, lam); Cc = topsis(X, gw, DIRECTION)
        rQ, rC = ranks(Q), ranks(Cc)
        res = pd.DataFrame({"Site": site_names,
                            "WASPAS": np.round(Q, 4), "WASPAS rank": rQ,
                            "TOPSIS (Ci)": np.round(Cc, 4), "TOPSIS rank": rC}
                           ).sort_values("WASPAS rank")
        c1, c2 = st.columns([1, 1.1])
        with c1:
            st.dataframe(res, use_container_width=True, hide_index=True)
            agree = int((rQ == rC).sum())
            st.metric("Rank agreement (WASPAS vs TOPSIS)", f"{agree}/{len(site_names)} sites")
        with c2:
            o = np.argsort(-Q)
            fig = go.Figure()
            fig.add_bar(y=[site_names[i] for i in o][::-1], x=[Q[i] for i in o][::-1],
                        orientation="h", name="WASPAS", marker_color=NAVY)
            fig.add_bar(y=[site_names[i] for i in o][::-1], x=[Cc[i] for i in o][::-1],
                        orientation="h", name="TOPSIS", marker_color=GOLD)
            fig.update_layout(barmode="group", height=440, title="Suitability scores",
                              margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        st.success(f"Top recommendation under current weights: **{res.iloc[0]['Site']}**")

# ============================================================ TAB 4 ==========
with tab4:
    st.subheader("Step 4 — What-if sensitivity")
    st.markdown("Shift the **cluster weights** to model a structural change, then see how the ranking moves. "
                "Presets correspond to the report's what-if scenarios.")
    preset = st.radio("Scenario preset", ["Baseline (interview weights)",
                                          "Scenario B — peak congestion",
                                          "Scenario D — energy mandate", "Custom"],
                      horizontal=True)
    base = cluster_w
    if preset == "Baseline (interview weights)":
        cw = base
    elif preset == "Scenario B — peak congestion":
        cw = np.array([0.40, 0.40, 0.13, 0.07])
    elif preset == "Scenario D — energy mandate":
        cw = np.array([0.34, 0.20, 0.18, 0.28])
    else:
        cw = base
    cols = st.columns(4)
    cw_in = []
    for i, c in enumerate(CLUSTERS):
        cw_in.append(cols[i].slider(c, 0.0, 1.0, float(round(cw[i], 3)), 0.01, key=f"sc_{i}"))
    cw_in = np.array(cw_in); cw_in = cw_in / cw_in.sum()

    df = st.session_state.sites_df.dropna(); X = df.values.astype(float); site_names = list(df.index)
    gw_base = global_weights(base, st.session_state.shares); gw_base /= gw_base.sum()
    gw_sc = global_weights(cw_in, st.session_state.shares); gw_sc /= gw_sc.sum()
    r_base = ranks(waspas(X, gw_base, DIRECTION))
    Q_sc = waspas(X, gw_sc, DIRECTION); r_sc = ranks(Q_sc)

    comp = pd.DataFrame({"Site": site_names, "Baseline rank": r_base,
                         "Scenario rank": r_sc, "Δ": r_base - r_sc}
                        ).sort_values("Scenario rank")
    c1, c2 = st.columns([1, 1.2])
    with c1:
        st.dataframe(comp, use_container_width=True, hide_index=True)
        st.caption("Δ > 0 means the site improved (moved up) under the scenario.")
    with c2:
        fig = go.Figure()
        for i, s in enumerate(site_names):
            fig.add_trace(go.Scatter(x=["Baseline", "Scenario"], y=[r_base[i], r_sc[i]],
                                     mode="lines+markers", name=s))
        fig.update_layout(title="Rank movement (1 = best)", height=440,
                          yaxis=dict(autorange="reversed", dtick=1),
                          margin=dict(l=10, r=10, t=40, b=10),
                          legend=dict(font=dict(size=9)))
        st.plotly_chart(fig, use_container_width=True)
    st.info(f"Top under this scenario: **{comp.iloc[0]['Site']}**  ·  "
            f"Baseline top: **{site_names[int(np.argmin(r_base))]}**")

# ============================================================ TAB 5 ==========
with tab5:
    st.subheader("About this model")
    st.markdown(f"""
**Logistics Bullseye** demonstrates an AHP-weighted multi-criteria decision model for
micro-fulfilment centre (MFC) placement in Dubai, built for the IBR Term 2 project of
**Muhammad Tayseer Wadiwala (MS25GL109)**, SP Jain School of Global Management.

**Method**
1. **AHP** converts expert pairwise judgments into criterion weights (with a consistency check, CR < 0.10).
2. **WASPAS** (weighted sum + weighted product) and **TOPSIS** (closeness to the ideal) independently rank sites — agreement between them validates robustness.
3. **Sensitivity analysis** re-weights the clusters to model the study's what-if scenarios.

**Criterion directions**
- *Benefit* (higher = more suitable): all except the two below.
- *Cost* (higher = less suitable): **B1 Peak-time volatility**, **D3 Energy intensity / order**.

**Data disclaimer**
All candidate-site scores are **illustrative proxies** reflecting the qualitative profile of each
district from the secondary research and expert interviews. They are **not** live operator or
authority data. The framework is built to ingest real zonal data unchanged once available.

**Expert panel status:** of six identified experts, two have been interviewed to date
(Saif Meer, Mehran Solkar); four are pending. Weights derived from the interviews will be
updated as the panel is completed.
""")
    st.caption("Built with Streamlit · pandas · numpy · plotly")
