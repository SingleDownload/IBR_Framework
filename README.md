# Logistics Bullseye — MFC Suitability Framework 🎯

An AHP-weighted **WASPAS / TOPSIS** decision model for micro-fulfilment centre (MFC)
placement in Dubai, with what-if sensitivity analysis.

Built for the IBR Term 2 project of **Muhammad Tayseer Wadiwala (MS25GL109)**,
SP Jain School of Global Management.

---

## What it does

| Tab | Purpose |
|-----|---------|
| ① **AHP Weights** | Edit the Saaty pairwise matrix → derive cluster weights + consistency ratio (CR), set sub-variable shares, view the 12 global criterion weights |
| ② **Candidate Sites** | Edit / add / remove candidate districts and their 0–100 criterion scores |
| ③ **Ranking** | Rank sites with WASPAS and TOPSIS; check rank agreement; adjust the WASPAS λ |
| ④ **What-If Sensitivity** | Shift cluster weights (presets for the report's scenarios) and watch the ranking move |
| ⑤ **About** | Method, criterion directions, and data disclaimer |

> ⚠️ **Illustrative data.** The candidate-site scores are proxy values reflecting each
> district's qualitative profile from the research — not live operator/authority data.
> The framework is built to ingest real zonal data unchanged once available.

---

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the URL it prints (usually http://localhost:8501).

---

## Deploy free on Streamlit Community Cloud

1. **Create a GitHub repo** (e.g. `logistics-bullseye-mfc`) and upload these three files to the **root** of the repo:
   - `app.py`
   - `requirements.txt`
   - `README.md`
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **New app** → select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy**. First build takes ~1–2 minutes; your app gets a public `*.streamlit.app` URL.

To update the app later, just push changes to the repo — Streamlit redeploys automatically.

---

## Method (one-paragraph summary)

The four variable clusters (Demand Density, Time-Dependent Travel, Zoning & Regulatory,
Energy & Sustainability) are weighted via **AHP** from expert pairwise judgments, with a
consistency ratio kept below 0.10. Sites are then scored with **WASPAS** (weighted sum +
weighted product) and **TOPSIS** (relative closeness to the ideal solution); agreement between
the two methods indicates a robust ranking. A **sensitivity analysis** re-weights the clusters
to model structural changes (peak congestion, an energy mandate), showing how the recommended
placement shifts with the decision environment.

Criterion directions: all benefit-type except **B1 Peak-time volatility** and
**D3 Energy intensity / order**, which are cost-type (higher = less suitable).
