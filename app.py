"""
Wildfire Prediction - Streamlit UI
"""

import json
import math
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from utils import EnsembleClassifier, COUNTY_ELEVATION  # noqa: F401 — needed for pickle
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Config ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
MODEL_DIR = ROOT / "models"
METRICS_F = MODEL_DIR / "metrics.json"

st.set_page_config(
    page_title="California Wildfire Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styles ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1e1e2e;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 4px 0;
        border-left: 4px solid #f97316;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; color: #94a3b8; }
    .metric-card p  { margin: 4px 0 0; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
    .fire-badge  { background:#ef4444; color:white; padding:4px 12px; border-radius:20px; font-weight:700; }
    .safe-badge  { background:#22c55e; color:white; padding:4px 12px; border-radius:20px; font-weight:700; }
    .section-header { border-bottom: 2px solid #f97316; padding-bottom: 6px; margin-bottom: 16px; }
</style>
""", unsafe_allow_html=True)


# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_metrics():
    if not METRICS_F.exists():
        return None
    with open(METRICS_F) as f:
        return json.load(f)


@st.cache_resource
def load_model(name: str):
    safe = name.replace(" ", "_").replace("+", "plus")
    path = MODEL_DIR / f"{safe}.pkl"
    if path.exists():
        return joblib.load(path)
    best = MODEL_DIR / "best_model.pkl"
    return joblib.load(best) if best.exists() else None


@st.cache_resource
def load_feature_cols():
    p = MODEL_DIR / "feature_cols.pkl"
    return joblib.load(p) if p.exists() else []


@st.cache_data
def load_geojson():
    p = MODEL_DIR / "california_counties.geojson"
    with open(p) as f:
        return json.load(f)


@st.cache_resource
def load_shap_explainer(model_name: str):
    model = load_model(model_name)
    if model is None:
        return None
    try:
        return shap.TreeExplainer(model)
    except Exception:
        try:
            return shap.LinearExplainer(model, masker=shap.maskers.Independent(
                pd.DataFrame(columns=load_feature_cols()), max_samples=100
            ))
        except Exception:
            return None


def build_input_row(county, month, temp_max, temp_min, humidity, wind_speed,
                    precip, drought, tmax_7d, hum_7d, tmax_14d, hum_14d,
                    prev_day_fire=0, prev2_day_fire=0, fire_7d=0,
                    wind_dir=180) -> dict:
    month_sin    = math.sin(2 * math.pi * month / 12)
    month_cos    = math.cos(2 * math.pi * month / 12)
    vpd          = max(0, (1 - humidity / 100) * 0.6108 * math.exp(17.27 * temp_max / (temp_max + 237.3)))
    fire_season  = 1 if month in [6, 7, 8, 9, 10] else 0
    wind_dir_rad = wind_dir * math.pi / 180
    elevation    = COUNTY_ELEVATION.get(county, 500)
    # FFWI (Fosberg Fire Weather Index)
    wind_mph = wind_speed * 2.237
    temp_f   = temp_max * 9 / 5 + 32
    h, t     = humidity, temp_f
    if h < 10:
        emc = 0.03229 + 0.281073 * h - 0.000578 * t * h
    elif h <= 50:
        emc = 2.22749 + 0.160107 * h - 0.014784 * t
    else:
        emc = 21.0606 + 0.005565 * h**2 - 0.00035 * t * h - 0.483199 * h
    eta  = max(0.0, min(1.0, emc / 30))
    ffwi = max(0.0, (wind_mph**2 + 1)**0.5 * (1 - 2*eta + 1.5*eta**2 - 0.5*eta**3) / 0.3002)
    return {
        "temp_max": temp_max, "temp_min": temp_min,
        "humidity": humidity, "wind_speed": wind_speed,
        "precipitation": precip, "month": month,
        "month_sin": month_sin, "month_cos": month_cos,
        "day_of_year": month * 30, "weekend_flag": 0,
        "fire_season_flag": fire_season, "drought_index": drought,
        "temp_max_7d_rolling_mean": tmax_7d, "humidity_7d_rolling_mean": hum_7d,
        "temp_max_14d_rolling_mean": tmax_14d, "humidity_14d_rolling_mean": hum_14d,
        "temp_max_30d_rolling_mean": temp_max, "humidity_30d_rolling_mean": humidity,
        "temperature_anomaly": temp_max - tmax_14d,
        "vpd": vpd,
        "wind_speed_drought_interaction": wind_speed * drought,
        "temp_max_humidity_interaction": temp_max * humidity,
        "prev_day_fire": float(prev_day_fire),
        "prev2_day_fire": float(prev2_day_fire),
        "fire_7d_rolling": float(fire_7d),
        "elevation": float(elevation),
        "wind_dir_sin": math.sin(wind_dir_rad),
        "wind_dir_cos": math.cos(wind_dir_rad),
        "offshore_wind_flag": float((wind_dir <= 90) or (wind_dir >= 315)),
        "ffwi": ffwi,
        f"county_{county}": 1.0,
    }


def batch_predict_all_counties(model, all_feat_cols, counties_list,
                                month, temp_max, temp_min, humidity,
                                wind_speed, precip, drought,
                                wind_dir=180) -> pd.DataFrame:
    rows = []
    for county in counties_list:
        row = build_input_row(
            county, month, temp_max, temp_min, humidity,
            wind_speed, precip, drought,
            tmax_7d=temp_max, hum_7d=humidity,
            tmax_14d=temp_max, hum_14d=humidity,
            wind_dir=wind_dir,
        )
        rows.append(row)
    X = pd.DataFrame(rows).reindex(columns=all_feat_cols, fill_value=0).fillna(0).astype(float)
    probas = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"county": counties_list, "fire_prob": probas})


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🔥🌲")
    st.title("WildFire Prediction")
    st.caption("California — 58 Counties · 2016–2025")
    st.divider()
    page = st.radio("Navigate", ["📊 Model Performance", "🗺️ Prediction Tool", "📈 Regression Analysis"])

metrics = load_metrics()
if metrics is None:
    st.error("Models not trained yet. Run `python train_models.py` first.")
    st.stop()

clf_results = metrics["classification"]
best_name   = metrics["best_model"]
feat_cols   = metrics["feature_cols"]
counties    = metrics["counties"]

# Shared model name helpers — used by both Performance and Prediction pages
model_names      = list(clf_results.keys())
display_names    = [f"⭐ {n} (best)" if n == best_name else n for n in model_names]
display_to_name  = dict(zip(display_names, model_names))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Model Performance":
    st.markdown("## 📊 Model Performance Dashboard")
    st.caption(f"Best model: **⭐ {best_name}** · Evaluated on held-out test set (2024–2025)")

    selected_display = st.selectbox(
        "Select model to inspect",
        display_names,
        index=model_names.index(best_name),
    )
    selected = display_to_name[selected_display]
    split    = st.radio("Evaluate on", ["test", "val"], horizontal=True)
    m        = clf_results[selected][split]

    # ── Key Metrics Row ────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    def kpi(col, label, val, fmt=".4f"):
        col.markdown(f"""<div class='metric-card'><h3>{label}</h3><p>{val:{fmt}}</p></div>""", unsafe_allow_html=True)

    kpi(c1, "ROC-AUC",       m["roc_auc"])
    kpi(c2, "PR-AUC",        m["pr_auc"])
    kpi(c3, "F1 (fire)",     m["f1_fire"])
    kpi(c4, "Recall (fire)", m["recall_fire"])
    kpi(c5, "Precision",     m["precision_fire"])

    st.divider()

    # ── Comparison Table ───────────────────────────────────────────────────────
    st.markdown("### All Models — Comparison")
    rows = []
    for name, res in clf_results.items():
        t = res[split]
        label = f"⭐ {name}" if name == best_name else name
        rows.append({
            "Model":            label,
            "ROC-AUC":          t["roc_auc"],
            "PR-AUC":           t["pr_auc"],
            "F1 (fire)":        t["f1_fire"],
            "Recall (fire)":    t["recall_fire"],
            "Precision (fire)": t["precision_fire"],
            "Accuracy":         t["accuracy"],
            "Threshold":        t["threshold"],
        })
    df_cmp = pd.DataFrame(rows).set_index("Model")

    def highlight_best(s):
        is_best = s == s.max()
        return ["background-color: #1f4e1f; color: #86efac" if v else "" for v in is_best]

    def highlight_best_row(row):
        is_best_row = row.name.startswith("⭐")
        return ["border-left: 3px solid #f97316" if is_best_row else "" for _ in row]

    st.dataframe(
        df_cmp.style
            .apply(highlight_best)
            .apply(highlight_best_row, axis=1)
            .format("{:.4f}"),
        use_container_width=True
    )

    st.divider()

    # ── Curves ────────────────────────────────────────────────────────────────
    col_roc, col_pr = st.columns(2)

    with col_roc:
        st.markdown("### ROC Curves")
        fig_roc = go.Figure()
        fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                          line=dict(dash="dash", color="gray"))
        colors = px.colors.qualitative.Plotly
        for i, (name, res) in enumerate(clf_results.items()):
            rc = res[split]["roc_curve"]
            auc = res[split]["roc_auc"]
            fig_roc.add_trace(go.Scatter(
                x=rc["fpr"], y=rc["tpr"],
                name=f"{name} ({auc:.3f})",
                line=dict(color=colors[i % len(colors)], width=2.5 if name == selected else 1.5),
                opacity=1.0 if name == selected else 0.55,
            ))
        fig_roc.update_layout(
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            height=420, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_pr:
        st.markdown("### Precision-Recall Curves")
        fig_pr = go.Figure()
        baseline = sum(1 for v in clf_results[selected][split]["confusion_matrix"][1] if isinstance(v, (int,float))) / sum(
            sum(row) for row in clf_results[selected][split]["confusion_matrix"])
        fig_pr.add_shape(type="line", x0=0, y0=baseline, x1=1, y1=baseline,
                         line=dict(dash="dash", color="gray"))
        for i, (name, res) in enumerate(clf_results.items()):
            prc = res[split]["pr_curve"]
            auc = res[split]["pr_auc"]
            fig_pr.add_trace(go.Scatter(
                x=prc["recall"], y=prc["precision"],
                name=f"{name} ({auc:.3f})",
                line=dict(color=colors[i % len(colors)], width=2.5 if name == selected else 1.5),
                opacity=1.0 if name == selected else 0.55,
            ))
        fig_pr.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
            legend=dict(orientation="h", yanchor="bottom", y=-0.4),
            height=420, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
        )
        st.plotly_chart(fig_pr, use_container_width=True)

    # ── Confusion Matrix ───────────────────────────────────────────────────────
    st.markdown("### Confusion Matrix")
    cm = np.array(m["confusion_matrix"])
    cm_labels = ["No Fire", "Fire"]
    fig_cm = go.Figure(go.Heatmap(
        z=cm, x=cm_labels, y=cm_labels,
        colorscale="Oranges",
        text=[[f"{v:,}" for v in row] for row in cm],
        texttemplate="%{text}", textfont=dict(size=18),
        showscale=True,
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted", yaxis_title="Actual",
        height=360, width=420,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
    )
    col_cm, col_fi = st.columns([1, 2])
    col_cm.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature Importance ────────────────────────────────────────────────────
    with col_fi:
        st.markdown("### Feature Importance (Top 20)")
        fi_all = metrics.get("feature_importance", {})
        if selected in fi_all:
            fi = pd.Series(fi_all[selected]).sort_values(ascending=True)
            fig_fi = px.bar(
                fi, orientation="h",
                labels={"value": "Importance", "index": "Feature"},
                color=fi.values,
                color_continuous_scale="Oranges",
            )
            fig_fi.update_layout(
                showlegend=False, coloraxis_showscale=False,
                height=400, margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#f1f5f9",
            )
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance not available for this model.")

    # ── County Fire Risk Map ───────────────────────────────────────────────────
    st.divider()
    st.markdown("### 🗺️ Statewide County Fire Risk Map")
    st.caption("Predicted fire probability for all counties — adjust conditions to explore risk")

    geojson  = load_geojson()
    map_feat = load_feature_cols()

    # Model selector for map
    map_sel_display = st.selectbox(
        "Model for map",
        display_names,
        index=model_names.index(best_name),
        key="map_model_select",
    )
    map_model = load_model(display_to_name[map_sel_display])

    mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
    map_month    = mc1.slider("Month",          1, 12,   7, key="map_month")
    map_temp     = mc2.slider("Max Temp (°C)",  0, 50,  30, key="map_temp")
    map_hum      = mc3.slider("Humidity (%)",   0, 100, 30, key="map_hum")
    map_wind     = mc4.slider("Wind (m/s)",     0,  30,   5, key="map_wind")
    map_precip   = mc5.slider("Precip (mm)",    0,  50,   0, key="map_precip")
    map_wind_dir = mc6.slider("Wind Dir (°)",   0, 359, 180, key="map_wind_dir",
                              help="0=N  90=E  180=S  270=W  — NE (0-90°) = offshore/Santa Ana")

    risk_df = batch_predict_all_counties(
        map_model, map_feat, counties,
        map_month, float(map_temp), float(map_temp) - 10,
        float(map_hum), float(map_wind), float(map_precip),
        drought=1.0, wind_dir=float(map_wind_dir)
    )

    # Convert to percentage for display
    risk_df["fire_pct"] = risk_df["fire_prob"] * 100
    max_pct = max(risk_df["fire_pct"].max(), 0.5)   # floor so scale never collapses

    fig_map = px.choropleth(
        risk_df,
        geojson=geojson,
        locations="county",
        featureidkey="properties.county",
        color="fire_pct",
        color_continuous_scale=["#1a2e1a", "#eab308", "#f97316", "#ef4444"],
        range_color=[0, max_pct],
        labels={"fire_pct": "Fire Prob (%)"},
        hover_name="county",
        hover_data={"fire_pct": ":.2f"},
    )
    fig_map.update_geos(
        fitbounds="locations", visible=False,
        bgcolor="rgba(0,0,0,0)"
    )
    fig_map.update_layout(
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#f1f5f9",
        coloraxis_colorbar=dict(
            title="Fire Prob %",
            ticksuffix="%",
            tickformat=".1f",
            thickness=14,
        ),
        geo=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # Top 5 at-risk counties for current conditions
    top5 = risk_df.nlargest(5, "fire_prob")
    st.markdown("**Highest risk counties for these conditions:**")
    cols_top = st.columns(5)
    for i, (_, row) in enumerate(top5.iterrows()):
        name = row["county"].replace(" County", "")
        cols_top[i].metric(name, f"{row['fire_prob']:.1%}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Prediction Tool
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Prediction Tool":
    st.markdown("## 🗺️ Wildfire Risk Prediction")
    st.caption("Enter weather conditions to predict fire probability for any California county")

    all_feat_cols = load_feature_cols()

    # Model selector
    pred_display_names = [f"⭐ {n} (best)" if n == best_name else n for n in model_names]
    pred_display_to_name = dict(zip(pred_display_names, model_names))
    pred_selected_display = st.selectbox(
        "Model",
        pred_display_names,
        index=model_names.index(best_name),
        key="pred_model_select",
    )
    pred_model_name = pred_display_to_name[pred_selected_display]
    model = load_model(pred_model_name)
    pred_threshold = clf_results[pred_model_name]["test"]["threshold"]

    if model is None:
        st.error("Model file not found. Run `python train_models.py` first.")
        st.stop()

    with st.form("prediction_form"):
        st.markdown("### 📍 Location & Date")
        col1, col2 = st.columns(2)
        county   = col1.selectbox("County", sorted(counties))
        month    = col2.slider("Month", 1, 12, 7)

        st.markdown("### 🌡️ Weather Conditions")
        wc1, wc2, wc3, wc4 = st.columns(4)
        temp_max   = wc1.number_input("Max Temp (°C)",   value=30.0, min_value=-10.0, max_value=55.0)
        temp_min   = wc2.number_input("Min Temp (°C)",   value=15.0, min_value=-20.0, max_value=45.0)
        humidity   = wc3.number_input("Humidity (%)",    value=30.0, min_value=0.0,   max_value=100.0)
        wind_speed = wc4.number_input("Wind Speed (m/s)",value=5.0,  min_value=0.0,   max_value=40.0)

        wc5, wc6, wc7 = st.columns(3)
        precip     = wc5.number_input("Precipitation (mm)", value=0.0, min_value=0.0, max_value=200.0)
        drought    = wc6.slider("Drought Index (0=none, 4=extreme)", 0.0, 4.0, 1.0, step=0.5)
        wind_dir   = wc7.slider("Wind Direction (°)", 0, 359, 180,
                                help="0=N  90=E  180=S  270=W  — NE (0-90°) = offshore/Santa Ana")

        st.markdown("### 📅 Rolling Averages (last N days)")
        ra1, ra2, ra3, ra4 = st.columns(4)
        tmax_7d  = ra1.number_input("Temp Max 7d avg",  value=float(temp_max))
        hum_7d   = ra2.number_input("Humidity 7d avg",  value=float(humidity))
        tmax_14d = ra3.number_input("Temp Max 14d avg", value=float(temp_max))
        hum_14d  = ra4.number_input("Humidity 14d avg", value=float(humidity))

        st.markdown("### 🕐 Recent Fire History (lag features)")
        lc1, lc2, lc3 = st.columns(3)
        prev_day_fire  = lc1.selectbox("Fire yesterday?", [0, 1], format_func=lambda x: "Yes 🔥" if x else "No")
        prev2_day_fire = lc2.selectbox("Fire 2 days ago?", [0, 1], format_func=lambda x: "Yes 🔥" if x else "No")
        fire_7d        = lc3.number_input("Fires in last 7 days (count)", value=0, min_value=0, max_value=7)

        submitted = st.form_submit_button("🔍 Predict Fire Risk", use_container_width=True)

    if submitted:
        row = build_input_row(
            county, month, temp_max, temp_min, humidity, wind_speed,
            precip, drought, tmax_7d, hum_7d, tmax_14d, hum_14d,
            prev_day_fire, prev2_day_fire, fire_7d,
            wind_dir=wind_dir,
        )
        X = pd.DataFrame([row]).reindex(columns=all_feat_cols, fill_value=0).astype(float)
        vpd = row["vpd"]
        fire_season = row["fire_season_flag"]

        proba = model.predict_proba(X)[0][1]
        threshold = pred_threshold
        pred = proba >= threshold

        # Base fire rate derived from test set class distribution
        BASE_RATE = metrics.get("base_rate", 0.0718)
        relative_risk = proba / BASE_RATE

        # Risk bands calibrated to relative risk vs. base rate
        if proba >= threshold:
            risk_level, risk_color = "Extreme", "🔴"
        elif relative_risk >= 4:
            risk_level, risk_color = "High", "🟠"
        elif relative_risk >= 2:
            risk_level, risk_color = "Moderate", "🟡"
        else:
            risk_level, risk_color = "Low", "🟢"

        st.divider()
        col_res, col_gauge = st.columns([1, 1])

        with col_res:
            st.markdown("### 🎯 Prediction Result")
            if pred:
                st.markdown("<h2><span class='fire-badge'>🔥 FIRE RISK DETECTED</span></h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2><span class='safe-badge'>✅ LOW FIRE RISK</span></h2>", unsafe_allow_html=True)

            st.metric("Fire Probability", f"{proba:.1%}", help="Raw model output probability")
            st.metric("Relative Risk", f"{relative_risk:.1f}×",
                      help=f"Compared to California baseline ({BASE_RATE:.1%} of days have fires)")
            st.metric("Decision Threshold", f"{threshold:.0%}")
            star = "⭐ " if pred_model_name == best_name else ""
            st.metric("Model Used", f"{star}{pred_model_name}")
            st.markdown(f"**Risk Level:** {risk_color} {risk_level}")
            st.markdown(f"**County:** {county}")

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number=dict(suffix="%", font=dict(size=42, color="#f1f5f9")),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color="#f1f5f9"),
                              tickvals=[0, 20, 40, 60, 80, 100]),
                    bar=dict(color="#ef4444" if pred else ("#f97316" if risk_level == "High" else
                             "#eab308" if risk_level == "Moderate" else "#22c55e"), thickness=0.35),
                    steps=[
                        dict(range=[0,  BASE_RATE * 200], color="#1a2e1a"),
                        dict(range=[BASE_RATE * 200, BASE_RATE * 400], color="#2e2a1a"),
                        dict(range=[BASE_RATE * 400, threshold * 100], color="#2e1e1a"),
                        dict(range=[threshold * 100, 100], color="#2e1a1a"),
                    ],
                    threshold=dict(
                        line=dict(color="#f97316", width=4),
                        thickness=0.85, value=threshold * 100
                    ),
                ),
                title=dict(text="Fire Probability  (orange line = decision threshold)",
                           font=dict(color="#94a3b8", size=13))
            ))
            fig_gauge.update_layout(
                height=300, margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font_color="#f1f5f9"
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        # ── Risk factor summary ─────────────────────────────────────────────────
        st.divider()
        st.markdown("### ⚠️ Risk Factor Analysis")
        factors = {
            "High Temperature": temp_max > 35,
            "Low Humidity (<30%)": humidity < 30,
            "High Wind Speed (>10 m/s)": wind_speed > 10,
            "No Recent Rainfall": precip < 1,
            "Fire Season (Jun–Oct)": fire_season == 1,
            "Drought Conditions": drought >= 2,
            "VPD > 2 kPa": vpd > 2,
        }
        fc1, fc2 = st.columns(2)
        for i, (factor, active) in enumerate(factors.items()):
            col = fc1 if i % 2 == 0 else fc2
            icon = "🔴" if active else "🟢"
            col.markdown(f"{icon} {factor}")

        # ── SHAP Explanation ────────────────────────────────────────────────────
        st.divider()
        st.markdown("### 🔍 Why this prediction? (SHAP)")
        st.caption("How much each feature pushed the probability up or down from the baseline")

        with st.spinner("Computing SHAP values..."):
            explainer = load_shap_explainer(pred_model_name)

        if explainer is not None:
            try:
                shap_vals = explainer.shap_values(X)
                # For binary classifiers shap_values returns list [neg, pos]
                if isinstance(shap_vals, list):
                    sv = shap_vals[1][0]
                else:
                    sv = shap_vals[0]

                feat_names = list(all_feat_cols)
                shap_series = pd.Series(sv, index=feat_names)

                # Drop near-zero and one-hot county columns for readability
                shap_display = shap_series[~shap_series.index.str.startswith("county_")]
                shap_display = shap_display.reindex(
                    shap_display.abs().nlargest(15).index
                ).sort_values()

                colors = ["#ef4444" if v > 0 else "#22c55e" for v in shap_display.values]

                fig_shap = go.Figure(go.Bar(
                    x=shap_display.values,
                    y=shap_display.index,
                    orientation="h",
                    marker_color=colors,
                    text=[f"{v:+.4f}" for v in shap_display.values],
                    textposition="outside",
                ))
                fig_shap.add_vline(x=0, line_color="#94a3b8", line_width=1)
                fig_shap.update_layout(
                    xaxis_title="SHAP value (impact on fire probability)",
                    height=420,
                    margin=dict(l=0, r=60, t=10, b=0),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font_color="#f1f5f9",
                    xaxis=dict(gridcolor="#2d3748"),
                )
                st.plotly_chart(fig_shap, use_container_width=True)
                st.caption("🔴 Red = increases fire risk · 🟢 Green = decreases fire risk")

                # Expected value baseline
                if hasattr(explainer, "expected_value"):
                    ev = explainer.expected_value
                    base = ev[1] if isinstance(ev, (list, np.ndarray)) else float(ev)
                    st.caption(f"Baseline (average) fire probability: **{base:.1%}** · "
                               f"This prediction: **{proba:.1%}**")
            except Exception as e:
                st.info(f"SHAP unavailable for this model: {e}")
        else:
            st.info("SHAP explainer not available for the selected model.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Regression Analysis
# ══════════════════════════════════════════════════════════════════════════════
else:
    st.markdown("## 📈 Fire Intensity Regression (FRP)")
    st.caption("XGBoost model predicting Fire Radiative Power on confirmed fire days")

    reg = metrics["regression"]

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("RMSE", f"{reg['rmse']:,.2f} MW")
    r2.metric("MAE",  f"{reg['mae']:,.2f} MW")
    r3.metric("R² (raw)", f"{reg['r2']:.3f}")
    r4.metric("R² (log scale)", f"{reg.get('r2_log', 'N/A')}" if reg.get('r2_log') is None else f"{reg['r2_log']:.3f}",
              help="R² on log-transformed FRP — more meaningful for skewed fire intensity data")

    st.divider()

    actuals = reg["actuals"]
    preds   = reg["predictions"]

    # Log-transform both for visualisation — same space the model was trained in
    log_act  = [math.log1p(v) for v in actuals]
    log_pred = [math.log1p(max(v, 0)) for v in preds]

    col_scatter, col_fi = st.columns(2)

    with col_scatter:
        st.markdown("### Actual vs. Predicted FRP  *(log scale)*")
        st.caption("log(1 + FRP) on both axes — shows the full dynamic range without outliers dominating")

        la = np.array(log_act)
        lp = np.array(log_pred)
        min_log = min(la.min(), lp.min())
        max_log = max(la.max(), lp.max())

        # OLS regression line through the scatter (slope < 1 = prediction shrinkage)
        slope, intercept = np.polyfit(la, lp, 1)
        fit_x = np.linspace(min_log, max_log, 100)
        fit_y = slope * fit_x + intercept

        # Annotate a few raw-scale tick positions on both axes
        tick_mw   = [10, 100, 500, 2000, 8000]
        tick_log  = [math.log1p(v) for v in tick_mw]
        tick_text = [f"{v:,}" for v in tick_mw]

        fig_sp = go.Figure()
        fig_sp.add_trace(go.Scatter(
            x=la.tolist(), y=lp.tolist(), mode="markers",
            marker=dict(color="#f97316", opacity=0.45, size=5),
            name="Fire days",
        ))
        # Perfect fit — clipped to data range so it doesn't extend into empty space
        fig_sp.add_trace(go.Scatter(
            x=[min_log, max_log], y=[min_log, max_log],
            mode="lines", line=dict(dash="dash", color="#94a3b8", width=1.5),
            name="Perfect fit (y = x)",
        ))
        # Fitted trend line
        fig_sp.add_trace(go.Scatter(
            x=fit_x.tolist(), y=fit_y.tolist(),
            mode="lines", line=dict(color="#3b82f6", width=2),
            name=f"Fitted trend  (slope={slope:.2f})",
        ))
        fig_sp.update_layout(
            xaxis=dict(title="Actual FRP (MW, log scale)", tickvals=tick_log,
                       ticktext=tick_text, gridcolor="#2d3748",
                       range=[min_log * 0.97, max_log * 1.03]),
            yaxis=dict(title="Predicted FRP (MW, log scale)", tickvals=tick_log,
                       ticktext=tick_text, gridcolor="#2d3748",
                       range=[min_log * 0.97, max_log * 1.03]),
            height=420, margin=dict(l=0, r=0, t=30, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_sp, use_container_width=True)

    with col_fi:
        st.markdown("### Feature Importance (Top 20)")
        fi = pd.Series(reg["feature_importance"]).sort_values(ascending=True)
        fig_fi = px.bar(
            fi, orientation="h",
            labels={"value": "Importance", "index": "Feature"},
            color=fi.values,
            color_continuous_scale="Oranges",
        )
        fig_fi.update_layout(
            showlegend=False, coloraxis_showscale=False,
            height=420, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
        )
        st.plotly_chart(fig_fi, use_container_width=True)

    # ── Log-space residuals + FRP distribution ────────────────────────────────
    st.divider()
    col_res, col_dist = st.columns(2)

    with col_res:
        st.markdown("### Log-Space Residuals")
        st.caption("log(1+actual) − log(1+predicted) — symmetric when the model is well-calibrated")
        log_resid = [a - p for a, p in zip(log_act, log_pred)]
        fig_res = px.histogram(
            x=log_resid, nbins=60,
            labels={"x": "log(1+Actual) − log(1+Predicted)"},
            color_discrete_sequence=["#f97316"],
        )
        fig_res.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
        fig_res.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            xaxis=dict(gridcolor="#2d3748"),
            yaxis=dict(title="Count", gridcolor="#2d3748"),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    with col_dist:
        st.markdown("### FRP Distribution: Actual vs. Predicted")
        st.caption("Density on log scale — ideal overlap shows the model captures the shape of fire intensity")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=log_act, nbinsx=50, histnorm="probability density",
            name="Actual", marker_color="#ef4444", opacity=0.6,
        ))
        fig_dist.add_trace(go.Histogram(
            x=log_pred, nbinsx=50, histnorm="probability density",
            name="Predicted", marker_color="#3b82f6", opacity=0.6,
        ))
        fig_dist.update_layout(
            barmode="overlay",
            xaxis=dict(title="log(1 + FRP)", gridcolor="#2d3748",
                       tickvals=tick_log, ticktext=tick_text),
            yaxis=dict(title="Density", gridcolor="#2d3748"),
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#f1f5f9",
            legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        )
        st.plotly_chart(fig_dist, use_container_width=True)
