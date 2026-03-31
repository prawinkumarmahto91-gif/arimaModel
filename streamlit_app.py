import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ARIMA Forecast Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Dark sidebar */
[data-testid="stSidebar"] {
    background: #0f1117;
    border-right: 1px solid #1e2130;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Main background */
.stApp {
    background: #0a0c12;
    color: #e2e8f0;
}

/* Metric cards */
div[data-testid="metric-container"] {
    background: #131722;
    border: 1px solid #1e2b45;
    border-radius: 12px;
    padding: 16px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-family: 'Space Mono', monospace !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #38bdf8 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 1.6rem !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 12px !important;
}

/* Slider */
[data-testid="stSlider"] > div > div > div {
    background: #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    color: white;
    border: none;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
    letter-spacing: 0.05em;
    padding: 10px 24px;
    transition: all 0.2s ease;
    width: 100%;
}
.stButton > button:hover {
    opacity: 0.88;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(14,165,233,0.35);
}

/* Section headers */
h1 {
    font-family: 'Space Mono', monospace !important;
    color: #f1f5f9 !important;
    letter-spacing: -0.02em;
}
h2, h3 {
    font-family: 'DM Sans', sans-serif !important;
    color: #cbd5e1 !important;
}

/* Info boxes */
.info-box {
    background: #131722;
    border: 1px solid #1e2b45;
    border-left: 3px solid #38bdf8;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 13px;
    color: #94a3b8;
    font-family: 'Space Mono', monospace;
    margin: 8px 0;
}

/* Divider */
hr { border-color: #1e2130 !important; }

/* Table */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("arimaModel.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ── Helper: get model info ────────────────────────────────────────────────────
def get_model_info(model):
    info = {}
    try:
        if hasattr(model, "model"):
            inner = model.model
            if hasattr(inner, "order"):
                info["order"] = inner.order
            if hasattr(inner, "seasonal_order"):
                info["seasonal_order"] = inner.seasonal_order
            if hasattr(inner, "endog_names"):
                info["target"] = str(inner.endog_names)
        if hasattr(model, "aic"):
            info["aic"] = round(float(model.aic), 4)
        if hasattr(model, "bic"):
            info["bic"] = round(float(model.bic), 4)
        if hasattr(model, "params"):
            info["n_params"] = len(model.params)
    except:
        pass
    return info


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.markdown("# 📈 ARIMA Forecast")
    st.markdown("<p style='color:#64748b; font-size:14px; margin-top:-12px;'>Time Series Forecasting Dashboard</p>", unsafe_allow_html=True)
with col_status:
    st.markdown("<br>", unsafe_allow_html=True)
    if model_loaded:
        st.success("✅ Model Ready", icon=None)
    else:
        st.error("❌ Model Error")

st.markdown("---")

if not model_loaded:
    st.error(f"Failed to load model: {load_error}")
    st.stop()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Forecast Settings")
    st.markdown("---")

    mode = st.radio(
        "Mode",
        ["🔮 Future Forecast", "📊 In-Sample Predict"],
        index=0
    )

    st.markdown("---")

    if mode == "🔮 Future Forecast":
        steps = st.slider("Forecast Steps", min_value=1, max_value=200, value=30)

        use_dates = st.checkbox("Show as Dates", value=True)
        if use_dates:
            start_date = st.date_input("Series End Date", value=datetime.today())
            freq = st.selectbox("Frequency", ["Daily", "Weekly", "Monthly", "Yearly"])
            freq_map = {"Daily": "D", "Weekly": "W", "Monthly": "MS", "Yearly": "YS"}

    else:
        in_start = st.number_input("Start Index", min_value=0, value=0, step=1)
        in_end   = st.number_input("End Index",   min_value=1, value=50, step=1)

    st.markdown("---")
    show_ci = st.checkbox("Show Confidence Interval", value=True)
    ci_alpha = st.slider("CI Level", min_value=0.05, max_value=0.5, value=0.05, step=0.05,
                          help="0.05 = 95% CI, 0.10 = 90% CI")

    st.markdown("---")
    run_btn = st.button("▶  Run Forecast")


# ── Model info cards ──────────────────────────────────────────────────────────
info = get_model_info(model)

c1, c2, c3, c4 = st.columns(4)
order = info.get("order", ("?", "?", "?"))
with c1:
    st.metric("AR (p)", order[0] if len(order) > 0 else "?")
with c2:
    st.metric("Integration (d)", order[1] if len(order) > 1 else "?")
with c3:
    st.metric("MA (q)", order[2] if len(order) > 2 else "?")
with c4:
    aic = info.get("aic", "N/A")
    st.metric("AIC", aic)

st.markdown("---")


# ── Run forecast ──────────────────────────────────────────────────────────────
if run_btn:
    with st.spinner("Running forecast..."):
        try:
            if mode == "🔮 Future Forecast":
                forecast_vals = model.forecast(steps=steps)

                # Confidence intervals
                if show_ci:
                    try:
                        pred_summary = model.get_forecast(steps=steps).summary_frame(alpha=ci_alpha)
                        lower = pred_summary["mean_ci_lower"].values
                        upper = pred_summary["mean_ci_upper"].values
                    except:
                        lower = upper = None
                else:
                    lower = upper = None

                # X axis
                if use_dates:
                    freq_str = freq_map[freq]
                    x_vals = pd.date_range(
                        start=pd.Timestamp(start_date) + timedelta(days=1),
                        periods=steps, freq=freq_str
                    ).strftime("%Y-%m-%d").tolist()
                else:
                    x_vals = list(range(1, steps + 1))

                y_vals = forecast_vals.tolist() if hasattr(forecast_vals, "tolist") else list(forecast_vals)
                x_label = "Date" if use_dates else "Step"
                chart_title = f"Forecast — Next {steps} Steps"
                df_label = "Forecast Value"

            else:
                pred_vals = model.predict(start=int(in_start), end=int(in_end))
                x_vals = list(range(int(in_start), int(in_end) + 1))
                y_vals = pred_vals.tolist() if hasattr(pred_vals, "tolist") else list(pred_vals)
                lower = upper = None
                x_label = "Index"
                chart_title = f"In-Sample Predictions  [{int(in_start)} → {int(in_end)}]"
                df_label = "Predicted Value"

            # ── Plot ──────────────────────────────────────────────────────────
            fig = go.Figure()

            # CI band
            if lower is not None and upper is not None:
                fig.add_trace(go.Scatter(
                    x=x_vals + x_vals[::-1],
                    y=list(upper) + list(lower[::-1]),
                    fill="toself",
                    fillcolor="rgba(56,189,248,0.10)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    name=f"{int((1-ci_alpha)*100)}% CI"
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals, y=list(upper),
                    mode="lines",
                    line=dict(color="rgba(56,189,248,0.35)", width=1, dash="dot"),
                    name="Upper CI", showlegend=False
                ))
                fig.add_trace(go.Scatter(
                    x=x_vals, y=list(lower),
                    mode="lines",
                    line=dict(color="rgba(56,189,248,0.35)", width=1, dash="dot"),
                    name="Lower CI", showlegend=False
                ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode="lines+markers",
                name="Forecast",
                line=dict(color="#38bdf8", width=2.5),
                marker=dict(size=5, color="#38bdf8",
                            line=dict(color="#0ea5e9", width=1)),
            ))

            fig.update_layout(
                title=dict(text=chart_title, font=dict(family="Space Mono", size=15, color="#cbd5e1")),
                plot_bgcolor="#0f1117",
                paper_bgcolor="#0f1117",
                font=dict(family="DM Sans", color="#94a3b8"),
                xaxis=dict(
                    title=x_label,
                    gridcolor="#1e2130",
                    linecolor="#1e2130",
                    tickfont=dict(size=11)
                ),
                yaxis=dict(
                    title="Value",
                    gridcolor="#1e2130",
                    linecolor="#1e2130",
                    tickfont=dict(size=11)
                ),
                legend=dict(
                    bgcolor="#131722",
                    bordercolor="#1e2b45",
                    borderwidth=1,
                    font=dict(size=12)
                ),
                hovermode="x unified",
                height=440,
                margin=dict(l=20, r=20, t=50, b=20),
            )

            st.plotly_chart(fig, use_container_width=True)

            # ── Stats row ─────────────────────────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            with s1:
                st.metric("Min", f"{min(y_vals):.4f}")
            with s2:
                st.metric("Max", f"{max(y_vals):.4f}")
            with s3:
                st.metric("Mean", f"{np.mean(y_vals):.4f}")
            with s4:
                st.metric("Std Dev", f"{np.std(y_vals):.4f}")

            st.markdown("---")

            # ── Data table ────────────────────────────────────────────────────
            with st.expander("📋 View Raw Data", expanded=False):
                df_out = pd.DataFrame({x_label: x_vals, df_label: y_vals})
                if lower is not None:
                    df_out["Lower CI"] = lower
                    df_out["Upper CI"] = upper
                st.dataframe(df_out, use_container_width=True)

                csv = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download CSV",
                    data=csv,
                    file_name="arima_forecast.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error(f"Forecast error: {e}")

else:
    # Placeholder
    st.markdown("""
    <div style='text-align:center; padding: 80px 0; color:#2d3748;'>
        <div style='font-size:56px;'>📈</div>
        <p style='font-family:Space Mono,monospace; font-size:14px; letter-spacing:0.1em; margin-top:16px;'>
            CONFIGURE SETTINGS AND CLICK RUN FORECAST
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#2d3748; font-size:12px; font-family:Space Mono,monospace;'>"
    "ARIMA FORECAST DASHBOARD · Powered by statsmodels & Streamlit"
    "</p>",
    unsafe_allow_html=True
)