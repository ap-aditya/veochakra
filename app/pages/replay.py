import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(page_title="Replay Reports", layout="wide")

st.title("Replay Previous Runs")

reports_root = Path("data/reports")
if not reports_root.exists():
    st.info("No reports found yet. Run the main app to generate one.")
    st.stop()

runs = sorted([p for p in reports_root.iterdir() if p.is_dir()])
if not runs:
    st.info("No report directories available.")
    st.stop()

run = st.selectbox("Select a run to replay", options=runs, format_func=lambda p: p.name)

col_left, col_right = st.columns()[1][2]

pf_csv = run / "per_frame.csv"
ev_csv = run / "events.csv"
shots_dir = run / "screenshots"
settings_json = run / "settings.json"

if not pf_csv.exists():
    st.error("per_frame.csv not found in selected run.")
    st.stop()

df_pf = pd.read_csv(pf_csv)
df_ev = pd.read_csv(ev_csv) if ev_csv.exists() else pd.DataFrame(columns=["start_frame","end_frame"])
settings = {}
if settings_json.exists():
    try:
        settings = json.loads(settings_json.read_text())
    except Exception:
        pass

with col_right:
    st.subheader("Score Timeline")
    thr = float(settings.get("threshold", 0.5))
    x = df_pf["frame_idx"].tolist()
    y = df_pf["score"].tolist()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Score"))
    fig.add_hline(y=thr, line_dash="dash", line_color="red", annotation_text="Threshold", annotation_position="top left")
    fig.update_layout(height=350, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 1]))
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")

with col_left:
    st.subheader("Events")
    if df_ev.empty:
        st.info("No events recorded.")
    else:
        for _, ev in df_ev.iterrows():
            start_f = int(ev.get("start_frame", -1))
            end_f = int(ev.get("end_frame", -1))
            st.markdown(f"- Frames {start_f} → {end_f}")
            if shots_dir.exists():
                candidate = shots_dir / f"frame_{start_f:06d}.png"
                if candidate.exists():
                    st.image(str(candidate), caption=f"Start frame {start_f}", use_container_width=True)