import json
from pathlib import Path
import zipfile
import io

import streamlit as st
import pandas as pd
import plotly.graph_objs as go

st.set_page_config(
    page_title="🔄 Replay Reports", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<div style="background: linear-gradient(90deg, #1f77b4, #ff7f0e); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; text-align: center;">
        🔄 Replay Previous Analysis Runs
    </h1>
    <p style="color: white; margin: 0; text-align: center; opacity: 0.9;">
        Browse and download historical anomaly detection results
    </p>
</div>
""", unsafe_allow_html=True)

reports_root = Path("data/reports")
if not reports_root.exists():
    st.info("No reports found yet. Run the main app to generate one.")
    st.stop()

runs = sorted([p for p in reports_root.iterdir() if p.is_dir()], reverse=True)
if not runs:
    st.info("No report directories available.")
    st.stop()

with st.container():
    st.subheader("📂 Available Analysis Reports")
    run = st.selectbox(
        "Select a run to replay", 
        options=runs, 
        format_func=lambda p: f"📊 {p.name}",
        help="Select an analysis run to view results and download data"
    )

st.markdown("---")
col_download, col_info = st.columns([2, 1])

with col_download:
    st.subheader("📥 Download Report")
    
    def create_report_zip(run_path):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in run_path.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(run_path)
                    zip_file.write(file_path, arcname)
        zip_buffer.seek(0)
        return zip_buffer.read()
    
    if st.button("📦 Prepare Download", type="primary", use_container_width=True):
        with st.spinner("Preparing report archive..."):
            zip_data = create_report_zip(run)
        
        st.download_button(
            label="📥 Download Complete Report",
            data=zip_data,
            file_name=f"replay_report_{run.name}.zip",
            mime="application/zip",
            help="Download all files from this analysis run",
            key=f"download_replay_{run.name}",
            use_container_width=True,
            type="secondary"
        )

with col_info:
    st.subheader("📋 Report Info")
    total_size = sum(f.stat().st_size for f in run.rglob('*') if f.is_file())
    file_count = len(list(run.rglob('*')))
    
    st.metric("Total Size", f"{total_size / 1024 / 1024:.1f} MB")
    st.metric("File Count", file_count)
    if (run / "screenshots").exists():
        screenshot_count = len(list((run / "screenshots").glob("*.png")))
        st.metric("Screenshots", screenshot_count)

st.markdown("---")

col_left, col_right = st.columns([1, 2])

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
    st.subheader("📊 Anomaly Score Timeline")
    thr = float(settings.get("threshold", 0.5))
    x = df_pf["frame_idx"].tolist()
    y = df_pf["score"].tolist()
    
    fig = go.Figure()
    colors = ['red' if score > thr else 'blue' for score in y]
    fig.add_trace(go.Scatter(
        x=x, 
        y=y, 
        mode="lines+markers", 
        name="Anomaly Score",
        line=dict(width=2),
        marker=dict(size=3),
        hovertemplate="Frame: %{x}<br>Score: %{y:.3f}<extra></extra>"
    ))
    fig.add_hline(
        y=thr, 
        line_dash="dash", 
        line_color="red", 
        line_width=2,
        annotation_text=f"Threshold ({thr})", 
        annotation_position="top left"
    )
    if not df_ev.empty:
        for _, ev in df_ev.iterrows():
            start_f = int(ev.get("start_frame", -1))
            end_f = int(ev.get("end_frame", -1))
            fig.add_vrect(
                x0=start_f, x1=end_f,
                fillcolor="red", opacity=0.2,
                layer="below", line_width=0,
                annotation_text=f"Event {start_f}-{end_f}",
                annotation_position="top"
            )
    
    fig.update_layout(
        height=400, 
        margin=dict(l=10, r=10, t=30, b=10), 
        yaxis=dict(range=[0, 1], title="Anomaly Score"),
        xaxis=dict(title="Frame Index"),
        hovermode='x unified',
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
    st.subheader("📈 Analysis Summary")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_score = max(y) if y else 0
        st.metric("Max Score", f"{max_score:.3f}")
    
    with col2:
        avg_score = sum(y) / len(y) if y else 0
        st.metric("Avg Score", f"{avg_score:.3f}")
    
    with col3:
        frames_above_threshold = sum(1 for score in y if score > thr)
        st.metric("Frames > Threshold", frames_above_threshold)

with col_left:
    st.subheader("🎯 Detected Events")
    
    if df_ev.empty:
        st.info("🔍 No anomaly events were detected in this run.")
    else:
        st.success(f"Found {len(df_ev)} anomaly event(s)")
        for i, (_, ev) in enumerate(df_ev.iterrows(), 1):
            start_f = int(ev.get("start_frame", -1))
            end_f = int(ev.get("end_frame", -1))
            duration = end_f - start_f + 1
            
            with st.expander(f"🚨 Event {i}: Frames {start_f} → {end_f} ({duration} frames)", expanded=True):
                col_img, col_details = st.columns([2, 1])
                
                with col_details:
                    st.write(f"**Start Frame:** {start_f}")
                    st.write(f"**End Frame:** {end_f}")
                    st.write(f"**Duration:** {duration} frames")
                    event_scores = [y[j] for j in range(len(y)) if start_f <= x[j] <= end_f]
                    if event_scores:
                        st.write(f"**Max Score:** {max(event_scores):.3f}")
                        st.write(f"**Avg Score:** {sum(event_scores)/len(event_scores):.3f}")
                
                with col_img:
                    if shots_dir.exists():
                        candidate = shots_dir / f"frame_{start_f:06d}.png"
                        if candidate.exists():
                            st.image(
                                str(candidate), 
                                caption=f"Event {i} - Frame {start_f}", 
                                use_container_width=True
                            )
                        else:
                            st.info("Screenshot not available")
                    else:
                        st.info("Screenshots directory not found")

st.markdown("---")
st.subheader("⚙️ Analysis Settings")

if settings:
    col_set1, col_set2, col_set3 = st.columns(3)
    
    with col_set1:
        st.write(f"**Threshold:** {settings.get('threshold', 'N/A')}")
        st.write(f"**Mode:** {settings.get('mode', 'N/A')}")
    
    with col_set2:
        st.write(f"**Smoothing:** {settings.get('smoothing', 'N/A')} frames")
        st.write(f"**FPS Cap:** {settings.get('fps_cap', 'N/A')}")
    
    with col_set3:
        st.write(f"**Draw IDs:** {settings.get('draw_ids', 'N/A')}")
        st.write(f"**Save Screenshots:** {settings.get('save_screens', 'N/A')}")
else:
    st.info("Settings information not available")

st.markdown("---")
st.subheader("📁 Report Files")

with st.expander("View all files in this report", expanded=False):
    for file_path in sorted(run.rglob('*')):
        if file_path.is_file():
            relative_path = file_path.relative_to(run)
            file_size = file_path.stat().st_size
            
            if file_size < 1024:
                size_str = f"{file_size} B"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} KB"
            else:
                size_str = f"{file_size / 1024 / 1024:.1f} MB"
            
            st.write(f"📄 `{relative_path}` ({size_str})")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <small>🔄 Replay functionality allows you to review past analysis without reprocessing</small>
</div>
""", unsafe_allow_html=True)