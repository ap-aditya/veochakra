import sys, os
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import time
import shutil
from pathlib import Path
from collections import deque
from datetime import datetime
import json

import streamlit as st
import numpy as np
import cv2
import plotly.graph_objs as go

from inference.pipeline import ScorerAdapter
from inference.cache_io import read_cache, write_cache, cache_path_for
from inference.events import EventConfig, EventState, update_events, finalize_report
from inference.draw import draw_overlays

st.set_page_config(page_title="AI Surveillance Anomaly Dashboard", layout="wide")

def auto_cleanup_old_files():
    try:
        uploads_dir = Path("data/uploads")
        if uploads_dir.exists():
            shutil.rmtree(uploads_dir)
            uploads_dir.mkdir(parents=True, exist_ok=True)
        
        reports_dir = Path("data/reports")
        if reports_dir.exists():
            shutil.rmtree(reports_dir)
            reports_dir.mkdir(parents=True, exist_ok=True)
        
        cache_dir = Path("artifacts/cache_dets")
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.jsonl"):
                if "_" in cache_file.stem and cache_file.stem.split("_")[-1].isdigit():
                    cache_file.unlink()
        return True
    except Exception as e:
        st.error(f"Auto-cleanup error: {e}")
        return False

@st.cache_resource
def load_scorer(art_dir: Path):
    return ScorerAdapter(art_dir)

def get_directory_size(path):
    total = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 * 1024)
    except:
        return 0

def save_uploaded_file(uploaded_file) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    uploads_dir = Path("data/uploads") / ts
    uploads_dir.mkdir(parents=True, exist_ok=True)
    out_path = uploads_dir / uploaded_file.name
    with open(out_path, "wb") as f:
        f.write(uploaded_file.read())
    return out_path

def init_run_dirs(video_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("data/reports") / f"{video_path.stem}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def draw_score_chart(scores: list, thr: float, placeholder):
    if not scores:
        placeholder.empty()
        return
    x = list(range(len(scores)))
    y = scores
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Smoothed score"))
    fig.add_hline(y=thr, line_dash="dash", line_color="red", annotation_text="Threshold", annotation_position="top left")
    fig.update_layout(height=300, margin=dict(l=10, r=10, t=20, b=10), yaxis=dict(range=[0, 1]))
    placeholder.plotly_chart(fig, use_container_width=True, theme="streamlit")

def render_events_list(events: list, placeholder):
    placeholder.markdown("### Events")
    if not events:
        placeholder.info("No events yet.")
        return
    last = events[-5:]
    lines = []
    for ev in last:
        lines.append(f"- Start frame: {ev.get('start_frame','?')}, End frame: {ev.get('end_frame','?')}")
    placeholder.markdown("\n".join(lines))

def cleanup_cache():
    cache_dir = Path("artifacts/cache_dets")
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        st.success("🗑️ Detection cache cleared successfully!")
    else:
        st.info("No cache directory found.")

def cleanup_uploads():
    uploads_dir = Path("data/uploads")
    if uploads_dir.exists():
        shutil.rmtree(uploads_dir)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        st.success("📁 Uploaded videos cleared successfully!")
    else:
        st.info("No uploads directory found.")

def cleanup_reports():
    reports_dir = Path("data/reports")
    if reports_dir.exists():
        shutil.rmtree(reports_dir)
        reports_dir.mkdir(parents=True, exist_ok=True)
        st.success("📊 Reports cleared successfully!")
    else:
        st.info("No reports directory found.")

def setup_sidebar():
    st.sidebar.header("Configuration")
    artifacts_dir = Path(st.sidebar.text_input("Artifacts directory", value="artifacts"))
    
    if not (artifacts_dir / "meta.json").exists():
        st.error("meta.json not found in the artifacts directory. Please copy training artifacts first.")
        st.stop()
    
    scorer = load_scorer(artifacts_dir)
    threshold_default = float(scorer.threshold)
    
    mode = st.sidebar.selectbox("Mode", ["Replay (use cached detections)", "Live detection (compute now)"])
    threshold = st.sidebar.slider("Threshold", 0.00, 1.00, float(threshold_default), 0.01)
    smooth_k = st.sidebar.slider("Smoothing window (frames)", 1, 21, 9, 2)
    fps_cap = st.sidebar.slider("Processing FPS cap", 4, 20, 12, 1)
    draw_ids = st.sidebar.checkbox("Draw IDs", True, help="Overlay track IDs with bounding boxes")
    save_screens = st.sidebar.checkbox("Save screenshots on events", True)
    zip_screens = st.sidebar.checkbox("Zip screenshots in report", True)
    
    uploaded = st.sidebar.file_uploader("Upload a video (mp4/avi)", type=["mp4", "avi"])
    start_btn = st.sidebar.button("Start")
    stop_btn = st.sidebar.button("Stop")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧹 Cleanup Options")
    auto_cleanup = st.sidebar.checkbox("Auto-cleanup on app restart", True, help="Automatically delete uploaded videos and cache when app restarts")
    clear_cache_btn = st.sidebar.button("🗑️ Clear Detection Cache", help="Delete all cached detections")
    clear_uploads_btn = st.sidebar.button("📁 Clear Uploaded Videos", help="Delete all uploaded video files")
    clear_reports_btn = st.sidebar.button("📊 Clear Old Reports", help="Delete all generated reports")
    
    cache_size = get_directory_size("artifacts/cache_dets") if Path("artifacts/cache_dets").exists() else 0
    uploads_size = get_directory_size("data/uploads") if Path("data/uploads").exists() else 0
    reports_size = get_directory_size("data/reports") if Path("data/reports").exists() else 0
    
    if cache_size > 0 or uploads_size > 0 or reports_size > 0:
        st.sidebar.markdown("**💾 Storage Usage:**")
        if cache_size > 0:
            st.sidebar.text(f"Cache: {cache_size:.1f} MB")
        if uploads_size > 0:
            st.sidebar.text(f"Uploads: {uploads_size:.1f} MB")
        if reports_size > 0:
            st.sidebar.text(f"Reports: {reports_size:.1f} MB")
    
    return {
        'artifacts_dir': artifacts_dir, 'scorer': scorer, 'mode': mode, 'threshold': threshold,
        'smooth_k': smooth_k, 'fps_cap': fps_cap, 'draw_ids': draw_ids, 'save_screens': save_screens,
        'zip_screens': zip_screens, 'uploaded': uploaded, 'start_btn': start_btn, 'stop_btn': stop_btn,
        'auto_cleanup': auto_cleanup, 'clear_cache_btn': clear_cache_btn, 'clear_uploads_btn': clear_uploads_btn,
        'clear_reports_btn': clear_reports_btn
    }

def initialize_session_state(mode, auto_cleanup):
    if "running" not in st.session_state:
        st.session_state.running = False
    if "per_frame_rows" not in st.session_state:
        st.session_state.per_frame_rows = []
    if "scores" not in st.session_state:
        st.session_state.scores = []
    if "events_state" not in st.session_state:
        st.session_state.events_state = EventState()
    if "video_path" not in st.session_state:
        st.session_state.video_path = None
    if "run_dir" not in st.session_state:
        st.session_state.run_dir = None
    if "use_cache" not in st.session_state:
        st.session_state.use_cache = (mode.startswith("Replay"))
    if "cleanup_done" not in st.session_state:
        st.session_state.cleanup_done = False
        if auto_cleanup:
            if auto_cleanup_old_files():
                st.session_state.cleanup_done = True

def handle_cleanup_buttons(clear_cache_btn, clear_uploads_btn, clear_reports_btn):
    if clear_cache_btn:
        try:
            cleanup_cache()
        except Exception as e:
            st.error(f"Error clearing cache: {e}")
    
    if clear_uploads_btn:
        try:
            cleanup_uploads()
        except Exception as e:
            st.error(f"Error clearing uploads: {e}")
    
    if clear_reports_btn:
        try:
            cleanup_reports()
        except Exception as e:
            st.error(f"Error clearing reports: {e}")

def handle_buttons(config):
    if config['start_btn']:
        if config['uploaded'] is None and st.session_state.video_path is None:
            st.warning("Please upload a video first.")
        else:
            if config['uploaded'] is not None:
                st.session_state.video_path = save_uploaded_file(config['uploaded'])
            st.session_state.run_dir = init_run_dirs(st.session_state.video_path)
            st.session_state.per_frame_rows = []
            st.session_state.scores = []
            st.session_state.events_state = EventState()
            st.session_state.running = True
            st.session_state.use_cache = config['mode'].startswith("Replay")

    if config['stop_btn']:
        st.session_state.running = False

def process_video_frame(frame, frame_idx, scorer, cached_dets, config, smoothed_deque, cfg, per_frame_dets_to_write, cached, placeholder_clear_counter):
    display_frame = frame
    
    tracked, features, score, dets = scorer.score_frame(display_frame, cached_dets=cached_dets)
    
    if cached is None:
        per_frame_dets_to_write.append([
            {"box": [d[0], d[1], d[2], d[3]], "conf": d[4], "cls": d[5]} for d in dets
        ])
    
    row = {"frame_idx": int(frame_idx), "score": float(score)}
    row.update({k: float(features[k]) for k in features.keys()})
    st.session_state.per_frame_rows.append(row)
    
    smoothed_deque.append(score)
    smoothed = float(np.mean(smoothed_deque)) if len(smoothed_deque) > 0 else float(score)
    
    update_events(
        st.session_state.events_state,
        cfg,
        frame_idx,
        float(smoothed),
        display_frame,
        tracked,
        st.session_state.run_dir,
        features=features,
    )
    
    return tracked, smoothed, placeholder_clear_counter + 1

def display_frame(display_frame, tracked, frame_idx, config, placeholder_clear_counter, video_placeholder):
    overlaid = draw_overlays(display_frame, tracked, draw_ids=bool(config['draw_ids']))
    overlaid_rgb = cv2.cvtColor(overlaid, cv2.COLOR_BGR2RGB)
    
    if placeholder_clear_counter % 30 == 0:
        video_placeholder.empty()
    
    try:
        video_placeholder.image(overlaid_rgb, caption=f"Frame {frame_idx}", use_container_width=True)
    except Exception as e:
        video_placeholder.empty()
        video_placeholder.image(overlaid_rgb, caption=f"Frame {frame_idx}", use_container_width=True)

def generate_final_report(config):
    if st.session_state.run_dir and st.session_state.per_frame_rows:
        settings = {
            "threshold": float(config['threshold']),
            "smooth_k": int(config['smooth_k']),
            "fps_cap": int(config['fps_cap']),
            "mode": config['mode'],
            "artifacts_dir": str(config['artifacts_dir']),
        }
        finalize_report(st.session_state.run_dir, st.session_state.per_frame_rows, st.session_state.events_state.events, settings, zip_images=config['zip_screens'])
        
        st.success(f"✅ Processing completed! Report auto-generated at: {st.session_state.run_dir.resolve()}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Frames", len(st.session_state.per_frame_rows))
        with col2:
            st.metric("Events Detected", len(st.session_state.events_state.events))
        with col3:
            avg_score = sum(row["score"] for row in st.session_state.per_frame_rows) / len(st.session_state.per_frame_rows) if st.session_state.per_frame_rows else 0
            st.metric("Avg Anomaly Score", f"{avg_score:.3f}")
        
        display_results_and_downloads(config)

def display_results_and_downloads(config):
    if st.session_state.events_state.events:
        st.subheader("📋 Detected Events Summary")
        for i, event in enumerate(st.session_state.events_state.events, 1):
            duration = event.get('end_frame', event.get('start_frame', 0)) - event.get('start_frame', 0) + 1
            st.write(f"**Event {i}:** Frames {event.get('start_frame', 'N/A')} - {event.get('end_frame', 'N/A')} (Duration: {duration} frames)")
    
    screenshots_dir = st.session_state.run_dir / "screenshots"
    
    if screenshots_dir.exists() and any(screenshots_dir.glob("*.png")):
        st.subheader("📸 Event Screenshots")
        
        screenshot_files = list(screenshots_dir.glob("*.png"))
        if screenshot_files:
            st.write(f"Found {len(screenshot_files)} screenshot(s):")
            
            if len(screenshot_files) <= 3:
                cols = st.columns(len(screenshot_files))
                for i, screenshot_file in enumerate(screenshot_files):
                    with cols[i]:
                        st.image(str(screenshot_file), caption=screenshot_file.name, use_container_width=True)
            else:
                for screenshot_file in screenshot_files:
                    st.image(str(screenshot_file), caption=screenshot_file.name, use_container_width=True)
    elif config['save_screens']:
        st.info("🔍 No anomaly screenshots were captured during this session.")
    
    st.write("---")
    st.subheader("📦 Download Complete Report")
    
    report_files = []
    per_frame_csv = st.session_state.run_dir / "per_frame.csv"
    events_csv = st.session_state.run_dir / "events.csv"
    screenshots_zip = st.session_state.run_dir / "screenshots.zip"
    settings_json = st.session_state.run_dir / "settings.json"
    
    if per_frame_csv.exists():
        report_files.append("� Per-frame anomaly scores (CSV)")
    if events_csv.exists():
        report_files.append("🎯 Detected events data (CSV)")
    if screenshots_zip.exists():
        report_files.append("📸 Event screenshots (ZIP)")
    if settings_json.exists():
        report_files.append("⚙️ Analysis settings (JSON)")
    
    if report_files:
        st.write("**This report includes:**")
        for file_desc in report_files:
            st.write(f"• {file_desc}")
        
        import zipfile
        import io
        
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            if per_frame_csv.exists():
                zip_file.write(per_frame_csv, f"per_frame_{st.session_state.run_dir.name}.csv")
            if events_csv.exists():
                zip_file.write(events_csv, f"events_{st.session_state.run_dir.name}.csv")
            if screenshots_zip.exists():
                zip_file.write(screenshots_zip, f"screenshots_{st.session_state.run_dir.name}.zip")
            if settings_json.exists():
                zip_file.write(settings_json, f"settings_{st.session_state.run_dir.name}.json")
        
        zip_buffer.seek(0)
        
        st.download_button(
            label="� Download Complete Analysis Report",
            data=zip_buffer.read(),
            file_name=f"anomaly_analysis_report_{st.session_state.run_dir.name}.zip",
            mime="application/zip",
            help="Download all analysis data, settings, and screenshots in one ZIP file"
        )
    else:
        st.info("No report files available for download.")

def main():
    config = setup_sidebar()
    initialize_session_state(config['mode'], config['auto_cleanup'])
    
    col_left, col_right = st.columns(2)
    video_placeholder = col_left.empty()
    chart_placeholder = col_right.empty()
    events_placeholder = col_right.empty()
    status_placeholder = st.empty()
    
    handle_buttons(config)
    handle_cleanup_buttons(config['clear_cache_btn'], config['clear_uploads_btn'], config['clear_reports_btn'])
    
    if st.session_state.running:
        vp = st.session_state.video_path
        if vp is None:
            st.stop()

        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            st.error("Failed to open the uploaded video.")
            st.session_state.running = False
        else:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            config['scorer'].start_stream(fps=fps, w=w, h=h)

            cached = None
            cache_path = cache_path_for(vp)
            if st.session_state.use_cache:
                cached = read_cache(vp)

            per_frame_dets_to_write = []
            smoothed_deque = deque(maxlen=max(3, int(config['smooth_k'])))
            smoothed_scores = []

            cfg = EventConfig(threshold=float(config['threshold']), save_screenshots=bool(config['save_screens']))

            frame_idx = 0
            last_time = time.time()
            interval = 1.0 / float(config['fps_cap'])
            placeholder_clear_counter = 0

            while st.session_state.running:
                ret, frame = cap.read()
                if not ret:
                    break

                cached_dets = None
                if cached is not None and frame_idx < len(cached):
                    cached_dets = cached[frame_idx]

                tracked, smoothed, placeholder_clear_counter = process_video_frame(
                    frame, frame_idx, config['scorer'], cached_dets, config, 
                    smoothed_deque, cfg, per_frame_dets_to_write, cached, placeholder_clear_counter
                )

                smoothed_scores.append(smoothed)
                display_frame(frame, tracked, frame_idx, config, placeholder_clear_counter, video_placeholder)
                draw_score_chart(smoothed_scores, float(config['threshold']), chart_placeholder)
                render_events_list(st.session_state.events_state.events, events_placeholder)

                now = time.time()
                dt = now - last_time
                if dt < interval:
                    time.sleep(interval - dt)
                last_time = time.time()

                frame_idx += 1

            cap.release()
            st.session_state.running = False

            if cached is None and per_frame_dets_to_write:
                write_cache(vp, per_frame_dets_to_write)
                status_placeholder.info(f"Detections cached to {cache_path.resolve()}")

            generate_final_report(config)

    else:
        st.markdown("### Upload a video and click Start to process.")
        if config['artifacts_dir'] and (config['artifacts_dir'] / "meta.json").exists():
            st.caption("Artifacts loaded. Default threshold applied. Use Replay mode for instant demos if caches exist.")

if __name__ == "__main__":
    main()
