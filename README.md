
# 🎯 AI Surveillance Anomaly Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Object%20Detection-green)](https://github.com/ultralytics/ultralytics)
[![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Scoring-orange)](https://lightgbm.readthedocs.io)

A comprehensive real-time anomaly detection system for video surveillance using state-of-the-art machine learning models. The dashboard combines **YOLOv8** object detection, **DeepSORT** tracking, and **LightGBM** classification to identify anomalous behavior in video streams with an intuitive web interface.

## 🌟 Features

### 🔍 **Advanced Detection Pipeline**
- **YOLOv8 Object Detection**: High-accuracy object detection with customizable classes
- **DeepSORT Tracking**: Persistent object tracking with unique IDs across frames
- **LightGBM Scoring**: Machine learning-based anomaly classification
- **Real-time Processing**: Configurable FPS processing for optimal performance

### 📊 **Interactive Dashboard**
- **Live Video Display**: Real-time video playback with detection overlays
- **Anomaly Score Visualization**: Dynamic plotting with threshold indicators
- **Event Detection**: Automatic anomaly event identification with hysteresis
- **Screenshot Capture**: Automatic screenshot saving during anomaly events

### 💾 **Intelligent Caching System**
- **Replay Mode**: Instant replay using cached detections (no reprocessing)
- **Live Detection Mode**: Real-time analysis with automatic cache generation
- **Cache Management**: Built-in tools for cleaning and managing detection caches

### 📈 **Comprehensive Reporting**
- **Per-frame Analysis**: Detailed CSV with scores and engineered features
- **Event Summaries**: Complete event logs with timestamps and durations
- **Screenshot Archives**: Organized image collections of detected anomalies
- **Complete Report Downloads**: Single-click download of all analysis data

### 🔄 **Multi-page Interface**
- **Main Dashboard**: Primary analysis and processing interface
- **Replay Page**: Browse and review historical analysis results
- **Auto-cleanup**: Configurable automatic cleanup of old files and caches

## � **Docker Deployment**

The application is fully dockerized for easy deployment and consistent environments.

### **Quick Start with Docker**

```bash
# 1. Build and start the application
docker compose up -d

# 2. Access the dashboard
# URL: http://localhost:8501

# 3. View logs
docker compose logs -f

# 4. Stop the application
docker compose down
```

### **Docker Requirements**
- **Docker & Docker Compose**: Must be installed
- **Port 8501**: Must be available on host
- **Model Files**: Place in `artifacts/` directory:
  - `artifacts/lgbm_frame_scorer.joblib`
  - `artifacts/meta.json`

### **Persistent Data**
- **Uploads**: `./data/uploads` (mounted as volume)
- **Reports**: `./data/reports` (mounted as volume) 
- **Artifacts**: `./artifacts` (mounted as volume)

---

## �🚀 Quick Start

### Prerequisites
- **Python 3.11**
- **uv** (package and environment manager by Astral)
- **Trained Model Artifacts** (see [Model Setup](#-model-setup))

### Installation

```bash
# 1. Clone or create project directory
mkdir anomaly-dashboard && cd anomaly-dashboard

# 2. Initialize project
uv init

# 3. Create directory structure
mkdir -p app inference artifacts data/uploads data/reports artifacts/cache_dets

# 4. Install dependencies (from provided pyproject.toml)
uv sync
```

### Model Setup

Place your trained model files in the `artifacts/` directory:

```
artifacts/
├── lgbm_frame_scorer.joblib    # Trained LightGBM model
└── meta.json                   # Model metadata and configuration
```

**Required `meta.json` format:**
```json
{
  "threshold": 0.5,
  "feature_names": ["feature1", "feature2", ...],
  "model_version": "1.0",
  "training_date": "2025-08-24"
}
```

### Running the Application

```bash
# Start the dashboard
uv run streamlit run app/streamlit_app.py
```

🌐 **Access the dashboard at:** http://localhost:8501

## 📖 User Guide

### 🎮 Dashboard Controls

#### **Sidebar Configuration**
- **📁 Artifacts Directory**: Path to model files (default: `artifacts`)
- **🔄 Processing Mode**: 
  - *Replay*: Use cached detections for instant analysis
  - *Live Detection*: Real-time processing with YOLO+DeepSORT
- **🎯 Anomaly Threshold**: Detection sensitivity (0.0 - 1.0)
- **📊 Smoothing Window**: Frame averaging for score stabilization
- **⚡ Processing FPS**: Frame rate limit for performance optimization
- **🖼️ Display Options**: Toggle tracking IDs and screenshot capture

#### **Video Processing Workflow**

1. **📤 Upload Video**: Drag & drop or browse for MP4/AVI files
2. **⚙️ Configure Settings**: Adjust threshold, smoothing, and processing options
3. **▶️ Start Processing**: Begin analysis with selected mode
4. **👀 Monitor Progress**: Watch live video feed with detection overlays
5. **📊 Review Results**: Analyze real-time score charts and event detection
6. **📋 Generate Report**: Create comprehensive analysis report
7. **💾 Download Results**: Get complete analysis package

### 📊 Interface Layout

#### **Main Dashboard**
```
┌─────────────────┬──────────────────────────┐
│     Sidebar     │     Video Display        │
│   - Settings    │   - Live Feed            │
│   - Controls    │   - Bounding Boxes       │
│   - File Info   │   - Track IDs            │
│                 │                          │
├─────────────────┼──────────────────────────┤
│    Events       │    Anomaly Score Chart   │
│   - Real-time   │   - Timeline View        │
│   - Timestamps  │   - Threshold Line       │
│   - Screenshots │   - Live Updates         │
└─────────────────┴──────────────────────────┘
```

#### **Results & Downloads**
- **📋 Event Summary**: Complete list of detected anomalies
- **📸 Screenshot Gallery**: Full-size event screenshots
- **📦 Complete Report**: Single download with all analysis data

### 🔄 Replay Mode

Access historical analysis through the **Replay** page:

1. Navigate to **"Replay Previous Runs"** in the page selector
2. Choose from available report directories
3. Review past analysis results with:
   - Interactive score timelines
   - Event summaries with screenshots
   - Complete historical data

## 🛠️ Advanced Configuration

### Performance Optimization

#### **Memory Management**
- Reduce **Processing FPS** (8-10 for CPU systems)
- Enable **Auto-cleanup** to manage disk space
- Use **Replay mode** for repeated analysis

#### **Custom Detection Classes**
Modify `CLASSES_TO_DETECT` in `inference/pipeline.py` to focus on specific object types:
```python
CLASSES_TO_DETECT = [0, 2, 5, 7]  # person, car, bus, truck
```

### Model Customization

#### **Feature Engineering**
Update feature extraction in `inference/pipeline.py`:
```python
def extract_features(detections, frame_info):
    # Custom feature extraction logic
    return engineered_features
```

#### **Scoring Models**
Replace LightGBM with custom models by updating `ScorerAdapter` class.

## 📁 Project Structure

```
anomaly-dashboard/
├── 📱 app/
│   ├── streamlit_app.py        # Main dashboard application
│   └── pages/
│       └── replay.py           # Historical analysis viewer
├── 🧠 inference/
│   ├── pipeline.py             # Core detection & scoring pipeline
│   ├── cache_io.py             # JSONL cache management
│   ├── events.py               # Event detection & reporting
│   └── draw.py                 # Video overlay rendering
├── 🎯 artifacts/
│   ├── lgbm_frame_scorer.joblib # Trained anomaly detection model
│   ├── meta.json               # Model configuration
│   └── cache_dets/             # Detection cache storage
├── 💾 data/
│   ├── uploads/                # User-uploaded videos
│   └── reports/                # Generated analysis reports
├── pyproject.toml              # Project dependencies
├── uv.lock                     # Dependency lock file
└── README.md                   # This documentation
```

## 🔧 Troubleshooting

### Common Issues

#### **🚫 Model Not Found**
```
Error: Cannot load scorer from artifacts directory
```
**Solution**: Ensure `lgbm_frame_scorer.joblib` and `meta.json` exist in `artifacts/`

#### **⚡ Performance Issues**
```
Video playback is slow or choppy
```
**Solutions**:
- Reduce Processing FPS to 8-10
- Enable person-only detection
- Use smaller video resolution
- Enable auto-cleanup for disk space

#### **💾 Cache Issues**
```
Replay mode not finding cached detections
```
**Solution**: Check `artifacts/cache_dets/` for matching `<video>_<size>.jsonl` files

#### **📊 No Data in Charts**
```
Score chart appears empty
```
**Solutions**:
- Ensure video processing has started
- Check that threshold is set correctly
- Verify model artifacts are loaded

### Debug Mode

Enable verbose logging by setting environment variables:
```bash
export STREAMLIT_LOGGER_LEVEL=debug
uv run streamlit run app/streamlit_app.py
```

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature-name`
3. **Commit** changes: `git commit -am 'Add feature'`
4. **Push** to branch: `git push origin feature-name`
5. **Submit** a Pull Request


## 🙏 Acknowledgments

- **[YOLOv8](https://github.com/ultralytics/ultralytics)** - Object detection framework
- **[DeepSORT](https://github.com/nwojke/deep_sort)** - Multi-object tracking
- **[LightGBM](https://lightgbm.readthedocs.io)** - Gradient boosting framework
- **[Streamlit](https://streamlit.io)** - Web application framework

---

**⭐ Star this repository if you find it useful!**