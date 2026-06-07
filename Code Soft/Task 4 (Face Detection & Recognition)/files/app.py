"""
Face Detection & Recognition — Streamlit Web App
Stack: MTCNN · InsightFace (ArcFace) · OpenCV · Scikit-Learn · Streamlit
Run:  streamlit run app.py
"""

import os
import cv2
import pickle
import tempfile
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
# Pure-numpy replacements for sklearn (avoids Cython/murmurhash build errors)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Return cosine similarity between two 1-D vectors."""
    a, b = np.asarray(a, dtype=np.float32).flatten(), np.asarray(b, dtype=np.float32).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0

def normalize(vec: np.ndarray) -> np.ndarray:
    """L2-normalise a 1-D or 2-D array (row-wise for 2-D)."""
    vec = np.asarray(vec, dtype=np.float32)
    if vec.ndim == 1:
        n = np.linalg.norm(vec)
        return vec / n if n > 0 else vec
    norms = np.linalg.norm(vec, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return vec / norms

# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Recognition System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.4rem; font-weight: 800; color: #4F8BF9; }
    .sub-header  { font-size: 1.1rem; color: #888; margin-bottom: 1.5rem; }
    .metric-box  {
        background: #1e1e2e; border-radius: 12px; padding: 16px 20px;
        border-left: 4px solid #4F8BF9; margin-bottom: 8px;
    }
    .tag-green  { background:#1a4731; color:#4ade80; border-radius:8px; padding:2px 10px; font-size:.85rem; }
    .tag-red    { background:#4c1b1b; color:#f87171; border-radius:8px; padding:2px 10px; font-size:.85rem; }
    .stAlert    { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
DB_PATH        = "embeddings/face_embeddings.pkl"
DATASET_DIR    = "dataset"
THRESHOLD_DEFAULT = 0.45


# ─────────────────────────────────────────────────────────────
# Cached model loaders
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading MTCNN …")
def load_mtcnn():
    from mtcnn import MTCNN
    return MTCNN()


@st.cache_resource(show_spinner="⏳ Loading ArcFace (buffalo_l) …")
def load_arcface():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(
        name="buffalo_l",
        root="models",
        providers=["CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


@st.cache_data(show_spinner=False)
def load_database(db_path: str = DB_PATH):
    if not Path(db_path).exists():
        return {}
    with open(db_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────
def preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    """CLAHE enhancement for lighting robustness."""
    img = cv2.resize(img_bgr, (640, 640))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def get_embedding(img_bgr: np.ndarray, face_app) -> np.ndarray | None:
    img_pre = preprocess_image(img_bgr)
    faces = face_app.get(img_pre)
    if not faces:
        return None
    face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    return normalize(np.array([face.embedding]))[0]


def detect_mtcnn(img_bgr: np.ndarray, detector, min_conf: float = 0.95):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)
    return [(*r["box"], r["confidence"]) for r in results if r["confidence"] >= min_conf]


def recognise(emb: np.ndarray, database: dict, threshold: float) -> tuple[str, float]:
    if not database:
        return "Unknown", 0.0
    scores = {n: cosine_similarity(emb, e) for n, e in database.items()}
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= threshold else ("Unknown", scores[best])


def annotate(frame: np.ndarray, database: dict, detector, face_app,
             threshold: float) -> tuple[np.ndarray, list]:
    output = frame.copy()
    boxes  = detect_mtcnn(frame, detector)
    detections = []

    for (x, y, w, h, conf) in boxes:
        x, y = max(0, x), max(0, y)
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            continue

        emb = get_embedding(crop, face_app)
        if emb is None:
            name, score, color = "No Embedding", 0.0, (128, 128, 128)
        else:
            name, score = recognise(emb, database, threshold)
            color = (0, 200, 80) if name != "Unknown" else (0, 60, 220)

        detections.append({"name": name, "score": score, "box": (x, y, w, h), "conf": conf})

        cv2.rectangle(output, (x, y), (x+w, y+h), color, 2)
        label = f"{name} ({score:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(output, (x, y-th-8), (x+tw+4, y), color, -1)
        cv2.putText(output, label, (x+2, y-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return output, detections


def save_database(db: dict, path: str = DB_PATH):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)
    load_database.clear()          # bust Streamlit cache


# ─────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    threshold = st.slider(
        "Recognition Threshold",
        min_value=0.20, max_value=0.80,
        value=THRESHOLD_DEFAULT, step=0.01,
        help="Higher = stricter. Lower = more permissive."
    )

    st.divider()
    st.markdown("## 📊 Database Status")
    database = load_database()
    if database:
        st.success(f"✅ {len(database)} person(s) enrolled")
        for name in database:
            st.markdown(f"&nbsp;&nbsp;👤 **{name}**", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No persons enrolled yet")

    st.divider()
    st.markdown("## 🗑️ Manage Database")
    if database and st.button("🗑️ Clear entire database", type="secondary"):
        if Path(DB_PATH).exists():
            os.remove(DB_PATH)
        load_database.clear()
        st.rerun()

    if database:
        person_to_del = st.selectbox("Remove a person", ["—"] + list(database.keys()))
        if person_to_del != "—" and st.button(f"Remove {person_to_del}"):
            database.pop(person_to_del)
            save_database(database)
            st.rerun()

# ─────────────────────────────────────────────────────────────
# Main header
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🧠 Face Detection & Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">MTCNN · ArcFace (buffalo_l) · Cosine Similarity</div>', unsafe_allow_html=True)

# Load models once
detector  = load_mtcnn()
face_app  = load_arcface()
database  = load_database()

# ─────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Recognise Image",
    "📹 Live Webcam",
    "➕ Enroll Person",
    "ℹ️ About",
])

# ═══════════════════════════════════════════════════
# TAB 1 — Recognise image
# ═══════════════════════════════════════════════════
with tab1:
    st.subheader("Upload an image to detect and recognise faces")
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        uploaded = st.file_uploader(
            "Choose an image (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

    with col2:
        if uploaded:
            file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
            img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            with st.spinner("🔍 Detecting & recognising …"):
                t0 = time.time()
                result_img, detections = annotate(img_bgr, database, detector, face_app, threshold)
                elapsed = time.time() - t0

            st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Result", use_container_width=True)

            if detections:
                st.markdown(f"⏱️ Inference time: **{elapsed*1000:.0f} ms** | Faces: **{len(detections)}**")
                for d in detections:
                    tag = "tag-green" if d["name"] != "Unknown" else "tag-red"
                    st.markdown(
                        f'<div class="metric-box">'
                        f'<span class="{tag}">{d["name"]}</span>&nbsp;&nbsp;'
                        f'Score: <b>{d["score"]:.3f}</b> &nbsp;|&nbsp; '
                        f'Detection conf: <b>{d["conf"]:.3f}</b>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.warning("No faces detected. Try a clearer image.")
        else:
            st.info("👆 Upload an image to get started.")

# ═══════════════════════════════════════════════════
# TAB 2 — Live webcam
# ═══════════════════════════════════════════════════
with tab2:
    st.subheader("📹 Real-Time Webcam Recognition")
    st.info(
        "⚠️ Streamlit's webcam widget captures a single snapshot per click.  \n"
        "For continuous live video, use the Jupyter notebook (`run_realtime_recognition()`)."
    )

    cam_image = st.camera_input("Take a snapshot")

    if cam_image:
        file_bytes = np.asarray(bytearray(cam_image.read()), dtype=np.uint8)
        img_bgr    = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        with st.spinner("🔍 Processing …"):
            result_img, detections = annotate(img_bgr, database, detector, face_app, threshold)

        col_a, col_b = st.columns(2)
        col_a.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),    caption="Original",  use_container_width=True)
        col_b.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption="Recognised", use_container_width=True)

        if detections:
            for d in detections:
                tag = "tag-green" if d["name"] != "Unknown" else "tag-red"
                st.markdown(
                    f'<div class="metric-box">'
                    f'<span class="{tag}">{d["name"]}</span> &nbsp; Score: <b>{d["score"]:.3f}</b>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No faces detected.")

# ═══════════════════════════════════════════════════
# TAB 3 — Enroll person
# ═══════════════════════════════════════════════════
with tab3:
    st.subheader("➕ Enroll a New Person")
    st.markdown(
        "Upload **10–50 face photos** of the new person.  \n"
        "Variety of angles, lighting & expressions gives best accuracy."
    )

    person_name = st.text_input("Person Name", placeholder="e.g. Lokesh")
    enroll_files = st.file_uploader(
        "Upload face images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("✅ Enroll Person", type="primary", disabled=not (person_name and enroll_files)):
        database = load_database()
        embeddings_list = []
        progress = st.progress(0, text="Extracting embeddings …")

        for i, f in enumerate(enroll_files):
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if img is None:
                continue

            emb = get_embedding(img, face_app)
            if emb is not None:
                embeddings_list.append(emb)

            # Augment: horizontal flip
            emb_f = get_embedding(cv2.flip(img, 1), face_app)
            if emb_f is not None:
                embeddings_list.append(emb_f)

            progress.progress((i + 1) / len(enroll_files),
                               text=f"Processing image {i+1}/{len(enroll_files)} …")

        if embeddings_list:
            mean_emb = normalize(np.array([np.mean(embeddings_list, axis=0)]))[0]
            database[person_name] = mean_emb
            save_database(database)
            st.success(
                f"✅ **{person_name}** enrolled with {len(embeddings_list)} embedding samples!"
            )
            st.balloons()
        else:
            st.error("❌ No valid face found in any uploaded image. Use clear, front-facing photos.")

# ═══════════════════════════════════════════════════
# TAB 4 — About
# ═══════════════════════════════════════════════════
with tab4:
    st.subheader("ℹ️ About this System")
    st.markdown("""
| Component | Technology | Notes |
|---|---|---|
| Face Detection | **MTCNN** | Detects frontal & profile faces + landmarks |
| Face Recognition | **ArcFace (buffalo_l)** | 512-d embeddings, 99.8 % LFW accuracy |
| Preprocessing | **CLAHE** | Adaptive histogram equalisation for lighting |
| Augmentation | **Horizontal flip** | Doubles training data coverage |
| Matching | **Cosine Similarity** | L2-normalised embeddings |
| Storage | **Pickle** | Lightweight local embedding database |

### Accuracy Improvements Over Baseline
- `buffalo_l` instead of default InsightFace model (highest available accuracy)
- CLAHE preprocessing eliminates poor-lighting failures
- Mean-embedding aggregation across all enrolled images per person
- L2 normalisation ensures cosine comparisons are scale-invariant
- Augmentation (flip) improves pose robustness
- MTCNN confidence threshold ≥ 0.95 ensures clean training data

### Run this app
```bash
streamlit run app.py
```

### Jupyter Notebook
Open `Face_Detection_Recognition.ipynb` and run cells in order.
""")
