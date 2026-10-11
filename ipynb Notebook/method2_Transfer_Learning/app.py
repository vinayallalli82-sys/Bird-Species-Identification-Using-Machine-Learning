import os
import base64
from io import BytesIO

import torch
import torch.nn as nn
import streamlit as st
from PIL import Image
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ============================================================
# CONFIG
# ============================================================

PROJECT_PATH = r"C:\Users\Vinay\Downloads\archive\bird_species_project"
DATASET_PATH = r"C:\Users\Vinay\Downloads\archive\CUB_200_2011"

MODEL_PATH = os.path.join(PROJECT_PATH, "efficientnet_b0_cub_finetuned.pth")
IMAGES_DIR = os.path.join(DATASET_PATH, "images")

LOW_CONFIDENCE = 0.40   # below this we warn the user


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Aviary · Bird Species Identification",
    page_icon="🪶",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@600;700&family=Inter:wght@400;500;600&display=swap');

:root {
  --bg: #f4f1ea;
  --card: #ffffff;
  --ink: #1d2a24;
  --muted: #6b7a72;
  --line: #e4dfd2;
  --green: #2f5d46;
  --green-soft: #e6efe9;
  --accent: #d98e2b;
  --warn-bg: #fff4e0;
  --warn-ink: #8a5a10;
}

html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif;
  color: var(--ink);
}
.stApp { background: var(--bg); }

#MainMenu, footer, header { visibility: hidden; }

.block-container {
  max-width: 1180px;
  padding-top: 0 !important;
  padding-bottom: 3rem;
}

/* ---------- Navbar ---------- */
.nav {
  display: flex; align-items: center; justify-content: space-between;
  padding: 22px 0;
  border-bottom: 1px solid var(--line);
  margin-bottom: 36px;
}
.brand {
  font-family: 'Fraunces', serif;
  font-size: 26px; font-weight: 700; color: var(--green);
  letter-spacing: -0.5px;
}
.brand span { color: var(--accent); }
.nav-links { display: flex; gap: 28px; font-size: 14px; color: var(--muted); font-weight: 500; }

/* ---------- Hero ---------- */
.hero { max-width: 760px; margin-bottom: 34px; }
.eyebrow {
  display: inline-block;
  font-size: 12px; font-weight: 600; letter-spacing: 1.4px; text-transform: uppercase;
  color: var(--green); background: var(--green-soft);
  padding: 6px 12px; border-radius: 999px; margin-bottom: 18px;
}
.hero h1 {
  font-family: 'Fraunces', serif;
  font-size: 54px; line-height: 1.05; letter-spacing: -1.5px;
  font-weight: 700; color: var(--ink); margin: 0 0 16px 0;
}
.hero h1 em { color: var(--green); font-style: normal; }
.hero p { font-size: 18px; line-height: 1.6; color: var(--muted); margin: 0; }

/* ---------- Cards ---------- */
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 26px;
  box-shadow: 0 1px 2px rgba(29,42,36,.04), 0 12px 32px rgba(29,42,36,.05);
}
.card-title {
  font-size: 12px; font-weight: 600; letter-spacing: 1.2px;
  text-transform: uppercase; color: var(--muted); margin-bottom: 14px;
}

/* ---------- Uploader ---------- */
[data-testid="stFileUploader"] section {
  background: #faf8f3;
  border: 2px dashed #cfc8b6;
  border-radius: 16px;
  padding: 30px;
}
[data-testid="stFileUploader"] section:hover { border-color: var(--green); }

[data-testid="stImage"] img { border-radius: 16px; }

/* ---------- Result ---------- */
.result-top { display: flex; align-items: center; gap: 24px; }
.ring {
  width: 104px; height: 104px; border-radius: 50%; flex-shrink: 0;
  display: grid; place-items: center;
}
.ring-inner {
  width: 80px; height: 80px; border-radius: 50%; background: #fff;
  display: grid; place-items: center;
  font-family: 'Fraunces', serif; font-size: 22px; font-weight: 700; color: var(--ink);
}
.species {
  font-family: 'Fraunces', serif;
  font-size: 32px; line-height: 1.1; font-weight: 700; color: var(--ink);
  letter-spacing: -0.5px;
}
.species-sub { font-size: 14px; color: var(--muted); margin-top: 6px; }

.badge {
  display: inline-block; font-size: 12px; font-weight: 600;
  padding: 4px 10px; border-radius: 999px; margin-bottom: 10px;
}
.badge.high { background: var(--green-soft); color: var(--green); }
.badge.mid  { background: var(--warn-bg); color: var(--warn-ink); }
.badge.low  { background: #fdeaea; color: #a23030; }

.warn {
  background: var(--warn-bg); color: var(--warn-ink);
  border-radius: 12px; padding: 12px 16px; font-size: 14px; margin-top: 18px;
  line-height: 1.5;
}

/* ---------- Top-3 bars ---------- */
.pred-row { margin-bottom: 16px; }
.pred-head { display: flex; justify-content: space-between; font-size: 14px; font-weight: 500; margin-bottom: 6px; }
.pred-head span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
.bar { height: 8px; background: #efebe0; border-radius: 99px; overflow: hidden; }
.bar > div { height: 100%; border-radius: 99px; background: var(--green); }
.bar.alt > div { background: #a9bfb2; }

/* ---------- Reference image ---------- */
.ref-img { width: 100%; border-radius: 14px; display: block; }
.ref-cap { font-size: 12px; color: var(--muted); margin-top: 8px; }

/* ---------- Empty state ---------- */
.empty { text-align: center; padding: 40px 10px; color: var(--muted); }
.empty .big { font-size: 46px; margin-bottom: 8px; }
.empty b { color: var(--ink); display: block; font-size: 17px; margin-bottom: 4px; }

/* ---------- Stats ---------- */
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-top: 44px; }
.stat {
  background: var(--card); border: 1px solid var(--line); border-radius: 16px; padding: 20px;
}
.stat .k { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
.stat .v { font-family: 'Fraunces', serif; font-size: 26px; font-weight: 700; margin-top: 6px; color: var(--ink); }

/* ---------- Footer ---------- */
.app-footer {
  margin-top: 50px; padding-top: 22px; border-top: 1px solid var(--line);
  display: flex; justify-content: space-between; font-size: 13px; color: var(--muted);
}

@media (max-width: 800px) {
  .hero h1 { font-size: 36px; }
  .stats { grid-template-columns: repeat(2, 1fr); }
  .nav-links { display: none; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA + MODEL
# ============================================================

@st.cache_data
def load_class_names():
    names = {}
    with open(os.path.join(DATASET_PATH, "classes.txt"), "r") as f:
        for line in f:
            cid, cname = line.strip().split(" ", 1)
            names[int(cid)] = cname
    return names


@st.cache_resource
def load_model():
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 200)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, weights


class_names = load_class_names()
model, weights = load_model()
transform = weights.transforms()


# ============================================================
# HELPERS
# ============================================================

def pretty(raw_name: str) -> str:
    """'001.Black_footed_Albatross' -> 'Black footed Albatross'"""
    return raw_name.split(".", 1)[1].replace("_", " ")


@st.cache_data(show_spinner=False)
def reference_image_b64(class_id: int):
    """First dataset photo of the predicted species, as a base64 JPEG."""
    folder = os.path.join(IMAGES_DIR, class_names[class_id])
    if not os.path.isdir(folder):
        return None
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".jpg"))
    if not files:
        return None
    img = Image.open(os.path.join(folder, files[0])).convert("RGB")
    img.thumbnail((700, 700))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def predict(image: Image.Image, k: int = 3):
    x = transform(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        top_p, top_i = torch.topk(probs, k, dim=1)
    return [
        (int(i) + 1, float(p))
        for i, p in zip(top_i[0].tolist(), top_p[0].tolist())
    ]


def confidence_level(p: float):
    if p >= 0.70:
        return "high", "High confidence", "#2f5d46"
    if p >= LOW_CONFIDENCE:
        return "mid", "Moderate confidence", "#d98e2b"
    return "low", "Low confidence", "#c0504d"


# ============================================================
# NAVBAR + HERO
# ============================================================

st.markdown(
    """
<div class="nav">
<div class="brand">Aviary<span>.</span></div>
<div class="nav-links"><span>Identify</span><span>Species</span><span>About</span></div>
</div>

<div class="hero">
<div class="eyebrow">AI · Computer Vision</div>
<h1>Know every bird <em>at a glance.</em></h1>
<p>Upload a photo and Aviary identifies the species from 200 North American birds,
with a confidence score and a reference photo to compare against.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# MAIN LAYOUT
# ============================================================

left, right = st.columns([1, 1.05], gap="large")

with left:
    st.markdown('<div class="card-title">1 · Upload a photo</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a bird image",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
        help="Use a clear photo where the bird fills a good part of the frame.",
    )

    if uploaded is not None:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, use_container_width=True)
        st.caption("Tip: close-up, well-lit photos with the whole bird visible work best.")

with right:
    st.markdown('<div class="card-title">2 · Result</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.markdown(
            """
<div class="card"><div class="empty">
<div class="big">🪶</div>
<b>No photo yet</b>
Upload an image on the left and the result will appear here automatically.
</div></div>
""",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Analyzing plumage, shape and colour…"):
            results = predict(image, 3)

        best_id, best_p = results[0]
        level, label, color = confidence_level(best_p)
        pct = best_p * 100
        deg = best_p * 360

        warn_html = ""
        if best_p < LOW_CONFIDENCE:
            warn_html = (
                '<div class="warn">The model is not very sure about this one. '
                "Try a sharper, closer photo, and compare with the alternatives below.</div>"
            )

        # ---- main result card (single HTML block) ----
        st.markdown(
            f"""
<div class="card">
<div class="result-top">
<div class="ring" style="background: conic-gradient({color} {deg}deg, #efebe0 0deg);">
<div class="ring-inner">{pct:.0f}%</div>
</div>
<div>
<span class="badge {level}">{label}</span>
<div class="species">{pretty(class_names[best_id])}</div>
<div class="species-sub">Most likely species · {pct:.1f}% match</div>
</div>
</div>
{warn_html}
</div>
""",
            unsafe_allow_html=True,
        )

        # ---- top-3 breakdown ----
        rows = ""
        for rank, (cid, p) in enumerate(results):
            alt = "" if rank == 0 else "alt"
            rows += f"""
<div class="pred-row">
<div class="pred-head"><span>{rank + 1}. {pretty(class_names[cid])}</span><span>{p * 100:.1f}%</span></div>
<div class="bar {alt}"><div style="width:{p * 100:.1f}%"></div></div>
</div>
"""
        st.markdown(
            f"""
<div class="card" style="margin-top:16px;">
<div class="card-title">Top 3 candidates</div>
{rows}
</div>
""",
            unsafe_allow_html=True,
        )

        # ---- reference photo from dataset ----
        ref = reference_image_b64(best_id)
        if ref:
            st.markdown(
                f"""
<div class="card" style="margin-top:16px;">
<div class="card-title">Reference photo</div>
<img class="ref-img" src="data:image/jpeg;base64,{ref}" />
<div class="ref-cap">Example of {pretty(class_names[best_id])} from the CUB-200-2011 dataset.</div>
</div>
""",
                unsafe_allow_html=True,
            )


# ============================================================
# STATS
# ============================================================

st.markdown(
    """
<div class="stats">
<div class="stat"><div class="k">Model</div><div class="v">EfficientNet-B0</div></div>
<div class="stat"><div class="k">Species</div><div class="v">200</div></div>
<div class="stat"><div class="k">Test accuracy</div><div class="v">55.00%</div></div>
<div class="stat"><div class="k">Training</div><div class="v">Fine-tuned</div></div>
</div>

<div class="app-footer">
<span>Aviary · Bird species identification</span>
<span>Trained on CUB-200-2011 · Predictions are probabilistic</span>
</div>
""",
    unsafe_allow_html=True,
)