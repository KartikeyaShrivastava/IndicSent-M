"""IndicSent-M — Indian Multimodal Sentiment Analyzer (Streamlit app).

    streamlit run app.py
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pandas as pd
import streamlit as st
from PIL import Image

from src.data.image_utils import DEFAULT_MAX_BYTES, InvalidImageError, load_image
from src.inference import InputError, ModelNotAvailable, SentimentPredictor, best_multimodal
from src.utils import PROJECT_ROOT, load_config, resolve_path

st.set_page_config(page_title="IndicSent-M · Sentiment Studio", page_icon="◈", layout="wide")

CFG = load_config()
REPORTS = resolve_path(CFG["paths"]["reports_dir"])
LABEL_STYLE = {"positive": ("#187451", "+"), "neutral": ("#64748b", "="), "negative": ("#b43f51", "−")}
EXP_NAMES = {"text": "Text-only · XLM-R", "image": "Image-only · ResNet-50",
             "multimodal_concat": "Multimodal · concatenation fusion",
             "multimodal_attention": "Multimodal · cross-attention fusion"}

st.markdown("""
<style>
:root {--ink:#172b49; --muted:#52647b; --line:#dce4ef; --accent:#2859d8;}
.stApp {background:#f6f8fc; color:var(--ink);}
.block-container {padding-top:2.5rem; padding-bottom:3rem; max-width:1240px;}
h1,h2,h3,h4 {letter-spacing:-.035em; color:var(--ink);}
h1 {font-weight:750 !important;} h3 {font-size:1.2rem !important;}
p,li {line-height:1.7;}
[data-testid="stCaptionContainer"] {opacity:1;}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color:var(--muted);}
[data-testid="stSidebar"] {background:#fff; border-right:1px solid var(--line);}
[data-testid="stSidebar"] [role="radiogroup"] {gap:.4rem;}
[data-testid="stSidebar"] [role="radiogroup"] label {
    padding:.65rem .8rem; border-radius:9px; width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
    background:#edf2ff; color:#234fbd;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:#f3f6fc;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-radius:14px !important;}
.st-key-caption_panel, .st-key-image_panel {background:#fff;}
[data-testid="stMetric"] {background:white; border:1px solid var(--line);
    border-radius:12px; padding:1rem 1.15rem;}
[data-testid="stMetricValue"] {font-size:1.8rem; letter-spacing:-.04em;}
[data-testid="stFileUploader"] {border-radius:12px;}
[data-testid="stImage"] img {border-radius:10px;}
.stButton > button {border-radius:9px; min-height:2.75rem; font-weight:600;}
.stButton > button[kind="primary"] {box-shadow:0 3px 8px #2859d81a;}
button:focus-visible, input:focus-visible, textarea:focus-visible,
[role="tab"]:focus-visible {outline:3px solid #2859d8 !important; outline-offset:3px;}
.hero {padding:.25rem 0 1.25rem; margin-bottom:.65rem; border-bottom:1px solid var(--line);}
.hero h1 {font-size:clamp(1.9rem,4vw,2.65rem); line-height:1.2; margin:.6rem 0; padding:0;}
.hero p {color:var(--muted); max-width:720px; margin:.6rem 0; font-size:1rem;}
.eyebrow {color:var(--accent); font-size:.72rem; font-weight:700; letter-spacing:.14em;
    text-transform:uppercase; margin-bottom:.5rem;}
.brand {display:flex; align-items:center; gap:.65rem; margin:.5rem 0 1.2rem;}
.brand-mark {display:grid; place-items:center; width:38px; height:38px; background:#2859d8;
    color:white; font-size:1.5rem; border-radius:10px;}
.brand strong {font-size:1.15rem; letter-spacing:-.04em;}
.brand small {display:block; color:var(--muted); font-size:.75rem;}
.card {background:#fff; border:1px solid var(--line); border-radius:14px;
    padding:1.3rem; min-height:180px; margin-bottom:.7rem;}
.card h4 {font-size:1.05rem; margin:.65rem 0 .5rem; padding:0;}
.card p {color:var(--muted); margin:0; font-size:.92rem;}
.card-index {font-size:.75rem; font-weight:700; color:var(--accent);}
.pill {display:inline-block; padding:.22rem .65rem; border-radius:6px; font-size:.76rem;
    font-weight:600; background:#edf2ff; color:#234fbd; margin:.15rem .25rem .15rem 0;}
.flow {display:grid; grid-template-columns:1fr auto 1fr auto 1fr; gap:1rem; align-items:center;
    background:white; border:1px solid var(--line); border-radius:14px; padding:1.5rem; margin:.7rem 0 1rem;}
.flow strong,.flow span {display:block;}.flow span {font-size:.85rem; color:var(--muted); margin-top:.4rem;}
.flow-arrow {color:#8d9db4; font-size:1.3rem;}
.result {background:white; border:1px solid var(--line); border-left:5px solid var(--result-color);
    border-radius:14px; padding:1.5rem; display:flex; justify-content:space-between;
    align-items:center; gap:1rem; margin-bottom:.8rem;}
.result h2 {color:var(--result-color); font-size:2rem; margin:.2rem 0; padding:0;}
.result .confidence {text-align:right;}.confidence strong {font-size:2rem; display:block; letter-spacing:-.05em;}
.small {font-size:.82rem; color:var(--muted);}
.probabilities {background:white; border:1px solid var(--line); border-radius:12px; padding:1.2rem; margin:.5rem 0 1rem;}
.prob-row {display:grid; grid-template-columns:80px 1fr 56px; align-items:center; gap:1rem; margin:.7rem 0; font-size:.86rem;}
.prob-track {background:#edf1f7; border-radius:20px; height:9px; overflow:hidden;}
.prob-fill {height:100%; border-radius:20px;}
.prob-value {text-align:right; font-variant-numeric:tabular-nums;}
.demo {border:2px dashed #b86b17; border-radius:12px; padding:1rem 1.2rem; background:#fff8eb; color:#714013;}
.demo h3 {color:#714013; font-size:1rem !important; margin:0 0 .4rem; padding:0;}
.empty {text-align:center; border:1px dashed #c7d3e5; border-radius:12px;
    padding:2rem 1rem; color:var(--muted); background:#fbfcff; margin:.5rem 0;}
.empty strong {display:block; color:var(--ink); margin-bottom:.4rem;}
.model-row {border-bottom:1px solid #edf1f7; padding:.7rem 0; font-size:.8rem;}
.model-row strong {display:block; font-weight:600;}
.footer {border-top:1px solid var(--line); padding-top:1rem; margin-top:2rem;
    font-size:.78rem; color:var(--muted);}
@media(max-width:760px) {
    .block-container {padding:3.5rem 1rem 2rem;}
    .flow {grid-template-columns:1fr; gap:.6rem; padding:1.2rem;}
    .flow-arrow {transform:rotate(90deg); justify-self:start;}
    .card {min-height:0;} .result {align-items:flex-start; flex-wrap:wrap;}
    .result .confidence {text-align:left;}
    .prob-row {grid-template-columns:65px 1fr 50px; gap:.6rem;}
    [data-testid="stHorizontalBlock"] {flex-wrap:wrap;}
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width:100% !important; flex:1 1 100% !important; min-width:0 !important;}
    [role="tablist"] {overflow-x:auto;}
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------------ helpers
@st.cache_resource(show_spinner="Loading model runtime…")
def get_predictor() -> SentimentPredictor:
    return SentimentPredictor()


def read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="hero"><div class="eyebrow">IndicSent-M / Sentiment Studio</div>'
                f'<h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p></div>', unsafe_allow_html=True)


def card(index: str, title: str, body: str) -> None:
    st.markdown(f'<div class="card"><span class="card-index">{html.escape(index)}</span>'
                f'<h4>{html.escape(title)}</h4><p>{html.escape(body)}</p></div>', unsafe_allow_html=True)


def navigate(page: str) -> None:
    st.session_state.navigation = page


def probability_bars(probabilities: dict) -> None:
    rows = []
    for label, (color, _) in LABEL_STYLE.items():
        value = float(probabilities[label])
        rows.append(f'<div class="prob-row"><span>{label.capitalize()}</span>'
                    f'<div class="prob-track" role="meter" aria-label="{label.capitalize()} probability" '
                    f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{value * 100:.1f}">'
                    f'<div class="prob-fill" style="width:{value * 100:.3f}%;background:{color}"></div></div>'
                    f'<span class="prob-value">{value:.1%}</span></div>')
    st.markdown('<div class="probabilities">' + ''.join(rows) + '</div>', unsafe_allow_html=True)


def image_preview(image: Image.Image, caption: str) -> None:
    preview = image.copy()
    preview.thumbnail((720, 420))
    st.image(preview, caption=caption, width="content")


def dataset_is_synthetic() -> bool:
    s = read_json(PROJECT_ROOT / "data" / "processed" / "stats.json")
    return bool(s and s.get("synthetic"))


# ------------------------------------------------------------------------------------ pages
def page_home() -> None:
    hero("A little context. A clearer sentiment.",
         "Explore how captions and images work together to express the sentiment of Indian-language memes.")
    st.markdown('<span class="pill">Multimodal analysis</span><span class="pill">Bengali · MemoSen</span>'
                '<span class="pill">Explainable predictions</span>', unsafe_allow_html=True)
    st.write("")
    st.button("Analyze a meme →", type="primary", on_click=navigate, args=("Analyze",))
    st.caption("Bring a caption, an image, or both. Compare positive, neutral, and negative sentiment.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        card("01 / EXPLORE", "One meme, three perspectives",
             "Analyze the caption, the picture, or their combination with separately trained models.")
    with c2:
        card("02 / UNDERSTAND", "Look beyond the label",
             "Explore influential words and image regions alongside the model’s class probabilities.")
    with c3:
        card("03 / COMPARE", "Evidence, with context",
             "Compare four models on the same test set, including a separate source-disjoint robustness evaluation.")
    st.markdown("### From meme to meaning")
    st.markdown('<div class="flow"><div><strong>Caption + image</strong><span>Text you supply and a JPG or PNG</span></div>'
                '<div class="flow-arrow" aria-hidden="true">→</div><div><strong>Encode + combine</strong>'
                '<span>XLM-R · ResNet-50 · multimodal fusion</span></div>'
                '<div class="flow-arrow" aria-hidden="true">→</div><div><strong>Sentiment + explanations</strong>'
                '<span>Positive, neutral, or negative</span></div></div>', unsafe_allow_html=True)
    with st.expander("Inside the architecture"):
        st.markdown("**Caption:** normalization → XLM-R base → 768-dimensional mean pooling → 256-dimensional projection.\n\n"
                    "**Image:** 224 × 224 input → ResNet-50 → 2048-dimensional features / 7 × 7 region grid → projection.\n\n"
                    "**Fusion:** concatenation or bidirectional cross-attention between text tokens and 49 image regions, "
                    "followed by an MLP and a three-class softmax.")
    st.info("Trained on Bengali memes, including Bengali–English code-mixing and romanised Bengali. "
            "Other languages are outside the training distribution. Captions must be entered manually; there is no OCR.")
    avail = get_predictor().available()
    if not avail:
        st.warning("No trained model checkpoints were found yet — see the **Analyze** page for setup steps.")
    elif dataset_is_synthetic():
        st.error("The current processed data is the SYNTHETIC sample — results are not meaningful.")


def _setup_message() -> None:
    st.warning("This model is not available yet. A trained checkpoint is required for a real prediction.")
    with st.expander("Model setup instructions"):
        st.markdown("Train the models first (≈1 h on an Apple M1 / faster on a GPU):")
        st.code("python -m src.data.download --dataset memosen\npython -m src.data.prepare\n"
                "python -m src.training.train_text\npython -m src.training.train_image\n"
                "python -m src.training.train_multimodal --fusion concat\n"
                "python -m src.training.train_multimodal --fusion attention\n\n"
                "# or all at once:\npython scripts/run_pipeline.py", language="bash")


def _show_demo(mode: str) -> None:
    st.markdown('<div class="demo"><h3>⚠️ DEMONSTRATION — NO MODEL WAS RUN</h3>'
                'This placeholder exists only to preview the interface layout. The values below are fixed '
                '(uniform) and carry <b>no information</b> about your input.</div>', unsafe_allow_html=True)
    st.caption(f"Layout preview · {mode.capitalize()} mode · fixed placeholder probabilities")
    probability_bars({label: 1 / 3 for label in LABEL_STYLE})


def _render_result(r, show_ig: bool) -> None:
    color, symbol = LABEL_STYLE.get(r.label, ("#333", ""))
    st.markdown("### Analysis result")
    st.markdown(f'<div class="result" style="--result-color:{color}"><div><div class="eyebrow">Predicted sentiment</div>'
                f'<h2>{symbol} {html.escape(r.label.capitalize())}</h2>'
                f'<span class="small">{html.escape(EXP_NAMES[r.experiment])}</span></div>'
                f'<div class="confidence"><strong>{r.confidence:.1%}</strong><span class="small">Model confidence</span>'
                '</div></div>', unsafe_allow_html=True)
    st.caption("Confidence is the model's predicted probability for this class — not a guarantee that "
               "the prediction is correct. Models can be confidently wrong, e.g. on sarcasm.")
    probability_bars(r.probabilities)
    for w in r.warnings:
        st.warning(w)
    if r.script_info:
        si = r.script_info
        st.caption(f"Detected script(s): {', '.join(si['scripts']) or 'none'} · "
                   f"{'code-mixed' if si['code_mixed'] else 'single script'} · hint: {si['language_hint']}")

    if r.text_occlusion or r.gradcam is not None:
        st.markdown("#### Why this prediction?")
    cols = st.columns(2) if (r.text_occlusion and r.gradcam is not None) else [st.container()]
    i = 0
    if r.text_occlusion:
        from src.explainability.text_explainer import render_html
        with cols[i], st.container(border=True):
            st.markdown("**Influential words** — occlusion (leave-one-word-out)")
            st.markdown(render_html(r.text_occlusion["words"], r.label), unsafe_allow_html=True)
            st.caption("Score = drop in P(predicted class) when the word is removed.")
            if show_ig and r.text_attribution:
                ig = r.text_attribution
                st.markdown("**Integrated Gradients**")
                if ig["reliable"]:
                    st.markdown(render_html(ig["words"], r.label), unsafe_allow_html=True)
                    st.caption(f"Completeness residual {ig['convergence_delta']:+.4f} ({ig['steps']} steps) — OK.")
                else:
                    st.info(f"IG did not converge for this input (completeness residual "
                            f"{ig['convergence_delta']:+.4f} vs. ΔP {ig['prob_input'] - ig['prob_baseline']:+.4f}); "
                            "its attributions are hidden because they would be misleading. Use occlusion above.")
        i += 1
    if r.gradcam is not None:
        with cols[min(i, len(cols) - 1)], st.container(border=True):
            st.markdown("**Image regions** — Grad-CAM (ResNet-50 layer4)")
            image_preview(r.gradcam, "Predicted-class sensitivity overlay")
            st.caption("Red = regions the predicted-class score is most sensitive to (7×7 grid, upsampled). "
                       "A sensitivity map, not a causal explanation; it often lights up caption text.")
    with st.expander("Limitations of these explanations"):
        st.markdown(
            "- Both explainers describe the **model's sensitivity**, not the true reason a person would give.\n"
            "- Occlusion removes one word at a time, so it misses interactions (e.g. negation + adjective).\n"
            "- Integrated Gradients depends on the chosen baseline and can fail to converge on transformer "
            "models; we check its completeness property and hide it when it fails.\n"
            "- Grad-CAM is coarse (7×7) and can highlight text rendered in the image rather than visual content.\n"
            "- In multimodal mode the two explanations are computed separately with the other input held fixed.")


def page_analyze() -> None:
    hero("Analyze a meme", "Add your inputs, choose a perspective, and explore the model’s prediction.")
    pred = get_predictor()
    avail = pred.available()
    if dataset_is_synthetic() and avail:
        st.error("Checkpoints were trained on SYNTHETIC sample data — predictions are not meaningful.")

    st.markdown("### 01 / Choose a perspective")
    mode = st.radio("Analysis mode", ["Multimodal (text + image)", "Text-only", "Image-only"], horizontal=True)
    mode_key = {"Multimodal (text + image)": "multimodal", "Text-only": "text", "Image-only": "image"}[mode]
    if mode_key == "multimodal":
        mm_opts = [e for e in ("multimodal_concat", "multimodal_attention") if e in avail] or ["multimodal_concat"]
        default = best_multimodal(avail) or mm_opts[0]
        exp = st.selectbox("Fusion model", mm_opts, index=mm_opts.index(default), format_func=EXP_NAMES.get,
                           help="Default = the multimodal model with the best validation macro-F1.")
    else:
        exp = mode_key

    st.markdown("### 02 / Add your meme")
    st.caption("Bengali and Bengali–English captions match the training data. Other languages may be unreliable.")
    if "example" not in st.session_state:
        st.session_state.example = None
    test_csv = PROJECT_ROOT / "data" / "processed" / "test.csv"
    examples_available = test_csv.exists()
    if st.button("Load a random meme", icon="🎲", disabled=not examples_available,
                 help="Loads an existing meme and its caption from the held-out test set. "
                      "Some examples contain crude or offensive humour."):
        try:
            examples = pd.read_csv(test_csv, keep_default_na=False)
            if not {"id", "text", "label", "image_path"}.issubset(examples.columns) or examples.empty:
                st.warning("No usable test examples were found. Prepare the dataset to enable random memes.")
            else:
                previous = st.session_state.example
                candidates = examples[examples["id"] != previous["id"]] if previous and len(examples) > 1 else examples
                row = candidates.sample(1).iloc[0]
                st.session_state.example = row.to_dict()
                st.session_state.text_input = row["text"]
                # A requested example replaces a previous upload as well as its caption.
                st.session_state.upload_revision = st.session_state.get("upload_revision", 0) + 1
        except (OSError, ValueError, pd.errors.ParserError) as e:
            st.warning(f"The test examples could not be loaded: {e}")
    if not examples_available:
        st.caption("Random memes become available after the MemoSen dataset is downloaded and prepared. "
                   "This loads dataset examples; it does not generate new meme images.")
    ex = st.session_state.example

    input_panels = st.columns(2) if mode_key == "multimodal" else [st.container()]
    text, image_src = None, None
    if mode_key in ("text", "multimodal"):
        with input_panels[0], st.container(border=True, key="caption_panel"):
            st.markdown("#### Caption")
            st.caption("Type or paste the text from your meme. Text is not extracted from the image.")
            text = st.text_area("Meme text / caption", key="text_input", height=170,
                                placeholder="আজকের দিনটা দারুণ কাটলো! 😄\n\nOr try a Bengali–English caption.")
    if mode_key in ("image", "multimodal"):
        with input_panels[-1], st.container(border=True, key="image_panel"):
            st.markdown("#### Image")
            st.caption("Upload the original meme. JPG, JPEG, or PNG · up to 10 MB.")
            up = st.file_uploader("Meme image", type=["jpg", "jpeg", "png"],
                                  key=f"meme_upload_{st.session_state.get('upload_revision', 0)}")
            if up is not None:
                data = up.getvalue()
                try:
                    preview = load_image(data, max_bytes=DEFAULT_MAX_BYTES)
                    image_src = data
                    image_preview(preview, "Uploaded image")
                except InvalidImageError as e:
                    st.error(f"Image rejected: {e}")
            elif ex:
                path = resolve_path(ex.get("image_source") or ex["image_path"])
                image_src = str(path)
                try:
                    with Image.open(path) as sample_image:
                        image_preview(sample_image, "Test-set image")
                except (OSError, ValueError):
                    image_src = None
                    st.warning("This example’s image is unavailable. Upload an image or load another example.")
            else:
                st.markdown('<div class="empty"><strong>Your image will appear here</strong>'
                            'Choose a file above to preview your meme.</div>', unsafe_allow_html=True)
    if ex:
        st.caption(f"Loaded test example `{ex['id']}` · annotated label: **{ex['label']}**. "
                   "Your uploaded image takes priority over the example image.")

    st.markdown("### 03 / Run your analysis")
    explain = st.toggle("Include word and image explanations", value=True,
                        help="Explanations take additional time. Only the explanations relevant to your mode are shown.")
    with st.expander("Advanced display options"):
        show_ig = st.toggle("Show Integrated Gradients (experimental)", value=False, disabled=not explain)
        st.caption("Shows the additional text attribution only when its convergence check passes.")
    demo = False
    if exp not in avail:
        _setup_message()
        demo = st.checkbox("Preview the interface in clearly-labelled DEMO mode (no model)")
    # Keep results through display-only reruns; invalidate them whenever the actual inputs change.
    signature = hashlib.sha256()
    signature.update(json.dumps([exp, text, explain, demo], ensure_ascii=False).encode("utf-8"))
    signature.update(image_src if isinstance(image_src, bytes) else str(image_src).encode("utf-8"))
    input_signature = signature.hexdigest()
    if st.session_state.get("analysis_signature") != input_signature:
        st.session_state.pop("analysis_result", None)
        st.session_state.pop("analysis_demo", None)
    st.session_state.analysis_signature = input_signature
    if st.button("Preview demo layout" if demo else "Analyze meme", type="primary", width="stretch",
                 disabled=exp not in avail and not demo):
        if demo:
            st.session_state.analysis_demo = True
        else:
            st.session_state.pop("analysis_result", None)
            try:
                with st.spinner("Analyzing your meme… Explanations may take a little longer." if explain
                                else "Analyzing your meme…"):
                    st.session_state.analysis_result = pred.predict(text, image_src, exp, explain=explain)
            except InputError as e:
                st.warning(str(e))
                return
            except ModelNotAvailable as e:
                st.error(str(e))
                return
            except Exception as e:  # never crash the UI
                st.error(f"Unexpected error during inference: {type(e).__name__}: {e}")
                return
    if st.session_state.get("analysis_demo"):
        _show_demo(mode_key)
    elif "analysis_result" in st.session_state:
        r = st.session_state.analysis_result
        st.divider()
        _render_result(r, show_ig)
        st.caption(f"Inference {r.latency_ms:.0f} ms on {r.device} (excluding explanations) · checkpoint epoch "
                   f"{r.checkpoint['epoch']}, val macro-F1 {r.checkpoint['val_macro_f1']:.3f}")
    elif exp in avail:
        st.markdown('<div class="empty"><strong>Ready when you are</strong>'
                    'Add your inputs and select Analyze meme to see the prediction here.</div>', unsafe_allow_html=True)


PROTOCOLS = {
    "Random group split (main)": (REPORTS, PROJECT_ROOT / CFG["paths"]["checkpoints_dir"]),
    "Source-disjoint split (robustness)": (PROJECT_ROOT / "artifacts" / "reports_source_disjoint",
                                           PROJECT_ROOT / "artifacts" / "checkpoints_source_disjoint"),
}


def page_comparison() -> None:
    hero("Compare the evidence", "Four models. Shared test examples. A closer look at what improves — and what does not.")
    avail_protocols = [k for k, (r, _) in PROTOCOLS.items() if (r / "comparison" / "comparison.json").exists()]
    if len(avail_protocols) > 1:
        proto = st.radio("Evaluation protocol", avail_protocols, horizontal=True,
                         help="Main: memes split at random (grouped by near-duplicates). Robustness: whole "
                              "source pages held out, so models cannot exploit page-specific style.")
    else:
        proto = avail_protocols[0] if avail_protocols else list(PROTOCOLS)[0]
    reports, ckpts = PROTOCOLS[proto]
    st.caption(f"Selected protocol: {proto}")
    if proto.startswith("Source-disjoint"):
        st.info("Robustness evaluation: entire source pages are held out. These results measure generalization "
                "to unfamiliar pages and should be interpreted separately from the random split.")
    else:
        st.info("Main evaluation: near-duplicate memes are grouped before random splitting. "
                "Source-page styles can still appear in both training and test sets.")
    comp = read_json(reports / "comparison" / "comparison.json")
    if not comp:
        st.info("No comparison yet. Run `python -m src.evaluation.evaluate` then "
                "`python -m src.evaluation.compare_models`.")
        return
    if dataset_is_synthetic():
        st.error("These numbers come from the SYNTHETIC sample data and are meaningless.")
    t = pd.DataFrame(comp["table"])
    trained = t[~t["experiment"].str.startswith("baseline")]
    best = trained.loc[trained["macro_f1"].idxmax()] if not trained.empty else None
    summary = st.columns(3)
    summary[0].metric("Models evaluated", len(trained))
    summary[1].metric("Shared test memes", f"{comp['n_common_test_examples']:,}")
    summary[2].metric("Highest model macro-F1", f"{best['macro_f1']:.3f}" if best is not None else "—")
    if best is not None:
        st.caption(f"Highest macro-F1 in this protocol: {best['model']}. A higher score alone does not establish statistical significance.")
    overview, diagnostics, statistics = st.tabs(["Overview", "Class & training details", "Statistical evidence"])
    with overview:
        st.markdown("### Model performance")
        show = t[["model", "accuracy", "macro_f1", "weighted_f1", "latency_median_ms"]].rename(columns={
            "model": "Model", "accuracy": "Accuracy", "macro_f1": "Macro-F1",
            "weighted_f1": "Weighted F1", "latency_median_ms": "Latency (ms)"})
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "Macro-F1": st.column_config.NumberColumn(format="%.3f"),
            "Weighted F1": st.column_config.NumberColumn(format="%.3f"),
            "Latency (ms)": st.column_config.NumberColumn(format="%.1f"),
        })
        st.caption("Macro-F1 averages the three class F1 scores equally and is the primary metric for this imbalanced dataset. "
                   "Latency is the median per-meme time at batch size 1, including preprocessing.")
        st.caption("The source-page prior uses metadata rather than content; it is a shortcut reference. "
                   "The majority-class baseline always predicts the most frequent training label.")
        p = reports / "comparison" / "accuracy_macro_f1.png"
        if p.exists():
            st.image(str(p), width="stretch")
        with st.expander("Findings derived from these results", expanded=True):
            for line in findings(t, comp):
                st.markdown(f"- {line}")
    with diagnostics:
        st.markdown("### Performance by sentiment")
        p = reports / "comparison" / "per_class_f1.png"
        if p.exists():
            st.image(str(p), width="stretch")
        p = reports / "comparison" / "confusion_matrices.png"
        if p.exists():
            st.markdown("#### Confusion matrices")
            st.image(str(p), width="stretch")
        hist = {e: ckpts / e / "history.csv" for e in EXP_NAMES}
        hist = {e: pd.read_csv(p) for e, p in hist.items() if p.exists()}
        if hist:
            st.markdown("#### Validation macro-F1 per epoch")
            st.line_chart(pd.DataFrame({EXP_NAMES[e]: h.set_index("epoch")["val_macro_f1"] for e, h in hist.items()}))
        gal = reports / "explanations"
        figs = sorted(gal.glob("*.png")) if gal.exists() else []
        if figs:
            with st.expander(f"Explanation gallery · {len(figs)} test memes"):
                for f in figs:
                    st.image(str(f), width="stretch")
    with statistics:
        st.markdown("### Detailed metrics")
        full = t[["model", "accuracy", "macro_precision", "macro_recall", "macro_f1", "weighted_f1",
                  "f1_positive", "f1_neutral", "f1_negative", "trainable_params_M", "training_min",
                  "latency_median_ms"]].set_index("model")
        st.dataframe(full, width="stretch")
        if comp.get("paired_bootstrap"):
            st.markdown("#### Paired bootstrap tests")
            st.caption("Differences in macro-F1 with 95% confidence intervals. Intervals spanning zero do not establish a difference.")
            st.dataframe(pd.DataFrame(comp["paired_bootstrap"]).T, width="stretch")
        if comp.get("modality_ablation"):
            st.markdown("#### Does the model use both inputs?")
            st.caption("Modality ablation: compare performance when the image is neutralized or the caption is removed.")
            rows = [{"model": EXP_NAMES[e], "condition": c, **v} for e, ab in comp["modality_ablation"].items()
                    for c, v in ab.items()]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def findings(t: pd.DataFrame, comp: dict) -> list[str]:
    """Plain-language statements derived only from the numbers in ``t`` and ``comp``."""
    names = {**EXP_NAMES, "baseline_majority": "majority-class baseline",
             "baseline_source_prior": "source-page prior (shortcut reference)"}
    out = []
    t = t[~t["experiment"].str.startswith("baseline")] if (~t["experiment"].str.startswith("baseline")).any() else t
    best = t.loc[t["macro_f1"].idxmax()]
    out.append(f"Highest test macro-F1 among trained models: **{best['model']}** ({best['macro_f1']:.3f}, 95% CI "
               f"{best['macro_f1_ci95'][0]:.3f}–{best['macro_f1_ci95'][1]:.3f}).")
    for k, v in comp.get("paired_bootstrap", {}).items():
        a, b = k.split("_vs_")
        lo, hi = v["ci95"]
        verdict = ("significantly better" if lo > 0 else "significantly worse" if hi < 0
                   else "not significantly different")
        out.append(f"{names.get(a, a)} vs {names.get(b, b)}: Δ macro-F1 = {v['observed_diff']:+.3f} "
                   f"(95% CI {lo:+.3f} to {hi:+.3f}) → {verdict} at the 95% level.")
    if "f1_neutral" in t:
        worst = t["f1_neutral"].max()
        out.append(f"Neutral is the hardest class (best neutral F1 = {worst:.3f}); it is only ~7% of MemoSen.")
    return out


def page_dataset() -> None:
    hero("The data behind the prediction", "Explore MemoSen, the preparation pipeline, and the scope of these experiments.")
    s = read_json(PROJECT_ROOT / "data" / "processed" / "stats.json")
    if not s:
        s = read_json(REPORTS / "dataset" / "stats.json")
        if s:
            st.info("Showing the saved evaluation dataset snapshot. Local processed data has not been generated.")
    if not s:
        st.info("Dataset statistics are not available yet. The methodology is described below.")
        with st.expander("Generate local dataset statistics"):
            st.code("python -m src.data.prepare", language="bash")
    else:
        if s.get("synthetic"):
            st.error("Processed data is the SYNTHETIC sample.")
        st.caption(f"Dataset: {s['dataset']} · {s['usable_records']:,} usable memes of {s['raw_records']:,} "
                   f"annotations · {s['split_source']}")
        c = st.columns(4)
        c[0].metric("Train", s["split_sizes"]["train"])
        c[1].metric("Validation", s["split_sizes"]["val"])
        c[2].metric("Test", s["split_sizes"]["test"])
        c[3].metric("Code-mixed captions", f"{s['code_mixed_fraction']:.0%}")
        figs = REPORTS / "dataset"
        distribution, characteristics = st.tabs(["Labels & scripts", "Caption & image details"])
        with distribution:
            cols = st.columns(2)
            for col, name in zip(cols, ["label_distribution.png", "script_distribution.png"]):
                if (figs / name).exists():
                    col.image(str(figs / name), width="stretch")
            with st.expander("Label counts by split"):
                st.dataframe(pd.DataFrame(s["label_distribution"]), width="stretch")
        with characteristics:
            cols = st.columns(2)
            for col, name in zip(cols, ["text_length.png", "image_sizes.png"]):
                if (figs / name).exists():
                    col.image(str(figs / name), width="stretch")
            with st.expander("Removed records & raw statistics"):
                st.json(s)
    st.markdown("### A reproducible research workflow")
    dataset, preparation, training, limitations = st.tabs(["Dataset", "Preprocessing", "Training & evaluation", "Limitations"])
    with dataset:
        st.markdown("""
#### MemoSen
Bengali meme sentiment dataset (Hossain, Sharif & Hoque, *LREC 2022*): meme images with their
embedded caption text transcribed and a positive / neutral / negative label. Captions are mostly
Bengali script, with English and romanised Bengali mixed in. No explicit licence is published with
the data; it is used here for non-commercial research with citation, and is not redistributed.
""")
    with preparation:
        st.markdown("""
#### Preprocessing
1. Load captions, image file names and labels; map labels to `positive / neutral / negative`.
2. Validate every image (decodable, size limits); drop records whose image is missing or corrupt.
3. Normalise text conservatively: Unicode NFC, URLs → `URL`, mentions → `@USER`, character floods
   capped at 3 — **emojis, punctuation, negations, casing and ZWJ/ZWNJ are kept**.
4. Script-based language identification (Bengali, Devanagari, Latin, …) and code-mixing flag.
5. Remove exact duplicates (same image hash + text); drop duplicates with conflicting labels.
6. Group near-duplicates (same image hash or same long caption) and split **by group**, stratified by
   label, 70/10/20 with seed 42, so no meme template or caption leaks across splits.
""")
    with training:
        st.markdown("""
#### Training (identical for all experiments)
AdamW (lr 2e-5 text encoder, 1e-4 ResNet layer4, 1e-3 new layers, weight decay 0.01), linear
warm-up 10 % + decay, batch 16, ≤ 8 epochs, early stopping (patience 3) and checkpoint selection
on **validation macro-F1**, gradient clipping 1.0, dropout 0.3, cross-entropy with square-root
inverse-frequency class weights. Frozen: XLM-R embeddings + layers 1–8, ResNet stem + layers 1–3.

#### Evaluation
Accuracy, macro & weighted precision / recall / F1, per-class report, confusion matrix, 95 %
bootstrap confidence intervals, paired bootstrap tests between models, per-meme latency, and a
modality ablation (neutral image / removed text) for the multimodal models.
""")
    with limitations:
        st.markdown("""
#### Limitations & responsible use
- Trained on ~3 k Bengali memes from a few sources: it will not generalise reliably to other
  languages, platforms or time periods. Hindi/Tamil/etc. inputs run but are out-of-distribution.
- Labels are subjective; sarcasm and cultural context are frequently misread.
- Some memes in the dataset contain crude, sexual or offensive humour.
- Do not use for moderation, profiling or any decision about individuals without human review.
""")


def page_about() -> None:
    hero("Built to explore the whole meme", "A research project on language, visual context, and explainable sentiment analysis.")
    with st.container(border=True):
        st.markdown("### Why multimodal sentiment?")
        st.write("Indian social media is multilingual, code-mixed, and visual. A cheerful caption over a grim image "
                 "can be sarcasm. IndicSent-M studies whether combining those signals improves sentiment "
                 "classification over analyzing text or images alone.")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.markdown("### Research objectives")
        st.markdown("1. Build text-only, image-only, and multimodal classifiers.\n"
                    "2. Compare them on identical splits with uncertainty estimates.\n"
                    "3. Study concatenation and cross-attention fusion.\n"
                    "4. Explain influential words and image regions, with their limitations.")
    with right, st.container(border=True):
        st.markdown("### What is implemented")
        st.markdown("- Data pipeline with grouped splits\n- Four model architectures\n"
                    "- Evaluation, bootstrap statistics, and modality ablation\n"
                    "- Occlusion, Integrated Gradients, and Grad-CAM\n- Streamlit demo, REST API, tests, and Docker")
        st.caption("Trained weights and the original dataset are not bundled with the repository.")
    st.markdown("### Technology")
    for group, detail in [
        ("Modeling", "PyTorch · XLM-RoBERTa · ResNet-50 · Hugging Face Transformers"),
        ("Evaluation", "scikit-learn · pandas · Matplotlib · Seaborn"),
        ("Application", "Python · Streamlit · FastAPI · Docker · pytest"),
    ]:
        st.markdown(f"**{group}**  \n{detail}")
    with st.expander("Future research directions"):
        st.write("Hinglish data (Memotion 3, registration required), OCR, CLIP-style vision–language encoders, "
                 "multiple training seeds, confidence calibration, sarcasm-aware training, "
                 "and human evaluation of explanations.")


PAGES = {"Home": page_home, "Analyze": page_analyze, "Model Comparison": page_comparison,
         "Dataset & Methodology": page_dataset, "About": page_about}

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-mark" aria-hidden="true">◈</span>'
                '<div><strong>IndicSent-M</strong><small>Multimodal sentiment studio</small></div></div>', unsafe_allow_html=True)
    st.caption("WORKSPACE")
    choice = st.radio("Navigate", list(PAGES), key="navigation", label_visibility="collapsed")
    st.divider()
    try:
        avail = get_predictor().available()
        st.markdown("**Model availability**")
        st.caption(f"{len(avail)} of {len(EXP_NAMES)} trained models available")
        st.progress(len(avail) / len(EXP_NAMES))
        with st.expander("Model & runtime details"):
            for e, name in EXP_NAMES.items():
                status = f"Available · validation macro-F1 {avail[e]['val_macro_f1']:.3f}" if e in avail else "Not trained"
                st.markdown(f'<div class="model-row"><strong>{html.escape(name)}</strong>'
                            f'<span class="small">{status}</span></div>', unsafe_allow_html=True)
            st.caption(f"Device: {get_predictor().device}")
    except Exception as e:
        st.error(f"Model runtime failed to initialise: {e}")
    st.markdown('<div class="footer">Bengali memes · MemoSen<br>Research demo · 3 sentiment classes</div>', unsafe_allow_html=True)

PAGES[choice]()
st.markdown('<div class="footer">IndicSent-M · Multimodal sentiment research<br>'
            'Predictions and explanations reflect model behavior. Cultural context and sarcasm can change meaning.</div>',
            unsafe_allow_html=True)
