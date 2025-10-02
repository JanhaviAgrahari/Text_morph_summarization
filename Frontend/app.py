import streamlit as st
import requests
import re
import os
import csv
import json
from datetime import datetime, timezone
from io import BytesIO
from typing import Optional, Tuple, Any

# --- Optional dependency imports (fail gracefully if missing) ---
try:
    import PyPDF2  # PDF parsing
except ImportError:  # pragma: no cover
    PyPDF2 = None
try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # Fallback PDF text extractor
except ImportError:  # pragma: no cover
    pdfminer_extract_text = None
try:
    from docx import Document  # DOCX parsing
except ImportError:  # pragma: no cover
    Document = None
try:
    import textstat  # Readability metrics
except ImportError:  # pragma: no cover
    textstat = None
try:
    import matplotlib.pyplot as plt  # Charts
except ImportError:  # pragma: no cover
    plt = None
try:
    import numpy as np  # For grouped bar placement (optional)
except ImportError:  # pragma: no cover
    np = None

# --- Constants & configuration ---
# Allow overriding backend URL via environment variable BACKEND_URL
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
API_URL = BACKEND_URL  # Alias for legacy references
ROUGE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rouge_score"))
os.makedirs(ROUGE_OUTPUT_DIR, exist_ok=True)

# --- Unified Translation Utilities (improved quality) ---
# We centralize translation so all tabs use the same logic and higher quality models (IndicTrans2) when available.
@st.cache_resource(show_spinner=False)
def _load_hf_translator(direction: str):
    """Load an IndicTrans2 (preferred) or Opus-MT model for a given direction.

    direction examples: 'en-hi', 'hi-en'. Falls back silently if model not available.
    Returns (tokenizer, model, device, model_name) or (None, None, -1, None) on failure.
    """
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        # Prefer IndicTrans2 (not bilingual pairs for every direction, but en-hi & hi-en exist)
        preferred = f"ai4bharat/indictrans2-{direction}"
        model_name = preferred
        try:
            tok = AutoTokenizer.from_pretrained(preferred, trust_remote_code=True)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(preferred, trust_remote_code=True)
        except Exception:
            # Fallback to opus-mt
            opus = f"Helsinki-NLP/opus-mt-{direction}"
            tok = AutoTokenizer.from_pretrained(opus)
            mdl = AutoModelForSeq2SeqLM.from_pretrained(opus)
            model_name = opus
        device = 0 if getattr(__import__('torch'), 'cuda').is_available() else -1  # type: ignore
        return tok, mdl, device, model_name
    except Exception:
        return None, None, -1, None

def _segment_paragraphs(text: str) -> list[str]:
    # Keep paragraph boundaries – split on 2+ newlines
    paras = [p.strip() for p in re.split(r"\n{2,}", text.replace("\r\n", "\n")) if p.strip()]
    return paras if paras else [text.strip()]

def _segment_sentences(paragraph: str) -> list[str]:
    # Lightweight sentence splitter – avoids downloading NLTK resources
    sents = re.split(r'(?<=[.!?])\s+', paragraph.strip())
    return [s for s in sents if s.strip()]

@st.cache_data(show_spinner=False, ttl=3600)
def _translate_with_hf_cached(chunks: tuple[str, ...], direction: str) -> list[str]:
    tok, mdl, device, _ = _load_hf_translator(direction)
    if tok is None or mdl is None:
        return []
    out: list[str] = []
    import torch
    for sent in chunks:
        if not sent.strip():
            out.append("")
            continue
        batch = tok(sent, return_tensors='pt', truncation=True, max_length=512)
        if device >= 0:
            batch = {k: v.to(device) for k, v in batch.items()}
            mdl.to(device)
        gen = mdl.generate(**batch, max_length=512, num_beams=4)
        decoded = tok.batch_decode(gen, skip_special_tokens=True)[0]
        out.append(decoded.strip())
    return out

@st.cache_data(show_spinner=False, ttl=1800)
def _google_translate_bulk(chunks: tuple[str, ...], src: str, tgt: str) -> list[str]:
    # Fallback using googletrans (may be slower / rate-limited)
    try:
        from googletrans import Translator
        tr = Translator()
        results = []
        for c in chunks:
            if not c.strip():
                results.append("")
                continue
            try:
                results.append(tr.translate(c, src=src, dest=tgt).text)
            except Exception:
                results.append(c)  # graceful fallback
        return results
    except Exception:
        return list(chunks)

def translate_text_full(text: str, src: str = 'en', tgt: str = 'hi') -> str:
    """High quality translation preserving full content & structure.

    Strategy:
        1. Split paragraphs, then sentences.
        2. Try IndicTrans2 / Opus-MT (cached) sentence-wise.
        3. If that fails or returns empty, fallback to googletrans.
        4. Reconstruct paragraphs with double newlines; ensure no truncation.
    """
    if not text or not text.strip() or src == tgt:
        return text
    direction = f"{src}-{tgt}"
    paragraphs = _segment_paragraphs(text)
    translated_paras: list[str] = []
    for para in paragraphs:
        sentences = _segment_sentences(para)
        tpl = tuple(sentences)
        hf_out = _translate_with_hf_cached(tpl, direction)
        if not hf_out or len([o for o in hf_out if o.strip()]) == 0:
            hf_out = _google_translate_bulk(tpl, src, tgt)
        # If counts mismatch, pad
        if len(hf_out) != len(sentences):
            hf_out = hf_out + ["" for _ in range(len(sentences) - len(hf_out))]
        translated_paras.append(" ".join(hf_out).strip())
    return "\n\n".join(translated_paras).strip()


# --- Page configuration & global styling (UI enhancement) ---
if "_page_config_set" not in st.session_state:
        st.set_page_config(
                page_title="TextMorph – AI Text Analytics",
                page_icon="📚",
                layout="wide",
                initial_sidebar_state="expanded",
        )
        st.session_state["_page_config_set"] = True

# Simple light/dark toggle (client side simulation)
theme_choice = st.sidebar.radio("Theme", ["Light", "Dark"], index=0, horizontal=True)
with st.sidebar:
    # Backend health indicator
    try:
        r = requests.get(f"{BACKEND_URL}/ping", timeout=3)
        if r.ok:
            st.caption(f"Backend: ✅ Connected ({BACKEND_URL})")
        else:
            st.caption(f"Backend: ⚠️ Unreachable ({BACKEND_URL})")
    except Exception:
        st.caption(f"Backend: ❌ Not reachable ({BACKEND_URL})")
dark = theme_choice == "Dark"

color_block = f"""
:root {{
    --bg-panel: {'#111827' if dark else '#ffffff'};
    --bg-alt: {'#1f2937' if dark else '#f8fafc'};
    --border-color: {'#374151' if dark else '#e2e8f0'};
    --text-color: {'#f1f5f9' if dark else '#1e293b'};
    --text-muted: {'#94a3b8' if dark else '#64748b'};
    --accent: #6366f1;
    --accent-grad-start: #6366f1;
    --accent-grad-end: #8b5cf6;
    --danger: #ef4444;
    --warn: #f59e0b;
    --success: #10b981;
}}
"""

body_bg = '#0f172a' if dark else '#f1f5f9'

base_css = """
<style>
html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }
BLOCK_COLOR_VARS
.block-container { padding-top: 1.2rem; padding-bottom: 2.5rem; }
.tm-header { background: linear-gradient(135deg,var(--accent-grad-start),var(--accent-grad-end)); padding: 1.8rem 2rem; border-radius: 18px; color: #fff; box-shadow: 0 6px 24px rgba(0,0,0,.18); margin-bottom: 1.2rem; position: relative; overflow: hidden; }
.tm-header:before { content: ""; position:absolute; inset:0; background: radial-gradient(circle at 25% 15%,rgba(255,255,255,.35),transparent 60%); }
.tm-header h1 { font-size: 2.2rem; line-height:1.15; margin:0 0 .4rem; font-weight:700; }
.tm-header p { margin:0; font-size:0.95rem; opacity:.9; font-weight:500; }
.tm-card { background: var(--bg-panel); border:1px solid var(--border-color); padding:1.2rem 1.15rem; border-radius:14px; box-shadow:0 2px 6px rgba(0,0,0,.06); transition:.18s; }
.tm-card:hover { box-shadow:0 4px 14px rgba(0,0,0,.10); }
.metric-card { background: linear-gradient(135deg,#6366f1,#8b5cf6); color:#fff; padding:1rem 0.9rem; border-radius:14px; position:relative; overflow:hidden; min-height:140px; }
.metric-green { background: linear-gradient(135deg,#10b981,#059669); }
.metric-yellow { background: linear-gradient(135deg,#f59e0b,#d97706); }
.metric-red { background: linear-gradient(135deg,#ef4444,#b91c1c); }
.metric-value { font-size:2.3rem; font-weight:700; letter-spacing:-1px; }
.metric-label { font-size:.85rem; text-transform:uppercase; opacity:.85; letter-spacing:.5px; }
div[data-baseweb="tab-list"] button { font-weight:600; }
textarea { font-family: 'JetBrains Mono','Consolas',monospace; font-size:0.85rem; line-height:1.35; }
.tm-footer { margin-top:2.5rem; padding:1.4rem 1rem; text-align:center; font-size:.75rem; color:var(--text-muted); border-top:1px solid var(--border-color); }
body { background: BODY_BG; color: var(--text-color); }
.stMarkdown, .stText, .stRadio, .stSelectbox, .stMultiSelect, .stSlider, .stButton, label, p, span, h1,h2,h3,h4,h5,h6 { color: var(--text-color)!important; }
section[data-testid="stSidebar"] > div { background: var(--bg-panel); }
section[data-testid="stSidebar"] * { color: var(--text-color)!important; }
hr { border-color: var(--border-color); }
/* Buttons */
.stButton>button { background: linear-gradient(135deg,var(--accent-grad-start),var(--accent-grad-end)); color:#fff; border:0; border-radius:10px; padding:0.5rem 0.9rem; font-weight:600; box-shadow:0 2px 8px rgba(99,102,241,.25); }
.stButton>button:hover { filter: brightness(1.05); box-shadow:0 4px 12px rgba(99,102,241,.35); }
/* Pills */
.tm-pills { margin:6px 0 2px; }
.tm-pill { display:inline-block; padding:4px 10px; border-radius:999px; font-size:.78rem; font-weight:600; margin:0 6px 6px 0; border:1px solid var(--border-color); background:var(--bg-alt); }
.tm-pill.acc { background:rgba(99,102,241,.12); color:#6366f1; border-color:rgba(99,102,241,.35); }
.tm-pill.ok { background:rgba(16,185,129,.12); color:#10b981; border-color:rgba(16,185,129,.35); }
.tm-pill.warn { background:rgba(245,158,11,.12); color:#f59e0b; border-color:rgba(245,158,11,.35); }
.tm-pill.info { background:rgba(59,130,246,.12); color:#3b82f6; border-color:rgba(59,130,246,.35); }
/* Cards */
.tm-card-border { border:1px dashed var(--border-color); border-radius:12px; padding:12px; }
/* File uploader */
div[data-testid="stFileUploader"] { background: var(--bg-panel); border:1px dashed var(--border-color); padding:10px; border-radius:10px; }
</style>
"""

css = base_css.replace("BLOCK_COLOR_VARS", color_block).replace("BODY_BG", body_bg)
st.markdown(css, unsafe_allow_html=True)

# Header (sticky visual identity)
st.markdown(
        """
<div class='tm-header'>
    <h1>📚 TextMorph</h1>
    <p>AI-powered readability, summarization & paraphrasing toolkit</p>
</div>
""",
        unsafe_allow_html=True,
)

# --- Small UI helpers ---
def _pill(label: str, kind: str = "acc") -> str:
    return f"<span class='tm-pill {kind}'>{label}</span>"

def show_pills(items: list[tuple[str, str]]):
    html = "<div class='tm-pills'>" + " ".join([_pill(text, kind) for text, kind in items]) + "</div>"
    st.markdown(html, unsafe_allow_html=True)

# --- Simple API helpers ---
def api_post(path: str, payload: dict, token: Optional[str] = None, timeout: int = 60) -> Tuple[bool, Any]:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{BACKEND_URL}{path}"
    try:
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        if r.ok:
            return True, r.json()
        try:
            return False, r.json().get("detail", r.text)
        except Exception:
            return False, r.text
    except Exception as e:  # pragma: no cover
        return False, str(e)

def api_get(path: str, token: Optional[str] = None) -> Tuple[bool, Any]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    url = f"{BACKEND_URL}{path}"
    try:
        r = requests.get(url, headers=headers, timeout=30)
        if r.ok:
            return True, r.json()
        try:
            return False, r.json().get("detail", r.text)
        except Exception:
            return False, r.text
    except Exception as e:  # pragma: no cover
        return False, str(e)

# Basic password strength checks (client-side only)
def _valid_password(p: str) -> tuple[bool, str | None]:
    if len(p) < 8:
        return False, "Password must be at least 8 characters long."
    if not re.search(r"[A-Z]", p):
        return False, "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", p):
        return False, "Password must include at least one lowercase letter."
    if not re.search(r"\d", p):
        return False, "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", p):
        return False, "Password must include at least one special character."
    return True, None


# Prefill reset token from URL if present
token_prefill = ""
try:
    qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
    if qp and "token" in qp:
        token_prefill = qp["token"] if isinstance(qp["token"], str) else (qp["token"][0] if qp["token"] else "")
except Exception:
    pass


# Create the main app tabs with icons (UI enhancement)
tabs = st.tabs([
    "🔐 Sign in",
    "🆕 Register",
    "🔄 Reset",
    "👤 Profile",
    "📊 Readability",
    "📝 Summarization",
    "✏️ Paraphrasing",
    "� Fine-tuned Summarization",
    "📚 Summarization Dataset Eval",
    "✏️ Fine-tuned Paraphrasing",
    "📚 Paraphrase Dataset Eval",
    "�📚 History",  # moved to last
])

# Sign in tab
with tabs[0]:
    st.markdown("""
        <div style="text-align:center;margin-bottom:20px">
            <img src="https://cdn-icons-png.flaticon.com/512/6681/6681204.png" width="60">
            <h3 style="font-weight:600;color:#334155">Sign in to Your Account</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="user@example.com")
        password = st.text_input("Password", type="password")
        sign_in = st.form_submit_button("Sign in", use_container_width=True)
    
    if sign_in:
        # Basic client-side validation for email/password
        if not email:
            st.error("Please enter your email.")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            st.error("Please enter a valid email address.")
        elif not password:
            st.error("Please enter your password.")
        else:
            # Authenticate via backend /login endpoint
            ok, data = api_post("/login", {"email": email, "password": password})
            if ok:
                st.success("Login successful")
                # Persist auth context
                st.session_state["token"] = data.get("access_token")
                st.session_state["logged_in"] = True
                st.session_state["user_email"] = email
                # Optionally cache /me
                try:
                    ok_me, me = api_get("/me", token=st.session_state["token"])  # type: ignore[index]
                    if ok_me and isinstance(me, dict):
                        st.session_state["me"] = me
                except Exception:
                    pass
                # Rerun to propagate session state into other tabs immediately
                st.rerun()
            else:
                st.error(data)


# Register tab
with tabs[1]:
    st.markdown("""
        <div style="text-align:center;margin-bottom:20px">
            <img src="https://cdn-icons-png.flaticon.com/512/3534/3534139.png" width="60">
            <h3 style="font-weight:600;color:#334155">Create New Account</h3>
        </div>
    """, unsafe_allow_html=True)
    
    with st.form("register_form"):
        reg_email = st.text_input("Email", placeholder="user@example.com")
        reg_password = st.text_input("Password", type="password")
        register_clicked = st.form_submit_button("Create account", use_container_width=True)
    
    if register_clicked:
    # Validate email and password before sending to backend
        if not reg_email:
            st.error("Please enter your email.")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", reg_email):
            st.error("Please enter a valid email address.")
        elif not reg_password:
            st.error("Please enter a password.")
        else:
            ok_pw, err = _valid_password(reg_password)
            if not ok_pw:
                st.error(err)
            else:
        # Create account via backend /register endpoint
                ok, data = api_post("/register", {"email": reg_email, "password": reg_password})
                if ok:
                    st.success("Account created. You can sign in now.")
                else:
                    st.error(data)


# Forgot password tab
with tabs[2]:
    st.markdown("""
        <div style="text-align:center;margin-bottom:20px">
            <img src="https://cdn-icons-png.flaticon.com/512/6357/6357042.png" width="60">
            <h3 style="font-weight:600;color:#334155">Reset Your Password</h3>
        </div>
    """, unsafe_allow_html=True)
    
    st.caption("Request a reset link to be sent to your email")
    with st.form("forgot_form"):
        fp_email = st.text_input("Email", placeholder="user@example.com")
        get_link = st.form_submit_button("Get reset link", use_container_width=True)
    
    if get_link:
        if not fp_email or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", fp_email):
            st.error("Please enter a valid email address.")
        else:
            # Request password reset email via /forgot-password
            ok, data = api_post("/forgot-password", {"email": fp_email})
            if ok:
                st.info(data.get("message", "If the email exists, a reset link has been sent."))
            else:
                st.error(data)

    st.divider()
    st.caption("Enter your reset token and new password")
    with st.form("reset_form"):
        token = st.text_input("Token", value=token_prefill)
        new_pw = st.text_input("New password", type="password")
        reset_clicked = st.form_submit_button("Reset password", use_container_width=True)
    
    if reset_clicked:
        if not token:
            st.error("Please enter the reset token.")
        else:
            ok_pw, err = _valid_password(new_pw)
            if not ok_pw:
                st.error(err)
            else:
                # Submit new password with token to /reset-password
                ok, data = api_post("/reset-password", {"token": token, "new_password": new_pw})
                if ok:
                    st.success("Password has been reset successfully.")
                    # Clear token from the URL
                    try:
                        if hasattr(st, "query_params"):
                            st.query_params.clear()
                        else:
                            st.experimental_set_query_params()
                    except Exception:
                        pass
                else:
                    st.error(data)


# Profile tab
with tabs[3]:
    token = st.session_state.get("token")
    if not token:
        st.info("Please sign in to access your profile")
    else:
    # Fetch current user profile via /me using bearer token
        ok_me, me = api_get("/me", token=token)
        if ok_me and isinstance(me, dict):
            st.markdown("""
                <div style="text-align:center;margin-bottom:20px">
                    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" width="60">
                    <h3 style="font-weight:600;color:#334155">User Profile</h3>
                </div>
            """, unsafe_allow_html=True)
            
            st.info(f"Signed in as: {me.get('email')}")
            
            with st.form("profile_form"):
                col1, col2 = st.columns(2)
                with col1:
                    name = st.text_input("Name", placeholder="Your name")
                with col2:
                    age_group = st.selectbox("Age group", ["18-25", "26-35", "36-50", "50+"])
                
                language = st.radio("Language preference", ["English", "हिंदी"], horizontal=True)
                save_profile = st.form_submit_button("Save profile", use_container_width=True)
            
            if save_profile:
                # Save profile preferences to backend
                ok, data = api_post(
                    "/profile",
                    {"name": name, "age_group": age_group, "language": language},
                    token=token,
                )
                if ok:
                    st.success("Profile updated successfully!")
                else:
                    st.error(data)


# Readability tab (visible for signed-in users)
with tabs[4]:
    token = st.session_state.get("token")
    if not token:
        st.markdown("""
            <div style="text-align:center;padding:40px 0;">
                <img src="https://cdn-icons-png.flaticon.com/512/6195/6195696.png" width="100" style="opacity:0.7;">
                <h3 style="margin-top:20px;font-weight:500;color:#64748b;">Please sign in to access readability analysis</h3>
                <p style="color:#94a3b8;margin-top:10px;">Sign in to analyze the readability of your documents</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="text-align:center;margin-bottom:20px">
                <img src="https://cdn-icons-png.flaticon.com/512/4359/4359848.png" width="60">
                <h3 style="margin-top:8px;font-weight:600;color:#334155">Document Readability Analysis</h3>
                <p style="color:#64748b;font-size:14px;">Upload a document to analyze its readability metrics</p>
            </div>
        """, unsafe_allow_html=True)

        # Upload with simplified container
        with st.container():
            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                # Centralized file uploader for supported document types
                up = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"])
                if not up:
                    st.caption("Supported formats: .txt, .pdf, .docx")

        def _extract_text(file) -> tuple[Optional[str], Optional[str]]:
            if file is None:
                return None, None
            name = (file.name or "").lower()
            data = file.read()
            file.seek(0)
            try:
                if name.endswith(".txt"):
                    return data.decode(errors="ignore"), None
                if name.endswith(".pdf") and PyPDF2 is not None:
                    reader = PyPDF2.PdfReader(BytesIO(data))
                    pages = [p.extract_text() or "" for p in reader.pages]
                    joined = "\n".join(pages)
                    if not joined and pdfminer_extract_text is not None:
                        try:
                            joined = pdfminer_extract_text(BytesIO(data)) or ""
                        except Exception:
                            pass
                    # If still empty, return informative message
                    if not joined:
                        return "", "No extractable text found in PDF (it may be scanned images)."
                    return joined, None
                if name.endswith(".docx") and Document is not None:
                    doc = Document(BytesIO(data))
                    return "\n".join([p.text for p in doc.paragraphs]), None
            except Exception:
                return None, "Unable to parse the file."
            return None, "Unsupported file type."

    # Extract raw text from the uploaded file (falls back for tricky PDFs)
        text, read_err = _extract_text(up)

        if up and read_err and (text is None or text == ""):
            # Differentiated error for PDFs with no text
            if (up.name or "").lower().endswith(".pdf") and "No extractable text" in read_err:
                st.warning(read_err)
                st.info("Try a text-based PDF or upload a .txt/.docx file.")
            else:
                st.error("Unable to read the file. Supported: .txt, .pdf, .docx")

        if up and text:
            if textstat is None:
                st.error("textstat is not installed. Please add 'textstat' to requirements and reinstall.")
            else:
                # Compute readability metrics from extracted text
                try:
                    fk_ease = round(float(textstat.flesch_reading_ease(text)), 1)
                except Exception:
                    fk_ease = float("nan")
                try:
                    gunning = round(float(textstat.gunning_fog(text)), 1)
                except Exception:
                    gunning = float("nan")
                try:
                    smog = round(float(textstat.smog_index(text)), 1)
                except Exception:
                    smog = float("nan")

                # Present key metrics as visual cards with simple indicators
                st.markdown("<h4 style='margin:30px 0 20px;font-weight:600;color:#1e293b;'>Readability Metrics</h4>", 
                           unsafe_allow_html=True)
                
                # Metrics display matching the image
                st.subheader("Readability Metrics")
                
                c1, c2, c3 = st.columns(3)
                with c1:
                    fk_class = "easy" if fk_ease >= 70 else "medium" if fk_ease >= 50 else "hard"
                    st.markdown(
                        f'''<div class="metric-card metric-green">
                            <div class="metric-value">{fk_ease}</div>
                            <div class="metric-label">Flesch-Kincaid</div>
                            <div class="indicator {fk_class}"></div>
                        </div>''', 
                        unsafe_allow_html=True
                    )
                    rating = "Moderate reading level" if fk_ease >= 50 and fk_ease < 70 else "Easy reading level" if fk_ease >= 70 else "Complex reading level"
                    st.caption(f"{rating}")
                    
                with c2:
                    gun_class = "easy" if gunning <= 10 else "medium" if gunning <= 14 else "hard"
                    st.markdown(
                        f'''<div class="metric-card metric-yellow">
                            <div class="metric-value">{gunning}</div>
                            <div class="metric-label">Gunning Fog</div>
                            <div class="indicator {gun_class}"></div>
                        </div>''', 
                        unsafe_allow_html=True
                    )
                    level = "High school level" if gunning <= 14 and gunning > 10 else "General audience" if gunning <= 10 else "College level"
                    st.caption(f"{level}")
                    
                with c3:
                    smog_class = "easy" if smog <= 9 else "medium" if smog <= 12 else "hard"
                    st.markdown(
                        f'''<div class="metric-card metric-red">
                            <div class="metric-value">{smog}</div>
                            <div class="metric-label">SMOG Index</div>
                            <div class="indicator {smog_class}"></div>
                        </div>''', 
                        unsafe_allow_html=True
                    )
                    level = "High school level" if smog <= 12 and smog > 9 else "Junior high level" if smog <= 9 else "College level"
                    st.caption(f"{level}")

                # Calculate readability distribution based on established score interpretations
                # Flesch Reading Ease: 90-100 Very Easy, 80-89 Easy, 70-79 Fairly Easy, 60-69 Standard, 
                #                      50-59 Fairly Difficult, 30-49 Difficult, 0-29 Very Difficult
                # Gunning Fog: <8 Universal, 8-10 Plain English, 11-12 Fairly Readable, 
                #              13-16 Difficult, 17-20 Academic, >20 Specialized
                # SMOG Index: <6 Universal, 7-9 Junior High, 10-12 High School, 
                #             13-16 College, 17+ Graduate
                
                # Heuristic: higher beginner score means simpler text overall
                beg = 0
                if fk_ease >= 70:  # High Flesch score = easy = beginner-friendly
                    beg += min(int(fk_ease * 0.8), 80)  # Cap at 80
                if gunning <= 10:  # Low Gunning = easier
                    beg += max(0, 40 - (gunning * 4))
                if smog <= 9:  # Low SMOG = easier
                    beg += max(0, 40 - (smog * 4))
                beg = min(100, max(10, beg))  # Min 10, max 100
                
                # Heuristic: intermediate band covers mid-range scores across metrics
                inter = 0
                if 50 <= fk_ease < 70:  # Mid Flesch = intermediate
                    inter += 50
                elif 30 <= fk_ease < 50:  # Lower mid = some intermediate
                    inter += 30
                if 11 <= gunning <= 14:  # Mid Gunning = intermediate
                    inter += 40
                elif gunning <= 10:  # Low Gunning = some intermediate
                    inter += 20
                if 9 <= smog <= 12:  # Mid SMOG = intermediate
                    inter += 40
                elif smog < 9:  # Low SMOG = some intermediate
                    inter += 20
                inter = min(100, max(10, inter))  # Min 10, max 100
                
                # Heuristic: advanced increases when readability becomes harder
                adv = 0
                if fk_ease < 50:  # Low Flesch = advanced
                    adv += max(0, 70 - fk_ease)
                if gunning > 12:  # High Gunning = advanced
                    adv += min(70, gunning * 4)
                if smog > 12:  # High SMOG = advanced
                    adv += min(70, smog * 4)
                adv = min(100, max(10, adv))  # Min 10, max 100

                try:
                    import pandas as pd
                    st.subheader("Reading Level Distribution")

                    # Simplified description based on score distribution
                    max_score = max(beg, inter, adv)
                    if max_score == beg:
                        st.info("This text is suitable for a general audience.")
                    elif max_score == inter:
                        st.info("This text requires moderate reading skills, typical of high school level.")
                    else:
                        st.info("This text contains advanced language patterns common in academic content.")

                    if plt is None:
                        st.warning("matplotlib not installed; run 'pip install matplotlib' to view distribution bar chart.")
                    else:
                        levels = ["Beginner", "Intermediate", "Advanced"]
                        scores = [beg, inter, adv]
                        colors = ["#10b981", "#f59e0b", "#ef4444"]
                        fig, ax = plt.subplots(figsize=(5.5, 3.0))
                        bars = ax.bar(levels, scores, color=colors)
                        ax.set_ylim(0, 100)
                        ax.set_ylabel("Score")
                        ax.set_title("Reading Level Distribution")
                        for b, val in zip(bars, scores):
                            ax.text(b.get_x() + b.get_width()/2, val + 1, f"{val}", ha='center', va='bottom', fontsize=8)
                        st.pyplot(fig, clear_figure=True)
                        st.caption("Higher scores indicate greater prevalence of that reading level")
                except Exception as e:
                    st.error(f"Could not generate chart: {e}")

                # Simplified feature info
                st.subheader("Document Analysis Features")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("⚡ **Real-time readability scoring**")
                    st.markdown("📊 **Visual complexity indicators**")
                with col2:
                    st.markdown("📝 **Multiple file format support**")
                    st.markdown("🔍 **Comprehensive text metrics**")


with tabs[5]:  # Tab 5 - Summarization
    token = st.session_state.get("token")
    if not token:
        st.markdown("""
            <div style='text-align:center;padding:50px 0;'>
                <h3 style='color:#334155;margin-bottom:10px;'>Sign in required</h3>
                <p style='color:#64748b;'>Please sign in to access the summarization feature.</p>
            </div>
        """, unsafe_allow_html=True)
        # Early return pattern – don't render summarization UI when not authenticated
        st.stop()
    st.subheader("Summarize Text or PDF")

    col1, col2 = st.columns(2)

    with col1:
        input_type = st.radio("Choose input type:", ["Plain Text", "PDF File"])
        model_choice = st.selectbox(
            "Select Model",
            options=["pegasus", "bart", "flan-t5"],
            index=0
        )
        summary_length = st.selectbox(
            "Summary Length",
            options=["short", "medium", "long"],
            index=1
        )

    with col2:
        st.write("Instructions:")
        st.markdown("- Paste text or upload a PDF.\n- Choose a model and summary length.\n- Click 'Generate Summary'.")

    text_input = ""
    pdf_file = None

    if input_type == "Plain Text":
        text_input = st.text_area("Paste your text here", height=200)
    else:
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])

    # Reference summary input for ROUGE (optional)
    reference_input = st.text_area("Reference Summary (optional for ROUGE evaluation)", height=150, help="Paste a human-written or gold summary here to compute ROUGE metrics.")

    # Initialize session state for last summary
    if 'last_summary' not in st.session_state:
        st.session_state['last_summary'] = None
        st.session_state['last_model'] = None
        st.session_state['last_length'] = None

    generate_clicked = st.button("Generate Summary")

    if generate_clicked:
        if (input_type == "Plain Text" and not text_input.strip()) or (input_type == "PDF File" and not pdf_file):
            st.error("Please provide text or upload a PDF.")
        else:
            with st.spinner("Generating summary..."):
                # Include user email if available for history tracking
                form_data = {
                    "model_choice": model_choice,
                    "summary_length": summary_length,
                    "input_type": "pdf" if input_type == "PDF File" else "text",
                    "text_input": text_input
                }
                # Add user email for history if present in session
                if st.session_state.get("user_email"):
                    form_data["user_email"] = st.session_state["user_email"]

                files = None
                if pdf_file:
                    files = {"pdf_file": (pdf_file.name, pdf_file, "application/pdf")}

                try:
                    response = requests.post(
                        f"{BACKEND_URL}/summarize/",
                        data=form_data,
                        files=files
                    )
                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success("Summary generated successfully!")

                        # Prepare original text for side-by-side view
                        original_text_display = ""
                        if input_type == "Plain Text":
                            original_text_display = text_input.strip()
                        else:
                            # Try to extract PDF text (lightweight) for display
                            try:
                                if pdf_file is not None and PyPDF2 is not None:
                                    pdf_file.seek(0)
                                    reader = PyPDF2.PdfReader(pdf_file)
                                    pages = [p.extract_text() or "" for p in reader.pages]
                                    original_text_display = "\n".join(pages).strip()
                                    # --- Normalize PDF text so it shows as readable paragraphs ---
                                    try:
                                        # Keep paragraph breaks (double newlines) but collapse single newlines to spaces
                                        t = original_text_display.replace("\r\n", "\n")
                                        # Join hyphenated line breaks like "exam-\nple" -> "example"
                                        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
                                        # Reduce 3+ newlines to 2 (standard paragraph break)
                                        t = re.sub(r"\n{3,}", "\n\n", t)
                                        # Replace remaining single newlines with a space (preserve double newlines)
                                        t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
                                        # Normalize extra spaces
                                        t = re.sub(r"[ \t]{2,}", " ", t)
                                        # Remove spaces before punctuation
                                        t = re.sub(r" +([.,;:!?])", r"\1", t)
                                        original_text_display = t.strip()
                                    except Exception:
                                        # If normalization fails for any reason, fall back to raw extracted text
                                        pass
                            except Exception:
                                original_text_display = "(Unable to extract PDF text for preview)"

                        summary_text_display = result["summary"].strip()

                        def _word_count(t: str) -> int:
                            return len([w for w in re.findall(r"\w+", t)]) if t else 0

                        wc_orig = _word_count(original_text_display)
                        wc_sum = _word_count(summary_text_display)
                        compression = (1 - (wc_sum / wc_orig)) * 100 if wc_orig > 0 else 0

                        col_orig, col_sum = st.columns(2)
                        with col_orig:
                            st.markdown("#### Original Text")
                            st.caption(f"{wc_orig} words")
                            st.text_area(label="Original", value=original_text_display[:15000], height=260, key="orig_view")
                        with col_sum:
                            st.markdown("#### Summary")
                            st.caption(f"{wc_sum} words")
                            st.text_area(label="Summary", value=summary_text_display, height=260, key="sum_view")
                            # Quick metrics row
                            metric_cols = st.columns(2)
                            with metric_cols[0]:
                                if wc_orig > 0:
                                    st.markdown(f"**Compression:** {compression:.0f}%")
                                else:
                                    st.markdown("**Compression:** n/a")
                            with metric_cols[1]:
                                # Placeholder for ROUGE F1 after evaluation (updated later)
                                st.markdown("**ROUGE F1:** _evaluate to view_")

                        # Optional feature list / hint
                        with st.container():
                            st.markdown("<div style='margin-top:14px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#f1f5f9'>" \
                                        "<strong>Key Features</strong><ul style='margin-top:6px;'>" \
                                        "<li>Multiple summary length options</li>" \
                                        "<li>Advanced models (Pegasus, BART, FLAN-T5)</li>" \
                                        "<li>Side-by-side comparison view</li>" \
                                        "<li>Quality metrics (ROUGE, compression)</li>" \
                                        "<li>Edit text areas and click Generate again to refine</li>" \
                                        "</ul></div>", unsafe_allow_html=True)
                        st.session_state['last_summary'] = result["summary"]
                        st.session_state['last_model'] = model_choice
                        st.session_state['last_length'] = summary_length

                        # Auto-evaluate if reference provided up front
                        if reference_input.strip():
                            with st.spinner("Evaluating ROUGE metrics..."):
                                try:
                                    rouge_payload = {
                                        "reference": reference_input.strip(),
                                        "candidate": result["summary"]
                                    }
                                    rouge_resp = requests.post(f"{BACKEND_URL}/evaluate/rouge", json=rouge_payload, timeout=120)
                                    rouge_json = rouge_resp.json()
                                    if 'scores' in rouge_json:
                                        st.markdown("### ROUGE Evaluation")
                                        scores = rouge_json['scores']
                                        try:
                                            import pandas as pd
                                            rows = []
                                            for metric_name, vals in scores.items():
                                                rows.append({"Metric": metric_name.upper(), "Precision": vals['precision'], "Recall": vals['recall'], "F1": vals['f1']})
                                            df = pd.DataFrame(rows)
                                            st.dataframe(df.style.format({"Precision": "{:.4f}", "Recall": "{:.4f}", "F1": "{:.4f}"}))

                                            if plt is None:
                                                st.warning("matplotlib not installed; run 'pip install matplotlib' to view charts.")
                                            else:
                                                # Grouped precision/recall/F1 chart
                                                fig2, ax2 = plt.subplots(figsize=(6, 3.2))
                                                metrics_order = list(scores.keys())
                                                x = np.arange(len(metrics_order)) if np is not None else list(range(len(metrics_order)))
                                                width = 0.25
                                                precisions = [scores[m]['precision'] for m in metrics_order]
                                                recalls = [scores[m]['recall'] for m in metrics_order]
                                                f1s = [scores[m]['f1'] for m in metrics_order]
                                                if np is not None:
                                                    ax2.bar(x - width, precisions, width, label='Precision', color='#3b82f6')
                                                    ax2.bar(x, recalls, width, label='Recall', color='#10b981')
                                                    ax2.bar(x + width, f1s, width, label='F1', color='#f59e0b')
                                                    ax2.set_xticks(x)
                                                    ax2.set_xticklabels([m.upper() for m in metrics_order])
                                                else:
                                                    # Fallback without numpy (less precise spacing)
                                                    positions = []
                                                    for i in range(len(metrics_order)):
                                                        base = i * 3 * width
                                                        positions.append(base)
                                                        ax2.bar(base, precisions[i], width, label='Precision' if i == 0 else '', color='#3b82f6')
                                                        ax2.bar(base + width, recalls[i], width, label='Recall' if i == 0 else '', color='#10b981')
                                                        ax2.bar(base + 2*width, f1s[i], width, label='F1' if i == 0 else '', color='#f59e0b')
                                                    ax2.set_xticks([p + width for p in positions])
                                                    ax2.set_xticklabels([m.upper() for m in metrics_order])
                                                ax2.set_ylim(0, 1)
                                                ax2.set_ylabel('Score')
                            
                                                ax2.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
                                                st.pyplot(fig2, clear_figure=True)

                                                st.caption(f"Reference tokens: {rouge_json.get('reference_tokens')} | Candidate tokens: {rouge_json.get('candidate_tokens')}")

                                                # --- Auto-save ROUGE scores to CSV ---
                                                try:
                                                    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                                                    csv_filename = os.path.join(ROUGE_OUTPUT_DIR, f"rouge_scores_{ts}.csv")
                                                    # Write a header + rows for each metric
                                                    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
                                                        writer = csv.writer(f)
                                                        writer.writerow(["metric", "precision", "recall", "f1", "reference_tokens", "candidate_tokens", "model", "summary_length"])
                                                        ref_tok = rouge_json.get('reference_tokens')
                                                        cand_tok = rouge_json.get('candidate_tokens')
                                                        for metric_name, vals in scores.items():
                                                            writer.writerow([
                                                                metric_name,
                                                                vals['precision'],
                                                                vals['recall'],
                                                                vals['f1'],
                                                                ref_tok,
                                                                cand_tok,
                                                                st.session_state.get('last_model'),
                                                                st.session_state.get('last_length')
                                                            ])
                                                    st.info(f"ROUGE scores saved to {csv_filename}")
                                                except Exception as file_e:
                                                    st.warning(f"Could not save ROUGE CSV: {file_e}")
                                        except Exception as viz_e:
                                            st.warning(f"ROUGE computed, but visualization failed: {viz_e}")
                                    else:
                                        st.warning("ROUGE evaluation response did not contain scores.")
                                except Exception as er:
                                    st.error(f"ROUGE evaluation failed: {er}")

                                # Calculate additional metrics (BLEU, perplexity, readability delta)
                                with st.spinner("Calculating additional metrics..."):
                                    try:
                                        metrics_payload = {
                                            "reference": reference_input.strip(),
                                            "candidate": result["summary"],
                                            "original_text": original_text_display if original_text_display else None
                                        }
                                        
                                        metrics_resp = requests.post(f"{BACKEND_URL}/evaluate/metrics", json=metrics_payload, timeout=120)
                                        metrics_json = metrics_resp.json()
                                        
                                        # Debug: Collapsible raw response
                                        with st.expander("Debug: Raw metrics API response", expanded=False):
                                            st.json(metrics_json)
                                        
                                        if "error" not in metrics_json:
                                            st.markdown("### Additional Quality Metrics")
                                            
                                            # BLEU scores
                                            if "bleu" in metrics_json and isinstance(metrics_json["bleu"], dict) and "error" not in metrics_json["bleu"]:
                                                bleu_scores = metrics_json["bleu"]
                                                st.markdown("#### BLEU Scores")
                                                bleu_cols = st.columns(5)
                                                
                                                with bleu_cols[0]:
                                                    st.metric("BLEU-1", f"{bleu_scores['bleu_1']:.4f}")
                                                with bleu_cols[1]:
                                                    st.metric("BLEU-2", f"{bleu_scores['bleu_2']:.4f}")
                                                with bleu_cols[2]:
                                                    st.metric("BLEU-3", f"{bleu_scores['bleu_3']:.4f}")
                                                with bleu_cols[3]:
                                                    st.metric("BLEU-4", f"{bleu_scores['bleu_4']:.4f}")
                                                with bleu_cols[4]:
                                                    st.metric("BLEU", f"{bleu_scores['bleu']:.4f}")
                                                
                                                st.caption("BLEU scores measure the similarity between the generated summary and the reference. Higher scores (0-1) indicate better matches.")
                                            elif "bleu" in metrics_json and isinstance(metrics_json["bleu"], dict) and "error" in metrics_json["bleu"]:
                                                st.info(f"BLEU unavailable: {metrics_json['bleu']['error']}")
                                            
                                            # Perplexity
                                            if "perplexity" in metrics_json and isinstance(metrics_json["perplexity"], dict) and "error" not in metrics_json["perplexity"]:
                                                perplexity = metrics_json["perplexity"]["perplexity"]
                                                ngram = metrics_json["perplexity"]["ngram"]
                                                st.markdown("#### Perplexity")
                                                st.metric("Perplexity", f"{perplexity:.2f}")
                                                st.caption(f"Perplexity measures how 'surprised' the model is by the text. Lower values indicate more fluent text. (Using {ngram}-gram model)")
                                            elif "perplexity" in metrics_json and isinstance(metrics_json["perplexity"], dict) and "error" in metrics_json["perplexity"]:
                                                st.info(f"Perplexity unavailable: {metrics_json['perplexity']['error']}")
                                            
                                            # Readability delta
                                            if "readability" in metrics_json and "error" not in metrics_json["readability"] and metrics_json["readability"].get("delta"):
                                                delta = metrics_json["readability"]["delta"]
                                                original = metrics_json["readability"]["original"]
                                                summary = metrics_json["readability"]["summary"]
                                                
                                                st.markdown("#### Readability Metrics")
                                                
                                                import pandas as pd
                                                
                                                # Create a dataframe with selected readability metrics (first 4 only)
                                                metrics_df = pd.DataFrame({
                                                    "Metric": [
                                                        "Flesch Reading Ease",
                                                        "Flesch-Kincaid Grade",
                                                        "Gunning Fog",
                                                        "SMOG Index",
                                                    ],
                                                    "Original": [
                                                        original["flesch_reading_ease"],
                                                        original["flesch_kincaid_grade"],
                                                        original["gunning_fog"],
                                                        original["smog_index"],
                                                    ],
                                                    "Summary": [
                                                        summary["flesch_reading_ease"],
                                                        summary["flesch_kincaid_grade"],
                                                        summary["gunning_fog"],
                                                        summary["smog_index"],
                                                    ],
                                                    "Delta": [
                                                        delta["flesch_reading_ease"],
                                                        delta["flesch_kincaid_grade"],
                                                        delta["gunning_fog"],
                                                        delta["smog_index"],
                                                    ]
                                                })
                                                
                                                # Format the delta column with a + sign for positive values
                                                metrics_df["Delta"] = metrics_df["Delta"].apply(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")
                                                
                                                # Display the dataframe
                                                st.dataframe(
                                                    metrics_df,
                                                    column_config={
                                                        "Original": st.column_config.NumberColumn(format="%.2f"),
                                                        "Summary": st.column_config.NumberColumn(format="%.2f")
                                                    }
                                                )
                
                                                st.caption("Readability metrics compare the complexity of the original text vs. the summary. For Flesch Reading Ease, higher scores mean easier readability. For all others, lower scores indicate easier readability.")
                                        else:
                                            st.warning(f"Failed to calculate additional metrics: {metrics_json.get('error', 'Unknown error')}")
                                    except Exception as metrics_err:
                                        st.warning(f"Failed to calculate additional metrics: {metrics_err}")

            
                except Exception as e:
                    st.error(f"An error occurred: {e}")


# New Paraphrasing tab
with tabs[6]:  # Tab 6 - Paraphrasing
    token = st.session_state.get("token")
    if not token:
        st.markdown(
            """
            <div style='text-align:center;padding:50px 0;'>
                <h3 style='color:#334155;margin-bottom:10px;'>Sign in required</h3>
                <p style='color:#64748b;'>Please sign in to access the paraphrasing feature.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.subheader("Paraphrase Text or PDF")

    col1, col2 = st.columns(2)
    with col1:
        p_input_type = st.radio("Choose input type:", ["Plain Text", "PDF File"], key="para_input_type")
        # Let users type any HF model name but prefill with a good default
        p_model = st.selectbox(
            "Select Model",
            options=[
                # Friendly aliases (resolved on backend)
                "pegasus",
                "bart",
                "flan-t5",
            ],
            index=0,
            help="Choose a friendly alias.",
        )
        p_length = st.selectbox("Target Length", ["short", "medium", "long"], index=1)
        p_creativity = st.slider("Creativity", 0.0, 1.0, 0.3, 0.05,
                                 help="Higher values increase randomness (temperature/top-p).")
    with col2:
        st.write("Instructions:")
        st.markdown("- Paste text or upload a PDF.\n- Choose model, target length, and creativity.\n- Click 'Paraphrase'.")

    p_text_input = ""
    p_pdf_file = None
    if p_input_type == "Plain Text":
        p_text_input = st.text_area("Paste your text here", height=200, key="para_text")
    else:
        p_pdf_file = st.file_uploader("Upload a PDF", type=["pdf"], key="para_pdf")

    # ---------------- Session State Initialization for Paraphrasing ----------------
    if 'paraphrase_results' not in st.session_state:
        st.session_state['paraphrase_results'] = None
    if 'paraphrase_original_text' not in st.session_state:
        st.session_state['paraphrase_original_text'] = ""
    if 'paraphrase_text_for_backend' not in st.session_state:
        st.session_state['paraphrase_text_for_backend'] = ""
    if 'paraphrase_model' not in st.session_state:
        st.session_state['paraphrase_model'] = None
    if 'paraphrase_length' not in st.session_state:
        st.session_state['paraphrase_length'] = None
    if 'paraphrase_creativity' not in st.session_state:
        st.session_state['paraphrase_creativity'] = None
    if 'paraphrase_original_analysis' not in st.session_state:
        st.session_state['paraphrase_original_analysis'] = None
    if 'paraphrase_visualizations' not in st.session_state:
        st.session_state['paraphrase_visualizations'] = None

    paraphrase_clicked = st.button("Paraphrase")

    if paraphrase_clicked:
        if (p_input_type == "Plain Text" and not p_text_input.strip()) or (p_input_type == "PDF File" and not p_pdf_file):
            st.error("Please provide text or upload a PDF.")
        else:
            # Build request payload
            text_for_backend = p_text_input.strip()
            original_text_display = text_for_backend

            if p_pdf_file is not None:
                try:
                    if PyPDF2 is not None:
                        p_pdf_file.seek(0)
                        reader = PyPDF2.PdfReader(p_pdf_file)
                        pages = [p.extract_text() or "" for p in reader.pages]
                        extracted = "\n".join(pages).strip()
                        try:
                            t = extracted.replace("\r\n", "\n")
                            t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
                            t = re.sub(r"\n{3,}", "\n\n", t)
                            t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
                            t = re.sub(r"[ \t]{2,}", " ", t)
                            t = re.sub(r" +([.,;:!?])", r"\1", t)
                            extracted = t.strip()
                        except Exception:
                            pass
                        original_text_display = extracted
                        text_for_backend = extracted
                    else:
                        original_text_display = "(PyPDF2 not available to extract PDF text)"
                except Exception:
                    original_text_display = "(Unable to extract PDF text for preview)"

            payload = {
                "model_name": p_model,
                "text": text_for_backend,
                "creativity": float(p_creativity),
                "length": p_length,
                "user_email": st.session_state.get("user_email") or st.session_state.get("me", {}).get("email"),
            }

            with st.spinner("Generating paraphrases... (first run may take a few minutes while the model loads)"):
                ok, data = api_post("/paraphrase/", payload, token=token, timeout=600)
                if not ok:
                    st.error(data)
                else:
                    st.success("Paraphrases generated successfully!")
                    # Persist results in session state for later ROUGE evaluation reruns
                    st.session_state['paraphrase_results'] = data.get("paraphrased_results", [])
                    st.session_state['paraphrase_original_analysis'] = data.get("original_text_analysis")
                    st.session_state['paraphrase_visualizations'] = data.get("visualizations", {})
                    st.session_state['paraphrase_original_text'] = original_text_display
                    st.session_state['paraphrase_text_for_backend'] = text_for_backend
                    st.session_state['paraphrase_model'] = p_model
                    st.session_state['paraphrase_length'] = p_length
                    st.session_state['paraphrase_creativity'] = float(p_creativity)

    # ---------------- Display Stored Paraphrase Results (if any) ----------------
    if st.session_state['paraphrase_results']:
        results = st.session_state['paraphrase_results']
        original_text_display = st.session_state['paraphrase_original_text']
        original_analysis = st.session_state['paraphrase_original_analysis']
        visualizations = st.session_state['paraphrase_visualizations']
        text_for_backend = st.session_state['paraphrase_text_for_backend']
        p_model = st.session_state['paraphrase_model']
        p_length = st.session_state['paraphrase_length']
        p_creativity = st.session_state['paraphrase_creativity']

        col_o, col_r = st.columns([1, 1])
        with col_o:
            st.markdown("#### Original Text")
            st.text_area("Original", original_text_display[:15000], height=240, key="para_orig_display")
            if original_analysis:
                st.caption("Original complexity:")
                try:
                    st.json(original_analysis)
                except Exception:
                    st.write(original_analysis)
        with col_r:
            st.markdown("#### Paraphrased Versions")
            for i, item in enumerate(results, 1):
                st.markdown(f"**Option {i}**")
                st.text_area(
                    label=f"Paraphrase {i}",
                    value=item.get("text", ""),
                    height=120,
                    key=f"para_out_display_{i}",
                )
                comp = item.get("complexity")
                if comp:
                    with st.expander("Complexity details", expanded=False):
                        try:
                            st.json(comp)
                        except Exception:
                            st.write(comp)

        # ---- Complexity Visualization ----
        if visualizations:
            st.markdown("### Text Complexity Analysis")
            if "breakdown" in visualizations and "profile" in visualizations:
                viz_col1, viz_col2 = st.columns([1, 1])
                with viz_col1:
                    st.markdown("#### Complexity Breakdown")
                    st.image(f"data:image/png;base64,{visualizations['breakdown']}")
                with viz_col2:
                    st.markdown("#### Complexity Profile")
                    st.image(f"data:image/png;base64,{visualizations['profile']}")
                st.info("""**Understanding the Charts:**\n- **Complexity Breakdown**: Distribution of text complexity levels\n- **Complexity Profile**: Comparison across reading levels""")
            elif "error" in visualizations:
                st.warning(f"Could not generate visualization charts: {visualizations['error']}")
            else:
                st.info("No visualization data available.")

        # ---- ROUGE Evaluation Section ----
        st.markdown("### Semantic Similarity (ROUGE Scores)")
        st.info("Evaluate how semantically similar your paraphrases are to the original text or a reference paraphrase.")

        # Initialize ROUGE session state if missing
        for k, default in [
            ('rouge_reference', ''),
            ('rouge_metrics', ["rouge1", "rouge2", "rougeL"]),
            ('use_stemming', True),
            ('rouge_eval_results', None)
        ]:
            if k not in st.session_state:
                st.session_state[k] = default

        with st.expander("ROUGE Evaluation Options", expanded=False):
            ref_col1, ref_col2 = st.columns([2, 1])
            with ref_col1:
                ref_val = st.text_area(
                    "Reference Paraphrase (Optional)",
                    st.session_state['rouge_reference'],
                    height=100,
                    help="Provide a gold-standard paraphrase to compare against the generated ones.",
                    key="ref_paraphrase_input2"
                )
                st.session_state['rouge_reference'] = ref_val
            with ref_col2:
                st.markdown("ROUGE Metrics")
                rouge_metrics = []
                colm1, colm2 = st.columns([1, 1])
                with colm1:
                    if st.checkbox("rouge1", value="rouge1" in st.session_state['rouge_metrics'], key="rouge1_check2"):
                        rouge_metrics.append("rouge1")
                    if st.checkbox("rouge2", value="rouge2" in st.session_state['rouge_metrics'], key="rouge2_check2"):
                        rouge_metrics.append("rouge2")
                    if st.checkbox("rougeL", value="rougeL" in st.session_state['rouge_metrics'], key="rougeL_check2"):
                        rouge_metrics.append("rougeL")
                st.session_state['rouge_metrics'] = rouge_metrics
                use_stemmer_val = st.checkbox("Use stemming", value=st.session_state['use_stemming'], key="use_stemming_checkbox2")
                st.session_state['use_stemming'] = use_stemmer_val

        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        with button_col1:
            evaluate_clicked = st.button("Evaluate ROUGE Scores", key="evaluate_rouge_button2")

        if evaluate_clicked:
            with st.spinner("Computing ROUGE scores..."):
                eval_payload = {
                    "model_name": p_model,
                    "text": text_for_backend,
                    "creativity": float(p_creativity),
                    "length": p_length,
                    "evaluate_rouge": True,
                    "rouge_metrics": st.session_state['rouge_metrics'],
                    "use_stemmer": st.session_state['use_stemming']
                }
                if st.session_state['rouge_reference'].strip():
                    eval_payload['reference_paraphrase'] = st.session_state['rouge_reference'].strip()
                ok_eval, rouge_data = api_post("/paraphrase/", eval_payload, token=token, timeout=300)
                if not ok_eval:
                    st.error(f"ROUGE evaluation failed: {rouge_data}")
                else:
                    st.session_state['rouge_eval_results'] = rouge_data
                    # If paraphrased_results returned, refresh session paraphrases (some models may regenerate)
                    if rouge_data.get('paraphrased_results'):
                        st.session_state['paraphrase_results'] = rouge_data.get('paraphrased_results')

        if st.session_state['rouge_eval_results'] is not None:
            rouge_data = st.session_state['rouge_eval_results']
            rouge_eval = rouge_data.get("rouge_evaluation", {})
            if rouge_eval.get("available") is False:
                st.warning(f"ROUGE evaluation not available: {rouge_eval.get('reason', 'rouge-score library not installed')}")
            else:
                rouge_results = rouge_eval.get("results", [])
                if rouge_results:
                    if "visualizations" in rouge_eval and "rouge_chart" in rouge_eval["visualizations"]:
                        st.image(f"data:image/png;base64,{rouge_eval['visualizations']['rouge_chart']}")
                    with st.expander("Detailed ROUGE Scores", expanded=False):
                        for i, result in enumerate(rouge_results):
                            st.markdown(f"#### Option {i+1}")
                            st.markdown("**vs Original Text:**")
                            vs_orig = result.get("vs_original", {})
                            orig_scores = vs_orig.get("scores", {})
                            try:
                                import pandas as pd
                                rows = [{"Metric": m.upper(), "Precision": v.get('precision',0), "Recall": v.get('recall',0), "F1": v.get('f1',0)} for m, v in orig_scores.items()]
                                if rows:
                                    df = pd.DataFrame(rows)
                                    st.dataframe(df.style.format({"Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"}))
                                else:
                                    st.info("No scores available")
                            except Exception:
                                st.json(orig_scores)
                            vs_ref = result.get("vs_reference")
                            if vs_ref:
                                st.markdown("**vs Reference Paraphrase:**")
                                ref_scores = vs_ref.get("scores", {})
                                try:
                                    rows = [{"Metric": m.upper(), "Precision": v.get('precision',0), "Recall": v.get('recall',0), "F1": v.get('f1',0)} for m, v in ref_scores.items()]
                                    if rows:
                                        df = pd.DataFrame(rows)
                                        st.dataframe(df.style.format({"Precision":"{:.4f}","Recall":"{:.4f}","F1":"{:.4f}"}))
                                    else:
                                        st.info("No scores available")
                                except Exception:
                                    st.json(ref_scores)
                            st.markdown("---")
                else:
                    st.warning("No ROUGE results returned from the backend.")

# Summarization Dataset Evaluation tab
with tabs[8]:
    st.markdown("""
        <div style="text-align:center;margin-bottom:20px">
            <img src="https://cdn-icons-png.flaticon.com/512/4471/4471606.png" width="60">
            <h3 style="font-weight:600;color:#334155">Evaluate on Dataset (articles vs reference summaries)</h3>
            <p style="color:#64748b;font-size:14px;">Load a CSV with columns <code>article</code> and <code>highlights</code>, choose a model (T5-small or BART-base), generate summaries with your selected fine-tuned model, and compare to references using BLEU-1..4, Perplexity, and Readability deltas.</p>
        </div>
    """, unsafe_allow_html=True)

    # Choose source
    src = st.radio("Data source", ["Built-in dataset", "Upload CSV"], horizontal=True)

    selected_df = None
    if src == "Built-in dataset":
        # Offer selection of known CSVs if present
        ds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "summarization_dataset"))
        options = []
        path_map = {}
        for name in ["train.csv", "validation.csv", "test.csv"]:
            p = os.path.join(ds_dir, name)
            if os.path.isfile(p):
                options.append(name)
                path_map[name] = p
        if not options:
            st.warning("No built-in dataset files found in 'summarization_dataset/'. Switch to Upload CSV.")
        else:
            choice = st.selectbox("Choose file", options)
            if choice:
                import pandas as pd
                try:
                    selected_df = pd.read_csv(path_map[choice])
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
    else:
        ds_file = st.file_uploader("Upload dataset CSV (expects columns: article, highlights)", type=["csv"], key="ds_eval_uploader")
        if ds_file is not None:
            import pandas as pd
            try:
                selected_df = pd.read_csv(ds_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if selected_df is not None:
        df = selected_df
        if df is not None:
            # Validate columns
            expected_cols = {"article", "highlights"}
            if not expected_cols.issubset(set(c.lower() for c in df.columns)):
                st.error("CSV must contain columns: article, highlights")
            else:
                # Normalize column names
                lower_map = {c.lower(): c for c in df.columns}
                art_col = lower_map["article"]
                ref_col = lower_map["highlights"]
                df = df[[art_col, ref_col]].rename(columns={art_col: "article", ref_col: "reference"})

                # Row range selection (1-based, inclusive)
                total_rows = len(df)
                c_start, c_end = st.columns(2)
                with c_start:
                    start_row = st.number_input(
                        "Start row", min_value=1, max_value=max(1, total_rows), value=1, step=1,
                        help="1-based index of the first row to evaluate."
                    )
                with c_end:
                    default_end = min(total_rows, int(start_row) + 19)
                    end_row = st.number_input(
                        "End row (inclusive)", min_value=int(start_row), max_value=max(1, total_rows), value=default_end, step=1,
                        help="1-based index of the last row to evaluate (inclusive)."
                    )
                st.caption("Tip: Indices are 1-based and inclusive. Example: 1–100 evaluates the first 100 rows.")

                # Apply slicing safely (convert to 0-based, end exclusive)
                s_idx = max(0, int(start_row) - 1)
                e_idx = min(total_rows, int(end_row))
                if s_idx >= e_idx:
                    st.error("Start row must be less than or equal to End row.")
                    st.stop()
                df = df.iloc[s_idx:e_idx].copy()

                # Model selection for fine-tuned summarization
                model_choice_ds = st.selectbox(
                    "Model",
                    options=["T5-small", "BART-base"],
                    index=0,
                    key="ds_model_choice",
                    help="Choose which fine-tuned model to use for generating summaries."
                )

                show_pills([
                    (f"Rows: {int(start_row)}–{int(end_row)} ({len(df)} samples)", "info"),
                    (f"Model: {model_choice_ds}", "acc"),
                    ("Eval: BLEU/PPL/Readability/ROUGE", "ok")
                ])
                st.caption("Click 'Run Evaluation' to generate summaries and compute metrics.")
                run = st.button("Run Evaluation", type="primary")

                if run:
                    results = []
                    prog = st.progress(0)
                    for i, (idx, row) in enumerate(df.iterrows(), start=1):
                        article = str(row["article"]) if not pd.isna(row["article"]) else ""
                        reference = str(row["reference"]) if not pd.isna(row["reference"]) else ""
                        if not article or not reference:
                            continue
                        try:
                            # Summarize via backend fine-tuned model
                            form = {
                                "summary_length": "medium",
                                "input_type": "text",
                                "text_input": article,
                                "model_choice": ("t5" if model_choice_ds == "T5-small" else "bart"),
                            }
                            resp = requests.post(f"{BACKEND_URL}/summarize/fine-tuned/", data=form, timeout=120)
                            resp.raise_for_status()
                            gen = resp.json().get("summary", "")

                            # Metrics
                            metrics_payload = {
                                "reference": reference,
                                "candidate": gen,
                                "original_text": article,
                            }
                            mresp = requests.post(f"{BACKEND_URL}/evaluate/metrics", json=metrics_payload, timeout=120)
                            mjson = mresp.json()

                            # Extract key metrics safely
                            def _get(d, path, default=None):
                                cur = d
                                try:
                                    for p in path:
                                        cur = cur[p]
                                    return cur
                                except Exception:
                                    return default

                            # ROUGE scores (F1) via backend
                            try:
                                r_payload = {"reference": reference, "candidate": gen}
                                r_resp = requests.post(f"{BACKEND_URL}/evaluate/rouge", json=r_payload, timeout=60)
                                r_json = r_resp.json() if r_resp.ok else {}
                                r_scores = r_json.get("scores", {}) if isinstance(r_json, dict) else {}
                                r1_f1 = float(r_scores.get("rouge1", {}).get("f1", float("nan"))) if isinstance(r_scores.get("rouge1", {}), dict) else float("nan")
                                r2_f1 = float(r_scores.get("rouge2", {}).get("f1", float("nan"))) if isinstance(r_scores.get("rouge2", {}), dict) else float("nan")
                                rL_f1 = float(r_scores.get("rougeL", {}).get("f1", float("nan"))) if isinstance(r_scores.get("rougeL", {}), dict) else float("nan")
                            except Exception:
                                r1_f1 = r2_f1 = rL_f1 = float("nan")

                            row_out = {
                                "index": idx,
                                "article": article,
                                "generated": gen,
                                "reference": reference,
                                # BLEU
                                "bleu_1": _get(mjson, ["bleu", "bleu_1"], float("nan")),
                                "bleu_2": _get(mjson, ["bleu", "bleu_2"], float("nan")),
                                "bleu_3": _get(mjson, ["bleu", "bleu_3"], float("nan")),
                                "bleu_4": _get(mjson, ["bleu", "bleu_4"], float("nan")),
                                "bleu": _get(mjson, ["bleu", "bleu"], float("nan")),
                                # ROUGE (F1)
                                "rouge1_f1": r1_f1,
                                "rouge2_f1": r2_f1,
                                "rougeL_f1": rL_f1,
                                # Perplexities
                                "ppl_candidate": _get(mjson, ["perplexity_candidate", "perplexity"], _get(mjson, ["perplexity", "perplexity"], float("nan"))),
                                "ppl_reference": _get(mjson, ["perplexity_reference", "perplexity"], float("nan")),
                                # Readability deltas
                                "delta_flesch_reading_ease": _get(mjson, ["readability_ref_candidate", "delta", "flesch_reading_ease"], float("nan")),
                                "delta_flesch_kincaid": _get(mjson, ["readability_ref_candidate", "delta", "flesch_kincaid_grade"], float("nan")),
                                "delta_gunning_fog": _get(mjson, ["readability_ref_candidate", "delta", "gunning_fog"], float("nan")),
                                "delta_smog_index": _get(mjson, ["readability_ref_candidate", "delta", "smog_index"], float("nan")),
                            }
                            results.append(row_out)
                        except Exception as e:
                            st.warning(f"Row {idx} failed: {e}")
                        finally:
                            prog.progress(min(i / len(df), 1.0))

                    if results:
                        rdf = pd.DataFrame(results)
                        st.success("Evaluation complete.")

                        # Side-by-side viewer for first few samples
                        st.markdown("### Sample Results")
                        sample_n = min(5, len(rdf))
                        for i in range(sample_n):
                            r = rdf.iloc[i]
                            st.markdown(f"#### Sample #{int(r['index'])}")
                            c1, c2, c3 = st.columns(3)
                            with c1:
                                st.markdown("**Article**")
                                st.write(r["article"])
                            with c2:
                                st.markdown("**Generated Summary**")
                                st.write(r["generated"])
                            with c3:
                                st.markdown("**Reference Summary**")
                                st.write(r["reference"])
                            m1, m2, m3 = st.columns(3)
                            with m1:
                                st.metric("BLEU-1", f"{r['bleu_1']:.4f}" if pd.notna(r['bleu_1']) else "-")
                                st.metric("BLEU-4", f"{r['bleu_4']:.4f}" if pd.notna(r['bleu_4']) else "-")
                            with m2:
                                st.metric("PPL (gen)", f"{r['ppl_candidate']:.2f}" if pd.notna(r['ppl_candidate']) else "-")
                                st.metric("PPL (ref)", f"{r['ppl_reference']:.2f}" if pd.notna(r['ppl_reference']) else "-")
                            with m3:
                                st.metric("Δ Flesch Ease (gen-ref)", f"{r['delta_flesch_reading_ease']:+.2f}" if pd.notna(r['delta_flesch_reading_ease']) else "-")
                                st.metric("Δ SMOG (gen-ref)", f"{r['delta_smog_index']:+.2f}" if pd.notna(r['delta_smog_index']) else "-")
                            st.divider()

                        # Download full results
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        out_csv = rdf.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="Download Results CSV",
                            data=out_csv,
                            file_name=f"eval_results_{ts}.csv",
                            mime="text/csv",
                        )

                        # Aggregate metrics
                        st.markdown("### Aggregates")
                        agg = rdf[[
                            "bleu_1","bleu_2","bleu_3","bleu_4","bleu",
                            "rouge1_f1","rouge2_f1","rougeL_f1",
                            "ppl_candidate","ppl_reference",
                            "delta_flesch_reading_ease","delta_flesch_kincaid","delta_gunning_fog","delta_smog_index"
                        ]].mean(numeric_only=True)
                        st.dataframe(agg.to_frame("mean").round(4))

                        # --- Visualizations ---
                        st.markdown("### Visualizations")
                        try:
                            import pandas as pd  # ensure available in this scope
                            if plt is None:
                                st.warning("matplotlib not installed; run 'pip install matplotlib' to view charts.")
                            else:
                                # 1) BLEU grouped bar
                                bleu_cols = ["bleu_1","bleu_2","bleu_3","bleu_4","bleu"]
                                bleu_vals = [float(agg.get(c, float('nan'))) for c in bleu_cols]
                                bleu_vals = [0.0 if pd.isna(v) else v for v in bleu_vals]
                                fig1, ax1 = plt.subplots(figsize=(6, 3.0))
                                x = list(range(len(bleu_cols))) if np is None else np.arange(len(bleu_cols))
                                ax1.bar(x, bleu_vals, color="#3b82f6")
                                ax1.set_xticks(x)
                                ax1.set_xticklabels([c.upper() for c in bleu_cols])
                                ax1.set_ylim(0, 1)
                                ax1.set_ylabel("Score")
                                ax1.set_title("Mean BLEU Scores")
                                for xi, v in zip(x, bleu_vals):
                                    ax1.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
                                # defer rendering; will place in columns

                                # 1b) ROUGE (F1) grouped bar
                                rouge_cols = ["rouge1_f1","rouge2_f1","rougeL_f1"]
                                rouge_labels = ["ROUGE-1 F1","ROUGE-2 F1","ROUGE-L F1"]
                                rouge_vals = [float(agg.get(c, float('nan'))) for c in rouge_cols]
                                rouge_vals = [0.0 if pd.isna(v) else v for v in rouge_vals]
                                fig1b, ax1b = plt.subplots(figsize=(6, 3.0))
                                x1b = list(range(len(rouge_cols))) if np is None else np.arange(len(rouge_cols))
                                ax1b.bar(x1b, rouge_vals, color="#06b6d4")
                                ax1b.set_xticks(x1b)
                                ax1b.set_xticklabels(rouge_labels)
                                ax1b.set_ylim(0, 1)
                                ax1b.set_ylabel("F1")
                                ax1b.set_title("Mean ROUGE (F1)")
                                for xi, v in zip(x1b, rouge_vals):
                                    ax1b.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
                                # defer rendering; will place in columns

                                # 2) Perplexity bar
                                ppl_cols = ["ppl_candidate","ppl_reference"]
                                ppl_vals = [float(agg.get(c, float('nan'))) for c in ppl_cols]
                                ppl_vals = [0.0 if pd.isna(v) else v for v in ppl_vals]
                                fig2, ax2 = plt.subplots(figsize=(5.5, 3.0))
                                x2 = list(range(len(ppl_cols))) if np is None else np.arange(len(ppl_cols))
                                ax2.bar(x2, ppl_vals, color=["#10b981", "#f59e0b"])
                                ax2.set_xticks(x2)
                                ax2.set_xticklabels([c.replace("ppl_","PPL ").title() for c in ppl_cols])
                                ax2.set_ylabel("Perplexity (lower is better)")
                                ax2.set_title("Mean Perplexity")
                                for xi, v in zip(x2, ppl_vals):
                                    ax2.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

                                # Render 4 charts in two columns (2 charts per column)
                                cols = st.columns(2)
                                with cols[0]:
                                    st.pyplot(fig1, clear_figure=True)   # BLEU
                                    st.pyplot(fig1b, clear_figure=True)  # ROUGE F1
                                with cols[1]:
                                    st.pyplot(fig2, clear_figure=True)   # Perplexity

                                # 3) Readability deltas bar
                                read_cols = [
                                    "delta_flesch_reading_ease","delta_flesch_kincaid",
                                    "delta_gunning_fog","delta_smog_index"
                                ]
                                read_labels = ["Δ Flesch Ease","Δ F-K Grade","Δ Gunning Fog","Δ SMOG"]
                                read_vals = [float(agg.get(c, float('nan'))) for c in read_cols]
                                read_vals = [0.0 if pd.isna(v) else v for v in read_vals]
                                fig3, ax3 = plt.subplots(figsize=(6, 3.0))
                                x3 = list(range(len(read_cols))) if np is None else np.arange(len(read_cols))
                                ax3.bar(x3, read_vals, color="#8b5cf6")
                                ax3.set_xticks(x3)
                                ax3.set_xticklabels(read_labels)
                                ax3.axhline(0, color="#94a3b8", linewidth=1)
                                ax3.set_ylabel("Delta (gen - ref)")
                                ax3.set_title("Mean Readability Deltas")
                                for xi, v in zip(x3, read_vals):
                                    ax3.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
                                # Readability chart stacked below in right column
                                with cols[1]:
                                    st.pyplot(fig3, clear_figure=True)

                                # Removed BLEU-1 distribution histogram as requested
                        except Exception as viz_e:
                            st.info(f"Could not render charts: {viz_e}")
                

# Simplified sidebar with essential information
st.sidebar.markdown("""
<div style="text-align:center;margin-bottom:20px;">
    <img src="https://cdn-icons-png.flaticon.com/512/3368/3368235.png" width="50">
    <h3 style="font-weight:600;color:#334155;">Textmorph</h3>
    <p style="color:#64748b;font-size:14px;">Text Analytics Platform</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.subheader("System Status")

# Simplified backend status indicator
try:
    # Ping health endpoint to indicate backend availability
    _ping = requests.get(f"{API_URL}/ping", timeout=2)
    if _ping.ok:
        st.sidebar.success("Backend: Online")
    else:
        st.sidebar.warning("Backend: Unreachable")
except Exception:
    st.sidebar.error("Backend: Offline")

# About section
st.sidebar.subheader("About")
st.sidebar.info("""
Textmorph provides advanced text analysis and summarization using AI algorithms:
- Readability analysis
- Text summarization
- Document processing
""")

# -------------------- Fine-tuned Summarization Tab --------------------

with tabs[7]:  # Tab 7 - Fine-tuned Summarization
    token = st.session_state.get("token")
    if not token:
        st.markdown("""
            <div style='text-align:center;padding:50px 0;'>
                <h3 style='color:#334155;margin-bottom:10px;'>Sign in required</h3>
                <p style='color:#64748b;'>Please sign in to access the fine-tuned summarization feature.</p>
            </div>
        """, unsafe_allow_html=True)
        # Early return pattern – don't render summarization UI when not authenticated
        st.stop()
        
    st.subheader("Summarize with Fine-tuned Model")
    
    col1, col2 = st.columns(2)

    with col1:
        input_type = st.radio("Choose input type:", ["Plain Text", "PDF File"], key="ft_input_type")
        model_choice = st.selectbox(
            "Model",
            options=["T5-small", "BART-base"],
            index=0,
            key="ft_model_choice"
        )
        summary_length = st.selectbox(
            "Summary Length",
            options=["short", "medium", "long"],
            index=1,
            key="ft_summary_length"
        )

    with col2:
        st.write("Instructions:")
        st.markdown("- Paste text or upload a PDF.\n- Choose a model (T5-small or BART-base) and summary length.\n- Click 'Generate Summary'.\n- This tab uses local fine-tuned models stored in your workspace.")

    text_input = ""
    pdf_file = None

    if input_type == "Plain Text":
        text_input = st.text_area("Paste your text here", height=200, key="ft_text_input")
    else:
        pdf_file = st.file_uploader("Upload a PDF", type=["pdf"], key="ft_pdf_file")

    # Reference summary removed for this tab per request

    # Initialize session state for last summary
    if 'ft_last_summary' not in st.session_state:
        st.session_state['ft_last_summary'] = None
        st.session_state['ft_last_length'] = None

    generate_clicked = st.button("Generate Summary", key="ft_generate_button")

    if generate_clicked:
        if (input_type == "Plain Text" and not text_input.strip()) or (input_type == "PDF File" and not pdf_file):
            st.error("Please provide text or upload a PDF.")
        else:
            with st.spinner("Generating summary with selected fine-tuned model..."):
                # Include user email if available for history tracking
                form_data = {
                    "summary_length": summary_length,
                    "input_type": "pdf" if input_type == "PDF File" else "text",
                    "text_input": text_input,
                    "model_choice": ("t5" if model_choice == "T5-small" else "bart"),
                }
                # Add user email for history if present in session
                if st.session_state.get("user_email"):
                    form_data["user_email"] = st.session_state["user_email"]

                files = None
                if pdf_file:
                    files = {"pdf_file": (pdf_file.name, pdf_file, "application/pdf")}

                try:
                    response = requests.post(
                        f"{BACKEND_URL}/summarize/fine-tuned/",
                        data=form_data,
                        files=files
                    )
                    result = response.json()

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        st.success(f"Summary generated successfully with {model_choice} model!")

                        # Prepare original text for side-by-side view
                        original_text_display = ""
                        if input_type == "Plain Text":
                            original_text_display = text_input.strip()
                        else:
                            # Try to extract PDF text (lightweight) for display
                            try:
                                if pdf_file is not None and PyPDF2 is not None:
                                    pdf_file.seek(0)
                                    reader = PyPDF2.PdfReader(pdf_file)
                                    pages = [p.extract_text() or "" for p in reader.pages]
                                    original_text_display = "\n".join(pages).strip()
                                    # --- Normalize PDF text so it shows as readable paragraphs ---
                                    try:
                                        # Keep paragraph breaks (double newlines) but collapse single newlines to spaces
                                        t = original_text_display.replace("\r\n", "\n")
                                        # Join hyphenated line breaks like "exam-\nple" -> "example"
                                        t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
                                        # Reduce 3+ newlines to 2 (standard paragraph break)
                                        t = re.sub(r"\n{3,}", "\n\n", t)
                                        # Replace remaining single newlines with a space (preserve double newlines)
                                        t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
                                        # Normalize extra spaces
                                        t = re.sub(r"[ \t]{2,}", " ", t)
                                        # Remove spaces before punctuation
                                        t = re.sub(r" +([.,;:!?])", r"\1", t)
                                        original_text_display = t.strip()
                                    except Exception:
                                        # If normalization fails for any reason, fall back to raw extracted text
                                        pass
                            except Exception:
                                original_text_display = "(Unable to extract PDF text for preview)"

                        summary_text_display = result["summary"].strip()

                        # Persist to history (original + result) for fine-tuned summarization
                        try:
                            email = st.session_state.get("user_email") or (st.session_state.get("me") or {}).get("email")
                            original_for_history = original_text_display or (text_input.strip() if input_type == "Plain Text" else "")
                            if email and original_for_history and summary_text_display:
                                hist_payload = {
                                    "email": email,
                                    "type": "summary",
                                    "original_text": original_for_history,
                                    "result_text": summary_text_display,
                                    "model": f"fine-tuned-{('t5' if model_choice == 'T5-small' else 'bart')}",
                                    "parameters": json.dumps({"summary_length": summary_length}),
                                }
                                requests.post(f"{BACKEND_URL}/history", json=hist_payload, timeout=15)
                        except Exception:
                            pass

                        # Persist core data for later (including language toggle use)
                        st.session_state['ft_last_summary'] = result["summary"]
                        st.session_state['ft_last_length'] = summary_length
                        st.session_state['ft_last_original'] = original_text_display
                        st.session_state['ft_last_model'] = model_choice

                        # Clear any stale translations for a new result
                        st.session_state.pop('ft_original_hi', None)
                        st.session_state.pop('ft_summary_hi', None)
                        st.session_state.pop('ft_translation_hash', None)

                        # Immediately render (English) with toggle support
                        st.session_state['ft_language'] = 'English'

                        # Helper functions for language toggle / translation
                        def _render_ft_summary_block():
                            """Render fine-tuned summary with independent language toggles for Original and Summary."""
                            orig_en = st.session_state.get('ft_last_original', '') or ''
                            sum_en = st.session_state.get('ft_last_summary', '') or ''
                            model_used = st.session_state.get('ft_last_model', model_choice)
                            import hashlib, re as _re2
                            base_hash = hashlib.sha1((orig_en + '\n' + sum_en).encode('utf-8', errors='ignore')).hexdigest()
                            # Initialize toggle states
                            if 'ft_language_original' not in st.session_state:
                                st.session_state['ft_language_original'] = 'English'
                            if 'ft_language_summary' not in st.session_state:
                                st.session_state['ft_language_summary'] = 'English'

                            col_tog1, col_tog2 = st.columns(2)
                            with col_tog1:
                                st.session_state['ft_language_original'] = st.radio("Original Text Language", ["English", "हिंदी"], horizontal=True, key="ft_lang_orig_toggle")
                            with col_tog2:
                                st.session_state['ft_language_summary'] = st.radio("Summary Language", ["English", "हिंदी"], horizontal=True, key="ft_lang_sum_toggle")

                            need_hi = any(lang == 'हिंदी' for lang in [st.session_state['ft_language_original'], st.session_state['ft_language_summary']])
                            if need_hi:
                                # Invalidate cached translations if underlying text changed
                                if st.session_state.get('ft_translation_hash') != base_hash:
                                    st.session_state.pop('ft_original_hi', None)
                                    st.session_state.pop('ft_summary_hi', None)
                                if ('ft_original_hi' not in st.session_state and st.session_state['ft_language_original'] == 'हिंदी') or \
                                   ('ft_summary_hi' not in st.session_state and st.session_state['ft_language_summary'] == 'हिंदी'):
                                    with st.spinner('Translating (may download models first time)...'):
                                        try:
                                            if st.session_state['ft_language_original'] == 'हिंदी' and 'ft_original_hi' not in st.session_state:
                                                st.session_state['ft_original_hi'] = translate_text_full(orig_en, 'en', 'hi')
                                            if st.session_state['ft_language_summary'] == 'हिंदी' and 'ft_summary_hi' not in st.session_state:
                                                st.session_state['ft_summary_hi'] = translate_text_full(sum_en, 'en', 'hi')
                                            st.session_state['ft_translation_hash'] = base_hash
                                        except Exception as te:
                                            st.warning(f"Hindi translation failed: {te}")
                                            if 'ft_original_hi' not in st.session_state:
                                                st.session_state['ft_original_hi'] = orig_en
                                            if 'ft_summary_hi' not in st.session_state:
                                                st.session_state['ft_summary_hi'] = sum_en

                            orig_display = st.session_state.get('ft_original_hi') if st.session_state['ft_language_original'] == 'हिंदी' else orig_en
                            sum_display = st.session_state.get('ft_summary_hi') if st.session_state['ft_language_summary'] == 'हिंदी' else sum_en

                            def _wc(t: str) -> int:
                                return len([w for w in _re2.findall(r"\w+", t)]) if t else 0
                            wc_orig_en = _wc(orig_en)
                            wc_sum_en = _wc(sum_en)
                            compression = (1 - (wc_sum_en / wc_orig_en)) * 100 if wc_orig_en > 0 else 0

                            col_orig, col_sum = st.columns(2)
                            with col_orig:
                                st.markdown("#### Original Text" + (" (हिंदी)" if st.session_state['ft_language_original'] == 'हिंदी' else ""))
                                cap_o = f"{_wc(orig_display)} words"
                                if st.session_state['ft_language_original'] == 'हिंदी':
                                    cap_o += f" • EN: {wc_orig_en}"
                                st.caption(cap_o)
                                # Force widget value refresh when language toggles by updating session_state before rendering
                                st.session_state['ft_orig_view'] = (orig_display or '')[:15000]
                                st.text_area(label="Original", value=st.session_state['ft_orig_view'], height=260, key="ft_orig_view")
                            with col_sum:
                                st.markdown(f"#### Summary ({model_used})" + (" (हिंदी)" if st.session_state['ft_language_summary'] == 'हिंदी' else ""))
                                cap_s = f"{_wc(sum_display)} words"
                                if st.session_state['ft_language_summary'] == 'हिंदी':
                                    cap_s += f" • EN: {wc_sum_en}"
                                st.caption(cap_s)
                                st.session_state['ft_sum_view'] = sum_display or ''
                                st.text_area(label="Summary", value=st.session_state['ft_sum_view'], height=260, key="ft_sum_view")
                                if wc_orig_en > 0:
                                    st.markdown(f"**Compression (English baseline):** {compression:.0f}%")
                                else:
                                    st.markdown("**Compression:** n/a")

                            with st.container():
                                if model_used == "T5-small":
                                    st.markdown("<div style='margin-top:14px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#f1f5f9'" \
                                                "<strong>Fine-tuned T5 Summarization Model</strong><ul style='margin-top:6px;'>" \
                                                "<li>Custom T5-small model fine-tuned on a specialized dataset</li>" \
                                                "<li>Optimized for high-quality, contextually relevant summaries</li>" \
                                                "<li>Enhanced beam search configuration for better outputs</li>" \
                                                "<li>Comprehensive metrics: ROUGE, BLEU, Perplexity, and Readability</li>" \
                                                "<li>Efficient processing of longer texts through chunking</li>" \
                                                "<li>Improved coherence and factual accuracy in generated summaries</li>" \
                                                "</ul></div>", unsafe_allow_html=True)
                                else:
                                    st.markdown("<div style='margin-top:14px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#f1f5f9'" \
                                                "<strong>Fine-tuned BART Summarization Model</strong><ul style='margin-top:6px;'>" \
                                                "<li>Local BART model loaded from fine_tuned_bart_summarizer/</li>" \
                                                "<li>Strong performance on abstractive summarization</li>" \
                                                "<li>Beam search decoding tuned for coherent outputs</li>" \
                                                "<li>Comprehensive metrics: ROUGE, BLEU, Perplexity, and Readability</li>" \
                                                "<li>Handles longer inputs via chunking</li>" \
                                                "</ul></div>", unsafe_allow_html=True)

                        _render_ft_summary_block()

                        # Reference-based evaluation removed for this tab

                except Exception as e:
                    st.error(f"An error occurred: {e}")

    # Persistent display (after first generation) allowing language toggle
    if st.session_state.get('ft_last_summary') and not generate_clicked:
        # Reuse the same render helper if defined (first run) else define a thin wrapper
        def _render_ft_summary_block_external():
            # External persistent renderer with independent toggles
            orig_en = st.session_state.get('ft_last_original', '') or ''
            sum_en = st.session_state.get('ft_last_summary', '') or ''
            model_used = st.session_state.get('ft_last_model', 'Model')
            import hashlib, re as _re2
            base_hash = hashlib.sha1((orig_en + '\n' + sum_en).encode('utf-8', errors='ignore')).hexdigest()
            if 'ft_language_original' not in st.session_state:
                st.session_state['ft_language_original'] = 'English'
            if 'ft_language_summary' not in st.session_state:
                st.session_state['ft_language_summary'] = 'English'
            col_tog1, col_tog2 = st.columns(2)
            with col_tog1:
                st.session_state['ft_language_original'] = st.radio("Original Text Language", ["English", "हिंदी"], horizontal=True, key="ft_lang_orig_toggle")
            with col_tog2:
                st.session_state['ft_language_summary'] = st.radio("Summary Language", ["English", "हिंदी"], horizontal=True, key="ft_lang_sum_toggle")
            need_hi = any(lang == 'हिंदी' for lang in [st.session_state['ft_language_original'], st.session_state['ft_language_summary']])
            if need_hi:
                if st.session_state.get('ft_translation_hash') != base_hash:
                    st.session_state.pop('ft_original_hi', None)
                    st.session_state.pop('ft_summary_hi', None)
                if ('ft_original_hi' not in st.session_state and st.session_state['ft_language_original'] == 'हिंदी') or \
                   ('ft_summary_hi' not in st.session_state and st.session_state['ft_language_summary'] == 'हिंदी'):
                    with st.spinner('Translating (may download models first time)...'):
                        try:
                            if st.session_state['ft_language_original'] == 'हिंदी' and 'ft_original_hi' not in st.session_state:
                                st.session_state['ft_original_hi'] = translate_text_full(orig_en, 'en', 'hi')
                            if st.session_state['ft_language_summary'] == 'हिंदी' and 'ft_summary_hi' not in st.session_state:
                                st.session_state['ft_summary_hi'] = translate_text_full(sum_en, 'en', 'hi')
                            st.session_state['ft_translation_hash'] = base_hash
                        except Exception as te:
                            st.warning(f"Hindi translation failed: {te}")
                            if 'ft_original_hi' not in st.session_state:
                                st.session_state['ft_original_hi'] = orig_en
                            if 'ft_summary_hi' not in st.session_state:
                                st.session_state['ft_summary_hi'] = sum_en
            orig_display = st.session_state.get('ft_original_hi') if st.session_state['ft_language_original'] == 'हिंदी' else orig_en
            sum_display = st.session_state.get('ft_summary_hi') if st.session_state['ft_language_summary'] == 'हिंदी' else sum_en
            def _wc(t: str) -> int:
                return len([w for w in _re2.findall(r"\w+", t)]) if t else 0
            wc_orig_en = _wc(orig_en)
            wc_sum_en = _wc(sum_en)
            compression = (1 - (wc_sum_en / wc_orig_en)) * 100 if wc_orig_en > 0 else 0
            col_orig, col_sum = st.columns(2)
            with col_orig:
                st.markdown("#### Original Text" + (" (हिंदी)" if st.session_state['ft_language_original'] == 'हिंदी' else ""))
                cap_o = f"{_wc(orig_display)} words"
                if st.session_state['ft_language_original'] == 'हिंदी':
                    cap_o += f" • EN: {wc_orig_en}"
                st.caption(cap_o)
                st.session_state['ft_orig_view'] = (orig_display or '')[:15000]
                st.text_area(label="Original", value=st.session_state['ft_orig_view'], height=260, key="ft_orig_view")
            with col_sum:
                st.markdown(f"#### Summary ({model_used})" + (" (हिंदी)" if st.session_state['ft_language_summary'] == 'हिंदी' else ""))
                cap_s = f"{_wc(sum_display)} words"
                if st.session_state['ft_language_summary'] == 'हिंदी':
                    cap_s += f" • EN: {wc_sum_en}"
                st.caption(cap_s)
                st.session_state['ft_sum_view'] = sum_display or ''
                st.text_area(label="Summary", value=st.session_state['ft_sum_view'], height=260, key="ft_sum_view")
                if wc_orig_en > 0:
                    st.markdown(f"**Compression (English baseline):** {compression:.0f}%")
                else:
                    st.markdown("**Compression:** n/a")
            with st.container():
                if model_used == "T5-small":
                    st.markdown("<div style='margin-top:14px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#f1f5f9'" \
                                "<strong>Fine-tuned T5 Summarization Model</strong><ul style='margin-top:6px;'>" \
                                "<li>Custom T5-small model fine-tuned on a specialized dataset</li>" \
                                "<li>Optimized for high-quality, contextually relevant summaries</li>" \
                                "<li>Enhanced beam search configuration for better outputs</li>" \
                                "<li>Comprehensive metrics: ROUGE, BLEU, Perplexity, and Readability</li>" \
                                "<li>Efficient processing of longer texts through chunking</li>" \
                                "<li>Improved coherence and factual accuracy in generated summaries</li>" \
                                "</ul></div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='margin-top:14px;padding:14px;border:1px solid #e2e8f0;border-radius:8px;background:#f1f5f9'" \
                                "<strong>Fine-tuned BART Summarization Model</strong><ul style='margin-top:6px;'>" \
                                "<li>Local BART model loaded from fine_tuned_bart_summarizer/</li>" \
                                "<li>Strong performance on abstractive summarization</li>" \
                                "<li>Beam search decoding tuned for coherent outputs</li>" \
                                "<li>Comprehensive metrics: ROUGE, BLEU, Perplexity, and Readability</li>" \
                                "<li>Handles longer inputs via chunking</li>" \
                                "</ul></div>", unsafe_allow_html=True)

        _render_ft_summary_block_external()


# -------------------- History Tab --------------------

with tabs[11]:  # History (moved to last)
    token = st.session_state.get("token")
    if not token:
        st.warning("Please sign in to view your history.")
    else:
        st.title("📚 Your Transformation History")
        st.write("Here is a record of your recent paraphrasing activities.")

        # Add filter options
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            filter_type = st.selectbox("Filter by type:", ["All", "Summary", "Paraphrase"], key="history_filter")
        with col2:
            limit = st.number_input("Limit results:", min_value=5, max_value=100, value=20, step=5, key="history_limit")
        with col3:
            refresh = st.button("Refresh", key="refresh_history")

        # Convert filter for API
        filter_param = None if filter_type == "All" else filter_type.lower()
        
        # Fetch history from API
        # Resolve email: prefer session_state.user_email, fallback to /me
        email = st.session_state.get("user_email") or (st.session_state.get("me") or {}).get("email")
        if not email:
            # Try one more time to fetch /me
            ok_me, me = api_get("/me", token=token)
            if ok_me and isinstance(me, dict):
                st.session_state["me"] = me
                email = me.get("email")
        if not email:
            st.info("Couldn't resolve your email from session; please sign out and sign in again.")
            st.stop()
        
        # Build the API URL
        if filter_param:
            history_url = f"/history?email={email}&type={filter_param}&limit={limit}"
        else:
            history_url = f"/history?email={email}&limit={limit}"

        success, history_data = api_get(history_url, token=token)

        if not success:
            st.error(f"Failed to load history: {history_data}")
        else:
            # Display history entries
            if "entries" in history_data and history_data["entries"]:
                for entry in history_data["entries"]:
                    entry_type = entry["type"].capitalize()
                    # Parse timestamp as UTC, then convert to local timezone for display
                    created_raw = (entry.get("created_at") or "").strip()
                    dt = None
                    # Try ISO 8601 parse first
                    try:
                        iso = created_raw.replace("Z", "+00:00")
                        dt = datetime.fromisoformat(iso)
                    except Exception:
                        pass
                    if dt is None:
                        # Fallbacks for common formats (naive, treat as UTC)
                        for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                            try:
                                dt = datetime.strptime(created_raw, fmt)
                                break
                            except Exception:
                                continue
                    if dt is None:
                        dt = datetime.utcnow().replace(tzinfo=timezone.utc)
                    # Assume UTC when tzinfo missing
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    local_dt = dt.astimezone()  # convert to local timezone
                    formatted_date = local_dt.strftime("%B %d, %Y at %I:%M %p")
                    
                    # Create an expandable section for each entry
                    with st.expander(f"{entry_type} on {formatted_date}", expanded=False):
                        st.markdown("### Original Text")
                        st.text_area("Original", entry["original_text"], height=150, disabled=True, key=f"orig_{entry['id']}")
                        
                        st.markdown("### Result")
                        st.text_area("Result", entry.get("result_text", ""), height=150, disabled=True, key=f"result_{entry['id']}")
                        
                        # Additional info
                        col1, col2, col3 = st.columns([1, 1, 1])
                        with col1:
                            st.info(f"Model: {entry['model']}")
                        with col2:
                            # Format the parameters nicely if possible
                            try:
                                import json
                                raw_params = entry.get('parameters')
                                params = json.loads(raw_params) if isinstance(raw_params, str) else (raw_params or {})
                                param_text = ", ".join([f"{k}: {v}" for k, v in params.items()]) if params else "Default parameters"
                            except Exception:
                                param_text = entry.get('parameters') or "Default parameters"
                            st.info(f"Parameters: {param_text}")
                        with col3:
                            # Add a delete button
                            if st.button("Delete", key=f"delete_{entry['id']}"):
                                delete_url = f"/history/{entry['id']}?email={email}"
                                response = requests.delete(f"{BACKEND_URL}{delete_url}")
                                success = response.ok
                                try:
                                    delete_result = response.json()
                                except:
                                    delete_result = response.text
                                if success:
                                    st.success("Entry deleted successfully!")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to delete entry: {delete_result}")
            else:
                st.info("No history entries found. Try creating some summaries or paraphrases first!")
                
    st.divider()
    st.caption("Your transformation history is stored securely and is only visible to you.")

# -------------------- Fine-tuned Paraphrasing (Local Model) --------------------

with tabs[9]:  # Tab 9 - Fine-tuned Paraphrasing
    token = st.session_state.get("token")
    if not token:
        st.markdown(
            """
            <div style='text-align:center;padding:50px 0;'>
                <h3 style='color:#334155;margin-bottom:10px;'>Sign in required</h3>
                <p style='color:#64748b;'>Please sign in to access the fine-tuned paraphrasing feature.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.subheader("Paraphrase with Fine-tuned Model (Local Files)")

    # Define available local model folders
    T5_PARAPHRASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_paraphrasing_model"))
    BART_PARAPHRASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bart_paraphrasing_model"))

    # Model selection and resolve selected directory
    model_choice = st.selectbox("Model", ["T5-small", "BART-base"], index=0, key="ftp_model_choice")
    MODEL_DIR = T5_PARAPHRASE_DIR if model_choice == "T5-small" else BART_PARAPHRASE_DIR

    if not os.path.isdir(MODEL_DIR):
        st.error(
            f"Could not find the selected paraphrasing model folder for {model_choice} at '{MODEL_DIR}'. "
            "Please ensure the directory exists and includes the tokenizer/model files."
        )
        st.stop()

    @st.cache_resource(show_spinner=False)
    def _load_finetuned_paraphraser(model_path: str):
        """Load a local Seq2Seq paraphrasing model (T5-small or BART-base) with its tokenizer.

        Returns (tokenizer, model, device_str).
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except Exception as e:
            raise RuntimeError(
                "transformers library is required. Add 'transformers' to requirements.txt and install."
            ) from e

        # Choose device if torch is available
        device = "cpu"
        try:
            import torch  # type: ignore
            device = "cuda" if hasattr(torch, "cuda") and torch.cuda.is_available() else "cpu"
        except Exception:
            pass

        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

        try:
            import torch  # type: ignore
            model = model.to(device)
        except Exception:
            # If torch missing, generation will still run on CPU-only environments in many cases
            device = "cpu"

        return tok, model, device

    def _extract_pdf_text(file) -> str:
        txt = ""
        try:
            if file is not None and PyPDF2 is not None:
                file.seek(0)
                reader = PyPDF2.PdfReader(file)
                pages = [p.extract_text() or "" for p in reader.pages]
                txt = "\n".join(pages).strip()
                # Normalize formatting a bit
                try:
                    t = txt.replace("\r\n", "\n")
                    t = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", t)
                    t = re.sub(r"\n{3,}", "\n\n", t)
                    t = re.sub(r"(?<!\n)\n(?!\n)", " ", t)
                    t = re.sub(r"[ \t]{2,}", " ", t)
                    t = re.sub(r" +([.,;:!?])", r"\1", t)
                    txt = t.strip()
                except Exception:
                    pass
        except Exception:
            txt = ""
        return txt

    def _generate_paraphrases(
        text: str,
        num_return_sequences: int = 1,
        max_length: int = 64,
        num_beams: int = 5,
        temperature: float = 1.0,
        do_sample: bool = False,
        top_p: float | None = None,
        top_k: int | None = None,
        repetition_penalty: float | None = None,
        length_penalty: float | None = None,
        instruction: str | None = None,
        model_path: str | None = None,
    ) -> list[str]:
        # Allow caller to override which local model folder to use; default to the Tab 10 selection
        use_model_dir = model_path or MODEL_DIR
        tok, model, device = _load_finetuned_paraphraser(use_model_dir)

        # Prepare inputs; apply T5 paraphrase prefix only for T5 models
        model_type = getattr(getattr(model, "config", None), "model_type", "")
        is_t5 = str(model_type).lower() == "t5"
        prefix = "paraphrase: " if is_t5 else ""
        extra = f" {instruction.strip()}" if instruction else ""
        input_text = prefix + text + extra
        enc = tok([input_text], return_tensors="pt", truncation=True)

        try:
            import torch  # type: ignore
            if device != "cpu":
                enc = {k: v.to(device) for k, v in enc.items()}
        except Exception:
            pass

        # Ensure beams >= return sequences when using beam search
        if not do_sample and num_return_sequences > num_beams:
            num_beams = max(num_beams, num_return_sequences)

        # Length controls
        max_len = int(max_length)
        # Set a more permissive min_length so short inputs don't get truncated unnaturally
        # ~60% of max, at least 4 tokens, and always leave room to stop before max
        min_len = max(4, min(int(max_len * 0.6), max_len - 8))
        # Ensure a minimum of new tokens regardless of input length
        try:
            min_new = max(6, min(max_len - 2, int(max_len * 0.5)))
        except Exception:
            min_new = None

        gen_kwargs = dict(
            max_length=max_len,
            min_length=min_len,
            num_return_sequences=int(num_return_sequences),
            # Anti-repetition for longer outputs
            no_repeat_ngram_size=3,
            encoder_no_repeat_ngram_size=3,
            eos_token_id=getattr(tok, 'eos_token_id', None),
            pad_token_id=getattr(tok, 'pad_token_id', None),
        )
        if min_new is not None:
            gen_kwargs["min_new_tokens"] = int(min_new)
        if do_sample:
            gen_kwargs.update(dict(
                do_sample=True,
                temperature=float(temperature),
                early_stopping=True,
            ))
        else:
            # For beam search, encourage longer outputs and avoid early stopping
            gen_kwargs.update(dict(
                num_beams=int(num_beams),
                length_penalty=float(length_penalty) if length_penalty is not None else 1.1,
                early_stopping=True,
            ))

        # Optional advanced controls
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = float(repetition_penalty)
        if length_penalty is not None:
            gen_kwargs["length_penalty"] = float(length_penalty)

        outputs = model.generate(**enc, **gen_kwargs)
        decoded = [tok.decode(out, skip_special_tokens=True) for out in outputs]
        # Deduplicate while preserving order
        seen = set()
        uniq = []
        for d in decoded:
            if d not in seen:
                seen.add(d)
                uniq.append(d)
        
        return uniq

    # -------- UI Controls --------
    colA, colB = st.columns(2)
    with colA:
        ft_input_type = st.radio("Choose input type:", ["Plain Text", "PDF File"], key="ftp_input_type")
        text_input = st.text_area("Paste your text here", height=220, key="ftp_text") if ft_input_type == "Plain Text" else ""
        pdf_file = None if ft_input_type == "Plain Text" else st.file_uploader("Upload a PDF", type=["pdf"], key="ftp_pdf")
    with colB:
        st.write("Generation settings:")
        complexity = st.selectbox("Complexity", ["Simple", "Medium", "Advanced"], index=0, help="Controls vocabulary and structure via decoding presets.")
        strategy = st.selectbox("Decoding Strategy", ["Auto (by Complexity)", "Beam search", "Sampling"], index=0)
        max_length = st.slider("Max length", 32, 128, 64, 4)

        # Default placeholders; filled via presets below
        num_beams = 5
        temperature = 1.0
        do_sample = False
        top_p = None
        top_k = None
        repetition_penalty = None
        length_penalty = None
        instruction = None

        def _complexity_preset(level: str) -> dict:
            if level == "Simple":
                return {
                    "do_sample": False,
                    "num_beams": 5,
                    "temperature": 1.0,
                    "top_p": None,
                    "top_k": None,
                    "repetition_penalty": 1.1,
                    "length_penalty": 0.9,
                    "instruction": "in simple language.",
                }
            elif level == "Medium":
                return {
                    "do_sample": True,
                    "num_beams": 1,
                    "temperature": 1.0,
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.0,
                    "length_penalty": 1.0,
                    "instruction": "clear and natural.",
                }
            else:  # Advanced
                return {
                    "do_sample": True,
                    "num_beams": 1,
                    "temperature": 1.2,
                    "top_p": 0.95,
                    "top_k": 100,
                    "repetition_penalty": 0.9,
                    "length_penalty": 1.2,
                    "instruction": "use advanced vocabulary and complex syntax.",
                }

        # Apply presets depending on strategy
        params = _complexity_preset(complexity)
        if strategy == "Auto (by Complexity)":
            st.caption(f"Auto strategy: settings derived from '{complexity}'.")
        elif strategy == "Beam search":
            params = _complexity_preset(complexity)
            params["num_beams"] = st.slider("Beams", 2, 8, int(params.get("num_beams", 5)))
            params["do_sample"] = False
            params["temperature"] = 1.0
        else:  # Sampling
            params = _complexity_preset(complexity)
            params["do_sample"] = True
            params["num_beams"] = 1
            params["temperature"] = st.slider("Temperature", 0.7, 1.5, float(params.get("temperature", 1.0)), 0.05)

        # Unpack params into local variables for generation call
        do_sample = params.get("do_sample", do_sample)
        num_beams = params.get("num_beams", num_beams)
        temperature = params.get("temperature", temperature)
        top_p = params.get("top_p", top_p)
        top_k = params.get("top_k", top_k)
        repetition_penalty = params.get("repetition_penalty", repetition_penalty)
        length_penalty = params.get("length_penalty", length_penalty)
        instruction = params.get("instruction", instruction)

    # Reference paraphrase input removed for this tab per request

    run = st.button("Generate Paraphrases", type="primary")

    if run:
        # Prepare text
        source_text = text_input.strip()
        if ft_input_type == "PDF File":
            if pdf_file is None:
                st.error("Please upload a PDF file.")
                st.stop()
            source_text = _extract_pdf_text(pdf_file)
            if not source_text:
                st.error("Unable to extract text from the uploaded PDF.")
                st.stop()

        if not source_text:
            st.error("Please enter some text to paraphrase.")
        else:
            with st.spinner("Generating paraphrases with fine-tuned model..."):
                try:
                    outs = _generate_paraphrases(
                        source_text,
                        num_return_sequences=1,
                        max_length=max_length,
                        num_beams=num_beams,
                        temperature=temperature,
                        do_sample=do_sample,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        length_penalty=length_penalty,
                        instruction=instruction,
                    )

                    # Store latest outputs for persistent language toggle
                    first_paraphrase = (outs[0] if outs else "")
                    st.session_state['ftp_last_original'] = source_text
                    st.session_state['ftp_last_paraphrase'] = first_paraphrase
                    st.session_state['ftp_last_model'] = model_choice
                    # Reset translations when new paraphrase generated
                    for k in ['ftp_original_hi','ftp_paraphrase_hi','ftp_translation_hash']:
                        st.session_state.pop(k, None)
                    st.session_state['ftp_language'] = 'English'

                    def _render_ftp_block():
                        orig_en = st.session_state.get('ftp_last_original','') or ''
                        para_en = st.session_state.get('ftp_last_paraphrase','') or ''
                        import hashlib, re as _re2
                        base_hash = hashlib.sha1((orig_en+'\n'+para_en).encode('utf-8', errors='ignore')).hexdigest()
                        if 'ftp_language_original' not in st.session_state:
                            st.session_state['ftp_language_original'] = 'English'
                        if 'ftp_language_paraphrase' not in st.session_state:
                            st.session_state['ftp_language_paraphrase'] = 'English'
                        col_tog1, col_tog2 = st.columns(2)
                        with col_tog1:
                            st.session_state['ftp_language_original'] = st.radio("Original Text Language", ["English","हिंदी"], horizontal=True, key="ftp_lang_orig_toggle")
                        with col_tog2:
                            st.session_state['ftp_language_paraphrase'] = st.radio("Paraphrase Language", ["English","हिंदी"], horizontal=True, key="ftp_lang_para_toggle")
                        need_hi = any(lang == 'हिंदी' for lang in [st.session_state['ftp_language_original'], st.session_state['ftp_language_paraphrase']])
                        if need_hi:
                            if st.session_state.get('ftp_translation_hash') != base_hash:
                                st.session_state.pop('ftp_original_hi', None)
                                st.session_state.pop('ftp_paraphrase_hi', None)
                            if ('ftp_original_hi' not in st.session_state and st.session_state['ftp_language_original']=='हिंदी') or \
                               ('ftp_paraphrase_hi' not in st.session_state and st.session_state['ftp_language_paraphrase']=='हिंदी'):
                                with st.spinner('Translating (may download models first time)...'):
                                    try:
                                        if st.session_state['ftp_language_original']=='हिंदी' and 'ftp_original_hi' not in st.session_state:
                                            st.session_state['ftp_original_hi'] = translate_text_full(orig_en,'en','hi')
                                        if st.session_state['ftp_language_paraphrase']=='हिंदी' and 'ftp_paraphrase_hi' not in st.session_state:
                                            st.session_state['ftp_paraphrase_hi'] = translate_text_full(para_en,'en','hi')
                                        st.session_state['ftp_translation_hash'] = base_hash
                                    except Exception as te:
                                        st.warning(f"Hindi translation failed: {te}")
                                        if 'ftp_original_hi' not in st.session_state:
                                            st.session_state['ftp_original_hi'] = orig_en
                                        if 'ftp_paraphrase_hi' not in st.session_state:
                                            st.session_state['ftp_paraphrase_hi'] = para_en
                        orig_display = st.session_state.get('ftp_original_hi') if st.session_state['ftp_language_original']=='हिंदी' else orig_en
                        para_display = st.session_state.get('ftp_paraphrase_hi') if st.session_state['ftp_language_paraphrase']=='हिंदी' else para_en
                        def _wc(t: str) -> int:
                            return len([w for w in _re2.findall(r"\w+", t)]) if t else 0
                        wc_orig_en = _wc(orig_en)
                        wc_para_en = _wc(para_en)
                        c1, c2 = st.columns(2)
                        with c1:
                            st.markdown("**Original**" + (" (हिंदी)" if st.session_state['ftp_language_original']=='हिंदी' else ""))
                            cap_o = f"{_wc(orig_display)} words"
                            if st.session_state['ftp_language_original']=='हिंदी':
                                cap_o += f" • EN: {wc_orig_en}"
                            st.caption(cap_o)
                            st.session_state['ftp_original_view'] = orig_display[:15000]
                            st.text_area("Original Text", value=st.session_state['ftp_original_view'], height=180, key="ftp_original_view")
                        with c2:
                            st.markdown("**Paraphrase**" + (" (हिंदी)" if st.session_state['ftp_language_paraphrase']=='हिंदी' else ""))
                            cap_p = f"{_wc(para_display)} words"
                            if st.session_state['ftp_language_paraphrase']=='हिंदी':
                                cap_p += f" • EN: {wc_para_en}"
                            st.caption(cap_p)
                            st.session_state['ftp_first_view'] = para_display
                            st.text_area("Paraphrase", value=st.session_state['ftp_first_view'], height=180, key="ftp_first_view")
                    st.markdown("### Original vs Paraphrase")
                    _render_ftp_block()

                    # Persist to history (original + result) for fine-tuned paraphrasing
                    try:
                        email = st.session_state.get("user_email") or (st.session_state.get("me") or {}).get("email")
                        if email and source_text and first_paraphrase:
                            hist_payload = {
                                "email": email,
                                "type": "paraphrase",
                                "original_text": source_text,
                                "result_text": first_paraphrase,
                                "model": f"fine-tuned-{('t5' if model_choice == 'T5-small' else 'bart')}",
                                "parameters": json.dumps({
                                    "strategy": strategy,
                                    "complexity": complexity,
                                    "max_length": max_length,
                                    "do_sample": do_sample,
                                    "num_beams": num_beams,
                                    "temperature": temperature,
                                    "top_p": top_p,
                                    "top_k": top_k,
                                    "repetition_penalty": repetition_penalty,
                                    "length_penalty": length_penalty,
                                }),
                            }
                            requests.post(f"{BACKEND_URL}/history", json=hist_payload, timeout=15)
                    except Exception:
                        pass

                    # Reference-based evaluation removed for this tab
                except Exception as e:
                    st.error(f"Failed to generate paraphrases: {e}")

    # Persistent display with language toggle after generation (when user toggles without regenerating)
    if st.session_state.get('ftp_last_paraphrase') and not run:
        def _render_ftp_block_external():
            # External persistent renderer with independent toggles
            orig_en = st.session_state.get('ftp_last_original','') or ''
            para_en = st.session_state.get('ftp_last_paraphrase','') or ''
            import hashlib, re as _re2
            base_hash = hashlib.sha1((orig_en+'\n'+para_en).encode('utf-8', errors='ignore')).hexdigest()
            if 'ftp_language_original' not in st.session_state:
                st.session_state['ftp_language_original'] = 'English'
            if 'ftp_language_paraphrase' not in st.session_state:
                st.session_state['ftp_language_paraphrase'] = 'English'
            col_tog1, col_tog2 = st.columns(2)
            with col_tog1:
                st.session_state['ftp_language_original'] = st.radio("Original Text Language", ["English","हिंदी"], horizontal=True, key="ftp_lang_orig_toggle")
            with col_tog2:
                st.session_state['ftp_language_paraphrase'] = st.radio("Paraphrase Language", ["English","हिंदी"], horizontal=True, key="ftp_lang_para_toggle")
            need_hi = any(lang == 'हिंदी' for lang in [st.session_state['ftp_language_original'], st.session_state['ftp_language_paraphrase']])
            if need_hi:
                if st.session_state.get('ftp_translation_hash') != base_hash:
                    st.session_state.pop('ftp_original_hi', None)
                    st.session_state.pop('ftp_paraphrase_hi', None)
                if ('ftp_original_hi' not in st.session_state and st.session_state['ftp_language_original']=='हिंदी') or \
                   ('ftp_paraphrase_hi' not in st.session_state and st.session_state['ftp_language_paraphrase']=='हिंदी'):
                    with st.spinner('Translating (may download models first time)...'):
                        try:
                            if st.session_state['ftp_language_original']=='हिंदी' and 'ftp_original_hi' not in st.session_state:
                                st.session_state['ftp_original_hi'] = translate_text_full(orig_en,'en','hi')
                            if st.session_state['ftp_language_paraphrase']=='हिंदी' and 'ftp_paraphrase_hi' not in st.session_state:
                                st.session_state['ftp_paraphrase_hi'] = translate_text_full(para_en,'en','hi')
                            st.session_state['ftp_translation_hash'] = base_hash
                        except Exception as te:
                            st.warning(f"Hindi translation failed: {te}")
                            if 'ftp_original_hi' not in st.session_state:
                                st.session_state['ftp_original_hi'] = orig_en
                            if 'ftp_paraphrase_hi' not in st.session_state:
                                st.session_state['ftp_paraphrase_hi'] = para_en
            orig_display = st.session_state.get('ftp_original_hi') if st.session_state['ftp_language_original']=='हिंदी' else orig_en
            para_display = st.session_state.get('ftp_paraphrase_hi') if st.session_state['ftp_language_paraphrase']=='हिंदी' else para_en
            def _wc(t: str) -> int:
                return len([w for w in _re2.findall(r"\w+", t)]) if t else 0
            wc_orig_en = _wc(orig_en)
            wc_para_en = _wc(para_en)
            st.markdown("### Original vs Paraphrase")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Original**" + (" (हिंदी)" if st.session_state['ftp_language_original']=='हिंदी' else ""))
                cap_o = f"{_wc(orig_display)} words"
                if st.session_state['ftp_language_original']=='हिंदी':
                    cap_o += f" • EN: {wc_orig_en}"
                st.caption(cap_o)
                st.session_state['ftp_original_view'] = orig_display[:15000]
                st.text_area("Original Text", value=st.session_state['ftp_original_view'], height=180, key="ftp_original_view")
            with c2:
                st.markdown("**Paraphrase**" + (" (हिंदी)" if st.session_state['ftp_language_paraphrase']=='हिंदी' else ""))
                cap_p = f"{_wc(para_display)} words"
                if st.session_state['ftp_language_paraphrase']=='हिंदी':
                    cap_p += f" • EN: {wc_para_en}"
                st.caption(cap_p)
                st.session_state['ftp_first_view'] = para_display
                st.text_area("Paraphrase", value=st.session_state['ftp_first_view'], height=180, key="ftp_first_view")
        _render_ftp_block_external()

## -------------------- Paraphrase Dataset Evaluation tab --------------------
with tabs[10]:  # Tab 10 - Paraphrase Dataset Eval
    token = st.session_state.get("token")
    if not token:
        st.markdown(
            """
            <div style='text-align:center;padding:50px 0;'>
                <h3 style='color:#334155;margin-bottom:10px;'>Sign in required</h3>
                <p style='color:#64748b;'>Please sign in to access paraphrase dataset evaluation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    st.markdown("""
        <div style="text-align:center;margin-bottom:20px">
            <img src="https://cdn-icons-png.flaticon.com/512/4471/4471606.png" width="60">
            <h3 style="font-weight:600;color:#334155">Evaluate Fine-tuned Paraphraser on Dataset</h3>
            <p style="color:#64748b;font-size:14px;">Use a built-in CSV (<code>train.csv</code>, <code>validation.csv</code>, <code>test.csv</code>) from the <code>paraphrase dataset/</code> folder or upload your own. We auto-detect columns like <code>text</code>/<code>input_text</code> and <code>reference</code>/<code>target_text</code> (case-insensitive). We'll generate paraphrases using your selected fine-tuned model (T5-small or BART-base) and compute BLEU-1..4, Perplexity, and Readability deltas.</p>
        </div>
    """, unsafe_allow_html=True)

    # Local fine-tuned model folders (same as Tab 10)
    T5_PARAPHRASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "saved_paraphrasing_model"))
    BART_PARAPHRASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bart_paraphrasing_model"))

    # Explicit model selection for this tab to ensure outputs come only from the chosen model
    model_choice_eval = st.selectbox("Model", ["T5-small", "BART-base"], index=0, key="ftp_ds_model_choice_tab11")
    MODEL_DIR_EVAL = T5_PARAPHRASE_DIR if model_choice_eval == "T5-small" else BART_PARAPHRASE_DIR
    if not os.path.isdir(MODEL_DIR_EVAL):
        st.error(
            f"Could not find the selected paraphrasing model folder for {model_choice_eval} at '{MODEL_DIR_EVAL}'. "
            "Please ensure the directory exists and includes the tokenizer/model files."
        )
        st.stop()

    # Source selection similar to summarization dataset eval
    src_para = st.radio("Data source", ["Built-in dataset", "Upload CSV"], horizontal=True, key="ftp_ds_source")

    selected_df = None
    if src_para == "Built-in dataset":
        ds_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "paraphrase dataset"))
        options = []
        path_map = {}
        for name in ["train.csv", "validation.csv", "test.csv"]:
            p = os.path.join(ds_dir, name)
            if os.path.isfile(p):
                options.append(name)
                path_map[name] = p
        if not options:
            st.warning("No built-in dataset files found in 'paraphrase dataset/'. Switch to Upload CSV.")
        else:
            choice = st.selectbox("Choose file", options, key="ftp_ds_choose_builtin")
            if choice:
                import pandas as pd
                try:
                    selected_df = pd.read_csv(path_map[choice])
                except Exception as e:
                    st.error(f"Failed to read CSV: {e}")
    else:
        ds_file = st.file_uploader("Upload dataset CSV (typical columns: input_text,target_text or text,reference)", type=["csv"], key="ftp_ds_eval")
        if ds_file is not None:
            import pandas as pd
            try:
                selected_df = pd.read_csv(ds_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if selected_df is not None:
        df = selected_df
        # Validate and map columns (case-insensitive)
        lower_cols = {c.lower(): c for c in df.columns}
        text_candidates = ["text", "input_text", "source", "article"]
        ref_candidates = ["reference", "target_text", "target", "paraphrase", "highlights"]
        text_col = next((lower_cols[c] for c in text_candidates if c in lower_cols), None)
        ref_col = next((lower_cols[c] for c in ref_candidates if c in lower_cols), None)
        if not text_col or not ref_col:
            st.error("CSV must contain text and reference columns (e.g., input_text/target_text or text/reference).")
            st.stop()
        df = df[[text_col, ref_col]].rename(columns={text_col: "text", ref_col: "reference"})

        # Row range selection
        total_rows = len(df)
        c1, c2 = st.columns(2)
        with c1:
            start_row = st.number_input("Start row", min_value=1, max_value=max(1, total_rows), value=1, step=1)
        with c2:
            default_end = min(total_rows, int(start_row) + 19)
            end_row = st.number_input("End row (inclusive)", min_value=int(start_row), max_value=max(1, total_rows), value=default_end, step=1)
        st.caption("Tip: Indices are 1-based and inclusive. Example: 1–100 evaluates the first 100 rows.")

        s_idx = max(0, int(start_row) - 1)
        e_idx = min(total_rows, int(end_row))
        if s_idx >= e_idx:
            st.error("Start row must be less than or equal to End row.")
            st.stop()
        df = df.iloc[s_idx:e_idx].copy()

        # Decoding/complexity options reused from the fine-tuned tab
        st.subheader("Paraphrasing Settings")
        colA, colB = st.columns(2)
        with colA:
            complexity_ds = st.selectbox("Complexity", ["Simple", "Medium", "Advanced"], index=0, key="ftp_ds_complexity")
        with colB:
            strategy_ds = st.selectbox("Decoding", ["Auto (by Complexity)", "Beam search", "Sampling"], index=0, key="ftp_ds_strategy")
        strict_meaning = st.checkbox("Strict meaning preservation (recommended)", value=True, help="Prioritize semantic faithfulness by using beam search and selecting the most similar candidate.")
        # Output length is auto-adjusted per input; no manual max length control in this tab.

        # Reuse the preset logic
        def _preset(level: str) -> dict:
            if level == "Simple":
                return {"do_sample": False, "num_beams": 5, "temperature": 1.0, "top_p": None, "top_k": None, "repetition_penalty": 1.1, "length_penalty": 0.9, "instruction": "in simple language."}
            elif level == "Medium":
                return {"do_sample": True, "num_beams": 1, "temperature": 1.0, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.0, "length_penalty": 1.0, "instruction": "clear and natural."}
            else:
                return {"do_sample": True, "num_beams": 1, "temperature": 1.2, "top_p": 0.95, "top_k": 100, "repetition_penalty": 0.9, "length_penalty": 1.2, "instruction": "use advanced vocabulary and complex syntax."}

        params_ds = _preset(complexity_ds)
        if strategy_ds == "Beam search":
            params_ds["num_beams"] = st.slider("Beams", 2, 8, int(params_ds.get("num_beams", 5)), key="ftp_ds_beams")
            params_ds["do_sample"] = False
            params_ds["temperature"] = 1.0
        elif strategy_ds == "Sampling":
            params_ds["do_sample"] = True
            params_ds["num_beams"] = 1
            params_ds["temperature"] = st.slider("Temperature", 0.7, 1.5, float(params_ds.get("temperature", 1.0)), 0.05, key="ftp_ds_temp")

        # Override for strict meaning: deterministic beams, meaning-preserving instruction
        if strict_meaning:
            params_ds["do_sample"] = False
            params_ds["temperature"] = 1.0
            params_ds["num_beams"] = max(6, int(params_ds.get("num_beams", 5)))
            params_ds["instruction"] = "Preserve the exact meaning. Keep all information. Output one sentence."

        # Optional: semantic similarity model for re-ranking (cached)
        @st.cache_resource(show_spinner=False)
        def _load_embedder():
            try:
                from sentence_transformers import SentenceTransformer
                return SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                return None

        def _semantic_similarity(a: str, b: str) -> float:
            try:
                embedder = _load_embedder()
                if embedder is None:
                    return 0.0
                embs = embedder.encode([a, b], normalize_embeddings=True)
                import numpy as _np
                return float((_np.array(embs[0]) * _np.array(embs[1])).sum())
            except Exception:
                return 0.0

        def _is_incomplete_sentence(orig: str, cand: str) -> bool:
            import re as _re
            c = (cand or "").strip()
            if not c:
                return True
            ends_ok = c.endswith(('.', '!', '?'))
            wc_o = len([w for w in _re.findall(r"\w+", orig)]) or 1
            wc_c = len([w for w in _re.findall(r"\w+", c)])
            too_short = wc_c < max(4, int(0.6 * wc_o))
            return (not ends_ok) or too_short

        # Heuristic: set output max_length to be similar to input length
        # We approximate decoder tokens from input words using a small factor, then clamp to safe bounds.
        def _suggest_max_len(text: str) -> int:
            try:
                import re as _re
                wc = len([w for w in _re.findall(r"\w+", text)])
            except Exception:
                wc = max(1, len(text) // 5)
            # For short sentences, give enough room to finish; for longer, allow more tokens
            approx_tokens = int(max(1, wc) * 1.8)  # scale up to reduce truncation risk
            return max(16, min(196, approx_tokens))

        show_pills([
            (f"Rows: {int(start_row)}–{int(end_row)} ({len(df)} samples)", "info"),
            (f"Model: {model_choice_eval}", "acc"),
            ("Strict meaning: ON" if strict_meaning else "Strict meaning: OFF", "warn" if not strict_meaning else "ok")
        ])
        st.caption("Output length is auto-adjusted per input. Click 'Run Paraphrase Evaluation' to start.")
        run_ds = st.button("Run Paraphrase Evaluation", type="primary")

        if run_ds:
            import pandas as pd
            rows_out = []
            prog = st.progress(0)
            for i, (idx, row) in enumerate(df.iterrows(), start=1):
                original = str(row["text"]) if pd.notna(row["text"]) else ""
                reference = str(row["reference"]) if pd.notna(row["reference"]) else ""
                if not original or not reference:
                    prog.progress(min(i / len(df), 1.0))
                    continue
                try:
                    # Generate paraphrase using the local fine-tuned model
                    max_len_for_row = _suggest_max_len(original)
                    num_ret = 5 if strict_meaning else 1
                    outs = _generate_paraphrases(
                        original,
                        num_return_sequences=int(num_ret),
                        max_length=int(max_len_for_row),
                        num_beams=int(params_ds.get("num_beams", 5)),
                        temperature=float(params_ds.get("temperature", 1.0)),
                        do_sample=bool(params_ds.get("do_sample", False)),
                        top_p=params_ds.get("top_p"),
                        top_k=params_ds.get("top_k"),
                        repetition_penalty=params_ds.get("repetition_penalty"),
                        length_penalty=params_ds.get("length_penalty"),
                        instruction=params_ds.get("instruction"),
                        model_path=MODEL_DIR_EVAL,
                    )
                    gen = outs[0] if outs else ""

                    # If strict meaning: select best candidate by combined semantic+ROUGE-L vs original
                    if strict_meaning and outs:
                        try:
                            # First compute semantic similarity for all candidates
                            sims = {cand: _semantic_similarity(original, cand) for cand in outs}
                            # Pre-select top-3 by semantic similarity to cut down API calls
                            top_cands = sorted(outs, key=lambda c: sims.get(c, 0.0), reverse=True)[:3]
                            # Compute ROUGE-L for the top candidates
                            rougeL = {}
                            for cand in top_cands:
                                payload = {"reference": original, "candidate": cand}
                                resp = requests.post(f"{BACKEND_URL}/evaluate/rouge", json=payload, timeout=60)
                                rjson = resp.json() if resp.ok else {}
                                scores = rjson.get("scores", {})
                                rl = scores.get("rougeL", {})
                                rougeL[cand] = float(rl.get("f1", 0.0)) if isinstance(rl, dict) else 0.0
                            # Combine: 0.6 * semantic + 0.4 * ROUGE-L, penalize incomplete endings
                            def _combined_score(cand: str) -> float:
                                base = 0.6 * sims.get(cand, 0.0) + 0.4 * rougeL.get(cand, 0.0)
                                return base - (0.1 if _is_incomplete_sentence(original, cand) else 0.0)
                            best_cand = max(top_cands, key=_combined_score)
                            gen = best_cand
                        except Exception:
                            # Fallback to original first candidate
                            gen = outs[0]

                    # If appears incomplete, retry with larger budget and pick the best finished candidate
                    if strict_meaning and _is_incomplete_sentence(original, gen):
                        try:
                            retry_max = min(256, int(max_len_for_row * 1.4))
                            retry_outs = _generate_paraphrases(
                                original,
                                num_return_sequences=3,
                                max_length=int(retry_max),
                                num_beams=int(params_ds.get("num_beams", 6)),
                                temperature=1.0,
                                do_sample=False,
                                instruction=params_ds.get("instruction"),
                                model_path=MODEL_DIR_EVAL,
                            )
                            # Prefer completed sentences; reuse the combined scorer when possible
                            if retry_outs:
                                # If embedder available, compute scores else pick the longest with punctuation
                                sims_retry = {c: _semantic_similarity(original, c) for c in retry_outs}
                                # Quick ROUGE-L on top-2
                                top2 = sorted(retry_outs, key=lambda c: sims_retry.get(c, 0.0), reverse=True)[:2]
                                rouge_retry = {}
                                for c in top2:
                                    payload = {"reference": original, "candidate": c}
                                    resp = requests.post(f"{BACKEND_URL}/evaluate/rouge", json=payload, timeout=60)
                                    rjson = resp.json() if resp.ok else {}
                                    scores = rjson.get("scores", {})
                                    rl = scores.get("rougeL", {})
                                    rouge_retry[c] = float(rl.get("f1", 0.0)) if isinstance(rl, dict) else 0.0
                                def _score_retry(cand: str) -> float:
                                    base = 0.6 * sims_retry.get(cand, 0.0) + 0.4 * rouge_retry.get(cand, 0.0)
                                    return base - (0.15 if _is_incomplete_sentence(original, cand) else 0.0)
                                gen = max(retry_outs, key=_score_retry)
                                # Ensure final punctuation
                                if not gen.strip().endswith(('.', '!', '?')):
                                    gen = gen.strip() + '.'
                        except Exception:
                            # Last-resort: add a period if missing
                            if gen and not gen.strip().endswith(('.', '!', '?')):
                                gen = gen.strip() + '.'

                    # Metrics via backend
                    metrics_payload = {
                        "reference": reference,
                        "candidate": gen,
                        "original_text": original,
                    }
                    mresp = requests.post(f"{BACKEND_URL}/evaluate/metrics", json=metrics_payload, timeout=120)
                    mjson = mresp.json()

                    # ROUGE (F1) via backend
                    try:
                        r_payload = {"reference": reference, "candidate": gen}
                        r_resp = requests.post(f"{BACKEND_URL}/evaluate/rouge", json=r_payload, timeout=60)
                        r_json = r_resp.json() if r_resp.ok else {}
                        r_scores = r_json.get("scores", {}) if isinstance(r_json, dict) else {}
                        r1_f1 = float(r_scores.get("rouge1", {}).get("f1", float("nan"))) if isinstance(r_scores.get("rouge1", {}), dict) else float("nan")
                        r2_f1 = float(r_scores.get("rouge2", {}).get("f1", float("nan"))) if isinstance(r_scores.get("rouge2", {}), dict) else float("nan")
                        rL_f1 = float(r_scores.get("rougeL", {}).get("f1", float("nan"))) if isinstance(r_scores.get("rougeL", {}), dict) else float("nan")
                    except Exception:
                        r1_f1 = r2_f1 = rL_f1 = float("nan")

                    def _get(d, path, default=None):
                        cur = d
                        try:
                            for p in path:
                                cur = cur[p]
                            return cur
                        except Exception:
                            return default

                    rows_out.append({
                        "index": idx,
                        "text": original,
                        "generated": gen,
                        "reference": reference,
                        # BLEU
                        "bleu_1": _get(mjson, ["bleu", "bleu_1"], float("nan")),
                        "bleu_2": _get(mjson, ["bleu", "bleu_2"], float("nan")),
                        "bleu_3": _get(mjson, ["bleu", "bleu_3"], float("nan")),
                        "bleu_4": _get(mjson, ["bleu", "bleu_4"], float("nan")),
                        "bleu": _get(mjson, ["bleu", "bleu"], float("nan")),
                        # ROUGE (F1)
                        "rouge1_f1": r1_f1,
                        "rouge2_f1": r2_f1,
                        "rougeL_f1": rL_f1,
                        # Perplexities
                        "ppl_candidate": _get(mjson, ["perplexity_candidate", "perplexity"], _get(mjson, ["perplexity", "perplexity"], float("nan"))),
                        "ppl_reference": _get(mjson, ["perplexity_reference", "perplexity"], float("nan")),
                        # Readability deltas (candidate vs reference)
                        "delta_flesch_reading_ease": _get(mjson, ["readability_ref_candidate", "delta", "flesch_reading_ease"], float("nan")),
                        "delta_flesch_kincaid": _get(mjson, ["readability_ref_candidate", "delta", "flesch_kincaid_grade"], float("nan")),
                        "delta_gunning_fog": _get(mjson, ["readability_ref_candidate", "delta", "gunning_fog"], float("nan")),
                        "delta_smog_index": _get(mjson, ["readability_ref_candidate", "delta", "smog_index"], float("nan")),
                    })
                except Exception as e:
                    st.warning(f"Row {idx} failed: {e}")
                finally:
                    prog.progress(min(i / len(df), 1.0))

            if rows_out:
                import pandas as pd
                rdf = pd.DataFrame(rows_out)
                st.success("Paraphrase evaluation complete.")

                st.markdown("### Sample Results")
                sample_n = min(5, len(rdf))
                for i in range(sample_n):
                    r = rdf.iloc[i]
                    st.markdown(f"#### Sample #{int(r['index'])}")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Original**")
                        st.write(r["text"])
                    with c2:
                        st.markdown("**Paraphrase**")
                        st.write(r["generated"])
                    with c3:
                        st.markdown("**Reference**")
                        st.write(r["reference"])
                    m1, m2, m3 = st.columns(3)
                    with m1:
                        st.metric("BLEU-1", f"{r['bleu_1']:.4f}" if pd.notna(r['bleu_1']) else "-")
                        st.metric("BLEU-4", f"{r['bleu_4']:.4f}" if pd.notna(r['bleu_4']) else "-")
                    with m2:
                        st.metric("PPL (gen)", f"{r['ppl_candidate']:.2f}" if pd.notna(r['ppl_candidate']) else "-")
                        st.metric("PPL (ref)", f"{r['ppl_reference']:.2f}" if pd.notna(r['ppl_reference']) else "-")
                    with m3:
                        st.metric("Δ Flesch Ease (gen-ref)", f"{r['delta_flesch_reading_ease']:+.2f}" if pd.notna(r['delta_flesch_reading_ease']) else "-")
                        st.metric("Δ SMOG (gen-ref)", f"{r['delta_smog_index']:+.2f}" if pd.notna(r['delta_smog_index']) else "-")
                    st.divider()

                ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_csv = rdf.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results CSV",
                    data=out_csv,
                    file_name=f"paraphrase_eval_results_{ts}.csv",
                    mime="text/csv",
                )

                st.markdown("### Aggregates")
                agg = rdf[[
                    "bleu_1","bleu_2","bleu_3","bleu_4","bleu",
                    "rouge1_f1","rouge2_f1","rougeL_f1",
                    "ppl_candidate","ppl_reference",
                    "delta_flesch_reading_ease","delta_flesch_kincaid","delta_gunning_fog","delta_smog_index"
                ]].mean(numeric_only=True)
                st.dataframe(agg.to_frame("mean").round(4))

                # --- Visualizations ---
                st.markdown("### Visualizations")
                try:
                    import pandas as pd  # ensure available in this scope
                    if plt is None:
                        st.warning("matplotlib not installed; run 'pip install matplotlib' to view charts.")
                    else:
                        # 1) BLEU grouped bar
                        bleu_cols = ["bleu_1","bleu_2","bleu_3","bleu_4","bleu"]
                        bleu_vals = [float(agg.get(c, float('nan'))) for c in bleu_cols]
                        bleu_vals = [0.0 if pd.isna(v) else v for v in bleu_vals]
                        fig1, ax1 = plt.subplots(figsize=(6, 3.0))
                        x = list(range(len(bleu_cols))) if np is None else np.arange(len(bleu_cols))
                        ax1.bar(x, bleu_vals, color="#3b82f6")
                        ax1.set_xticks(x)
                        ax1.set_xticklabels([c.upper() for c in bleu_cols])
                        ax1.set_ylim(0, 1)
                        ax1.set_ylabel("Score")
                        ax1.set_title("Mean BLEU Scores")
                        for xi, v in zip(x, bleu_vals):
                            ax1.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
                        # defer rendering; will place in columns

                        # 1b) ROUGE (F1) grouped bar
                        rouge_cols = ["rouge1_f1","rouge2_f1","rougeL_f1"]
                        rouge_labels = ["ROUGE-1 F1","ROUGE-2 F1","ROUGE-L F1"]
                        rouge_vals = [float(agg.get(c, float('nan'))) for c in rouge_cols]
                        rouge_vals = [0.0 if pd.isna(v) else v for v in rouge_vals]
                        fig1b, ax1b = plt.subplots(figsize=(6, 3.0))
                        x1b = list(range(len(rouge_cols))) if np is None else np.arange(len(rouge_cols))
                        ax1b.bar(x1b, rouge_vals, color="#06b6d4")
                        ax1b.set_xticks(x1b)
                        ax1b.set_xticklabels(rouge_labels)
                        ax1b.set_ylim(0, 1)
                        ax1b.set_ylabel("F1")
                        ax1b.set_title("Mean ROUGE (F1)")
                        for xi, v in zip(x1b, rouge_vals):
                            ax1b.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)
                        # defer rendering; will place in columns

                        # 2) Perplexity bar
                        ppl_cols = ["ppl_candidate","ppl_reference"]
                        ppl_vals = [float(agg.get(c, float('nan'))) for c in ppl_cols]
                        ppl_vals = [0.0 if pd.isna(v) else v for v in ppl_vals]
                        fig2, ax2 = plt.subplots(figsize=(5.5, 3.0))
                        x2 = list(range(len(ppl_cols))) if np is None else np.arange(len(ppl_cols))
                        ax2.bar(x2, ppl_vals, color=["#10b981", "#f59e0b"])
                        ax2.set_xticks(x2)
                        ax2.set_xticklabels([c.replace("ppl_","PPL ").title() for c in ppl_cols])
                        ax2.set_ylabel("Perplexity (lower is better)")
                        ax2.set_title("Mean Perplexity")
                        for xi, v in zip(x2, ppl_vals):
                            ax2.text(xi, v + 0.01, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

                        # Render 4 charts in two columns (2 charts per column)
                        cols = st.columns(2)
                        with cols[0]:
                            st.pyplot(fig1, clear_figure=True)   # BLEU
                            st.pyplot(fig1b, clear_figure=True)  # ROUGE F1
                        with cols[1]:
                            st.pyplot(fig2, clear_figure=True)   # Perplexity

                        # 3) Readability deltas bar
                        read_cols = [
                            "delta_flesch_reading_ease","delta_flesch_kincaid",
                            "delta_gunning_fog","delta_smog_index"
                        ]
                        read_labels = ["Δ Flesch Ease","Δ F-K Grade","Δ Gunning Fog","Δ SMOG"]
                        read_vals = [float(agg.get(c, float('nan'))) for c in read_cols]
                        read_vals = [0.0 if pd.isna(v) else v for v in read_vals]
                        fig3, ax3 = plt.subplots(figsize=(6, 3.0))
                        x3 = list(range(len(read_cols))) if np is None else np.arange(len(read_cols))
                        ax3.bar(x3, read_vals, color="#8b5cf6")
                        ax3.set_xticks(x3)
                        ax3.set_xticklabels(read_labels)
                        ax3.axhline(0, color="#94a3b8", linewidth=1)
                        ax3.set_ylabel("Delta (gen - ref)")
                        ax3.set_title("Mean Readability Deltas")
                        for xi, v in zip(x3, read_vals):
                            ax3.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}", ha='center', va='bottom' if v>=0 else 'top', fontsize=8)
                        # Readability chart stacked below in right column
                        with cols[1]:
                            st.pyplot(fig3, clear_figure=True)

                        # Removed BLEU-1 distribution histogram as requested
                except Exception as viz_e:
                    st.info(f"Could not render charts: {viz_e}")

# Global footer
st.markdown(
        """
<div class='tm-footer'>
    <strong>TextMorph</strong> • Built with Streamlit & FastAPI • © 2025
</div>
""",
        unsafe_allow_html=True,
)
