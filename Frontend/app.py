import streamlit as st
import requests
import re
from io import BytesIO
from typing import Optional

try:
    import textstat  # type: ignore
except Exception:
    textstat = None  # graceful fallback

try:
    import PyPDF2  # type: ignore
except Exception:
    PyPDF2 = None

try:
    from docx import Document  # type: ignore
except Exception:
    Document = None

try:
    from pdfminer.high_level import extract_text as pdfminer_extract_text  # type: ignore
except Exception:
    pdfminer_extract_text = None

API_URL = "http://127.0.0.1:8000"  # FastAPI backend


def api_post(path: str, payload: dict, timeout: int = 5, token: str | None = None):
    """POST to backend with robust error handling.

    Returns tuple (ok: bool, data: dict|str)
    - ok True => data is JSON body
    - ok False => data is error string for user display
    """
    url = f"{API_URL}{path}"
    try:
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.post(url, json=payload, timeout=timeout, headers=headers)
        body = None
        try:
            body = resp.json()
        except Exception:
            body = None
        if 200 <= resp.status_code < 300:
            return True, (body if body is not None else {})
        if body and isinstance(body, dict):
            detail = body.get("detail") or body.get("message")
            if isinstance(detail, list):
                detail = "; ".join([str(d.get("msg", d)) for d in detail])
            if detail:
                return False, f"{resp.status_code}: {detail}"
        return False, f"{resp.status_code}: {resp.reason or 'Request failed'}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend. If it is not running at http://127.0.0.1:8000"
    except requests.exceptions.Timeout:
        return False, "Backend request timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}"


def api_get(path: str, timeout: int = 5, token: str | None = None):
    url = f"{API_URL}{path}"
    try:
        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        resp = requests.get(url, timeout=timeout, headers=headers)
        body = None
        try:
            body = resp.json()
        except Exception:
            body = None
        if 200 <= resp.status_code < 300:
            return True, (body if body is not None else {})
        if body and isinstance(body, dict):
            detail = body.get("detail") or body.get("message")
            if isinstance(detail, list):
                detail = "; ".join([str(d.get("msg", d)) for d in detail])
            if detail:
                return False, f"{resp.status_code}: {detail}"
        return False, f"{resp.status_code}: {resp.reason or 'Request failed'}"
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to backend. Make sure it's running at http://127.0.0.1:8000"
    except requests.exceptions.Timeout:
        return False, "Backend request timed out."
    except requests.exceptions.RequestException as e:
        return False, f"Request error: {e}"


st.set_page_config(page_title="Text morph advanced summarisation using AI", layout="centered")

# Apply consistent app-wide styling with optimized CSS
st.markdown("""
<style>
    /* Base styling with custom font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="st-"] { font-family: 'Inter', sans-serif; }
    
    /* Header styling with gradient text */
    h1 { 
        color: #1e293b; font-weight: 700; font-size: 2.4rem; margin-bottom: 1rem;
        background: linear-gradient(90deg, #1e293b, #334155);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    h1 span.highlight { background: linear-gradient(90deg, #1d4ed8, #3b82f6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    h2, h3 { font-weight: 600; color: #1e293b; }
    
    /* Streamlined tabs styling */
    .stTabs { margin-top: 1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #f8fafc; border-radius: 8px 8px 0 0; border-bottom: 1px solid #e2e8f0; }
    .stTabs [data-baseweb="tab"] { height: 40px; background-color: #f1f5f9; border-radius: 6px 6px 0 0; font-weight: 500; }
    .stTabs [aria-selected="true"] { background-color: #e0e7ff !important; color: #3b82f6 !important; border-bottom: 2px solid #3b82f6; }
    
    /* Form and button styling */
    [data-testid="stForm"] { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 1.5rem; }
    button[kind="primary"] { background-color: #3b82f6; border-radius: 6px; font-weight: 500; }
    button[kind="primary"]:hover { background-color: #2563eb; }
    
    /* Card styling */
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 20px; margin-bottom: 20px; }
    
    /* Misc improvements */
    .stAlert { border-radius: 6px; }
    .block-container { max-width: 1000px; padding-top: 2rem; }
    footer { visibility: hidden; }
    
    /* Metrics styling (matching the image) */
    .metric-card { background: white; border: 1px solid #e2e8f0; border-radius: 8px; padding: 16px; text-align: center; }
    .metric-value { font-size: 32px; font-weight: 700; margin-bottom: 8px; }
    .metric-label { font-size: 14px; color: #64748b; }
    .metric-green .metric-value { color: #10b981; }
    .metric-yellow .metric-value { color: #f59e0b; }
    .metric-red .metric-value { color: #ef4444; }
    
    /* Indicator matching the image */
    .indicator { 
        height: 4px; 
        border-radius: 2px; 
        margin: 10px 0; 
        background: linear-gradient(to right, #10b981, #f59e0b, #ef4444);
        position: relative; 
    }
    .indicator::after { 
        content: ''; 
        position: absolute; 
        width: 12px; 
        height: 12px; 
        background: white; 
        border: 2px solid #3b82f6; 
        border-radius: 50%; 
        top: -4px; 
    }
    .indicator.easy::after { left: 15%; } 
    .indicator.medium::after { left: 50%; } 
    .indicator.hard::after { left: 85%; }
    
    /* Upload area */
    .upload-container { border: 2px dashed #94a3b8; border-radius: 8px; padding: 20px; text-align: center; }
    
    /* Feature card */
    .feature-card { background: #f0f9ff; border-left: 4px solid #3b82f6; border-radius: 8px; padding: 16px; margin-top: 24px; }
    .feature-title { font-weight: 600; color: #1e40af; margin-bottom: 12px; }
    .feature-item { display: flex; align-items: center; margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# App header with simplified design
st.markdown("""
<div style="text-align: center; margin-bottom: 2rem;">
    <img src="https://cdn-icons-png.flaticon.com/512/3368/3368235.png" width="70" style="margin-bottom: 0.5rem;">
    <h1>Text<span class="highlight">morph</span></h1>
    <p style="font-size:1.1rem;color:#475569;margin-top:-0.5rem;font-weight:500;">
        Advanced Text Summarisation Using AI
    </p>
</div>
""", unsafe_allow_html=True)

# Subtle style adjustments
st.markdown(
    """
    <style>
    .block-container {max-width: 760px;}
    .stForm button[kind="primary"] {min-width: 140px;}
    </style>
    """,
    unsafe_allow_html=True,
)


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


tabs = st.tabs(["Sign in", "Register", "Forgot password", "Profile", "Readability"])

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
        if not email:
            st.error("Please enter your email.")
        elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
            st.error("Please enter a valid email address.")
        elif not password:
            st.error("Please enter your password.")
        else:
            ok, data = api_post("/login", {"email": email, "password": password})
            if ok:
                st.success("Login successful")
                st.session_state["token"] = data.get("access_token")
                st.balloons()
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
                # Compute metrics
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

                # Colorful metric cards with improved indicators
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
                
                # Calculate beginner score (higher for simpler text)
                beg = 0
                if fk_ease >= 70:  # High Flesch score = easy = beginner-friendly
                    beg += min(int(fk_ease * 0.8), 80)  # Cap at 80
                if gunning <= 10:  # Low Gunning = easier
                    beg += max(0, 40 - (gunning * 4))
                if smog <= 9:  # Low SMOG = easier
                    beg += max(0, 40 - (smog * 4))
                beg = min(100, max(10, beg))  # Min 10, max 100
                
                # Calculate intermediate score
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
                
                # Calculate advanced score
                adv = 0
                if fk_ease < 50:  # Low Flesch = advanced
                    adv += max(0, 70 - fk_ease)
                if gunning > 12:  # High Gunning = advanced
                    adv += min(70, gunning * 4)
                if smog > 12:  # High SMOG = advanced
                    adv += min(70, smog * 4)
                adv = min(100, max(10, adv))  # Min 10, max 100

                try:
                    import altair as alt
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

                    df = pd.DataFrame({
                        "Level": ["Beginner", "Intermediate", "Advanced"],
                        "Score": [beg, inter, adv],
                        "Color": ["#10b981", "#f59e0b", "#ef4444"],
                        # Add order column for sorting
                        "Order": [1, 2, 3]
                    })
                    
                    # Chart with explicit ordering
                    chart = alt.Chart(df).mark_bar().encode(
                        x=alt.X("Level:N", sort=["Beginner", "Intermediate", "Advanced"]), 
                        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 100])),
                        color=alt.Color("Color:N", scale=None, legend=None),
                        tooltip=["Level:N", "Score:Q"]
                    ).properties(height=220)
                    
                    st.altair_chart(chart, use_container_width=True)
                    st.caption("Higher scores indicate greater prevalence of that reading level")
                    
                except Exception as e:
                    st.error(f"Could not generate chart: {str(e)}")

                # Simplified feature info
                st.subheader("Document Analysis Features")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("⚡ **Real-time readability scoring**")
                    st.markdown("📊 **Visual complexity indicators**")
                with col2:
                    st.markdown("📝 **Multiple file format support**")
                    st.markdown("🔍 **Comprehensive text metrics**")


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