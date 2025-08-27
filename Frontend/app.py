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
        return False, "Cannot connect to backend. Make sure it's running at http://127.0.0.1:8000"
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

# Minimal, professional header
st.markdown("# Text morph advanced summarisation using AI")

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
    with st.form("login_form"):
        email = st.text_input("Email", placeholder="user@example.com")
        password = st.text_input("Password", type="password")
        sign_in = st.form_submit_button("Sign in")
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
            else:
                st.error(data)


# Register tab
with tabs[1]:
    with st.form("register_form"):
        reg_email = st.text_input("Email", placeholder="user@example.com")
        reg_password = st.text_input("Password", type="password")
        register_clicked = st.form_submit_button("Create account")
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
    st.caption("Request a reset link or use your token to set a new password.")
    with st.form("forgot_form"):
        fp_email = st.text_input("Email", placeholder="user@example.com")
        get_link = st.form_submit_button("Get reset link")
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
    with st.form("reset_form"):
        token = st.text_input("Token", value=token_prefill)
        new_pw = st.text_input("New password", type="password")
        reset_clicked = st.form_submit_button("Reset password")
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
        st.info("Please sign in to manage your profile.")
    else:
        ok_me, me = api_get("/me", token=token)
        if ok_me and isinstance(me, dict):
            st.caption(f"Signed in as {me.get('email')}")
        with st.form("profile_form"):
            name = st.text_input("Name", placeholder="Your name")
            age_group = st.selectbox("Age group", ["18-25", "26-35", "36-50", "50+"])
            language = st.radio("Language preference", ["English", "हिंदी"], horizontal=True)
            save_profile = st.form_submit_button("Save profile")
        if save_profile:
            ok, data = api_post(
                "/profile",
                {"name": name, "age_group": age_group, "language": language},
                token=token,
            )
            if ok:
                st.success("Profile updated.")
            else:
                st.error(data)


# Readability tab (visible for signed-in users)
with tabs[4]:
    token = st.session_state.get("token")
    if not token:
        st.info("Please sign in to access readability analysis.")
    else:
        # Card-like container styles with colorful elements matching example
        st.markdown(
            """
            <style>
            .card {background:#0f172a14;border:1px solid #33415533;border-radius:12px;padding:16px;margin-bottom:16px}
            .upload-card{border:2px dashed #3b82f6;background:#1e293b0d}
            
            /* Metric cards with color styles */
            .metric-card{background:white;border:1px solid #e5e7eb;border-radius:8px;padding:16px;text-align:center;box-shadow:0 2px 5px rgba(0,0,0,0.1)}
            .metric-value{font-size:32px;font-weight:700;margin-bottom:4px}
            .metric-label{font-size:14px;color:#4b5563}
            
            /* Color coding for metrics */
            .metric-green .metric-value{color:#22c55e}
            .metric-yellow .metric-value{color:#eab308}
            .metric-red .metric-value{color:#ef4444}
            
            /* Color indicators for metrics */
            .indicator{height:6px;border-radius:3px;margin-top:8px;background:linear-gradient(to right, #22c55e, #eab308, #ef4444)}
            
            /* Feature card */
            .feature-card{background:#e0f2fe;border-left:4px solid #3b82f6;border-radius:6px;padding:16px;margin-top:24px}
            .feature-title{font-weight:600;color:#1e40af;margin-bottom:12px}
            .feature-item{display:flex;align-items:center;margin-bottom:8px}
            .feature-icon{color:#3b82f6;margin-right:8px;font-size:16px}
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Dashboard & Readability Analysis")

        # Upload
        with st.container():
            st.markdown('<div class="card upload-card">', unsafe_allow_html=True)
            up = st.file_uploader("Upload Document", type=["txt", "pdf", "docx"], label_visibility="visible")
            st.markdown('</div>', unsafe_allow_html=True)

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

                # Colorful metric cards with indicators like the example
                c1, c2, c3 = st.columns(3)
                with c1:
                    # Green for high Flesch (easier readability)
                    st.markdown(
                        f'''<div class="metric-card metric-green">
                            <div class="metric-value">{fk_ease}</div>
                            <div class="metric-label">Flesch-Kincaid</div>
                            <div class="indicator"></div>
                        </div>''', 
                        unsafe_allow_html=True
                    )
                with c2:
                    # Yellow for mid-range Gunning
                    st.markdown(
                        f'''<div class="metric-card metric-yellow">
                            <div class="metric-value">{gunning}</div>
                            <div class="metric-label">Gunning Fog</div>
                            <div class="indicator"></div>
                        </div>''', 
                        unsafe_allow_html=True
                    )
                with c3:
                    # Red for SMOG (often indicates complexity)
                    st.markdown(
                        f'''<div class="metric-card metric-red">
                            <div class="metric-value">{smog}</div>
                            <div class="metric-label">SMOG Index</div>
                            <div class="indicator"></div>
                        </div>''', 
                        unsafe_allow_html=True
                    )

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

                    df = pd.DataFrame({
                        "Level": ["Beginner", "Intermediate", "Advanced"],
                        "Score": [beg, inter, adv],
                        # Colors matching the example image: green, yellow, red
                        "Color": ["#4ade80", "#facc15", "#f87171"]
                    })
                    chart_title = "Reading Level Distribution"
                    chart = alt.Chart(df).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6).encode(
                        x=alt.X("Level", sort=None, title=None),
                        y=alt.Y("Score", scale=alt.Scale(domain=[0, 100]), title="Score"),
                        color=alt.Color("Color", scale=None, legend=None)
                    ).configure_view(
                        strokeWidth=0
                    ).configure_axis(
                        grid=True,
                        gridColor="#e5e7eb",
                        labelColor="#4b5563",
                        tickColor="#e5e7eb",
                        domainColor="#e5e7eb"
                    ).properties(height=220, title=chart_title)
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass

                # Analysis features list with icons and styling
                st.markdown('''
                <div class="feature-card">
                    <div class="feature-title">Analysis Features</div>
                    <div class="feature-item">
                        <span class="feature-icon">⚡</span>
                        Real-time readability scoring
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📊</span>
                        Visual complexity indicators
                    </div>
                    <div class="feature-item">
                        <span class="feature-icon">📝</span>
                        Comprehensive text metrics
                    </div>
                </div>
                ''', unsafe_allow_html=True)


# Lightweight backend health indicator
try:
    _ping = requests.get(f"{API_URL}/ping", timeout=2)
    if _ping.ok:
        st.sidebar.success("Backend: online")
    else:
        st.sidebar.warning("Backend: unreachable")
except Exception:
    st.sidebar.warning("Backend: unreachable")

