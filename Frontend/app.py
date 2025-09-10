import streamlit as st
import requests
import re
import os
import csv
from datetime import datetime
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
BACKEND_URL = "http://127.0.0.1:8000"
API_URL = BACKEND_URL  # Alias for legacy references
ROUGE_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "rouge_score"))
os.makedirs(ROUGE_OUTPUT_DIR, exist_ok=True)

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


# Create the main app tabs
tabs = st.tabs(["Sign in", "Register", "Forgot password", "Profile", "Readability", "Summarization", "Paraphrasing", "History"])

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

# -------------------- History Tab --------------------

with tabs[7]:  # Tab 7 - History
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
                    # Parse timestamp flexibly (with or without microseconds / timezone)
                    created_raw = entry.get("created_at") or ""
                    try:
                        created_date = datetime.strptime(created_raw, "%Y-%m-%dT%H:%M:%S.%f")
                    except Exception:
                        try:
                            created_date = datetime.strptime(created_raw, "%Y-%m-%d %H:%M:%S")
                        except Exception:
                            created_date = datetime.utcnow()
                    formatted_date = created_date.strftime("%B %d, %Y at %H:%M %p")
                    
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