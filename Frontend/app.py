import streamlit as st
import requests
import re

API_URL = "http://127.0.0.1:8000"  # FastAPI backend


def api_post(path: str, payload: dict, timeout: int = 5):
    """POST to backend with robust error handling.

    Returns tuple (ok: bool, data: dict|str)
    - ok True => data is JSON body
    - ok False => data is error string for user display
    """
    url = f"{API_URL}{path}"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        # Try to parse JSON either way
        body = None
        try:
            body = resp.json()
        except Exception:
            body = None

        if 200 <= resp.status_code < 300:
            return True, (body if body is not None else {})

        # Surface backend error details when available
        if body and isinstance(body, dict):
            detail = body.get("detail") or body.get("message")
            if isinstance(detail, list):
                # FastAPI validation errors can be a list
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

st.set_page_config(page_title="User Login & Profile", layout="centered")

# Show a fixed toast when registration just succeeded (doesn't affect layout)
if "registration_success" in st.session_state and st.session_state.get("registration_success", False):
        toast_html = '''
        <style>
        .st-toast-msg{position:fixed;top:80px;left:50%;transform:translateX(-50%);z-index:99999;
            background:#2ecc71;color:#fff;padding:12px 20px;border-radius:8px;
            box-shadow:0 6px 18px rgba(0,0,0,0.2);font-weight:600;}
        .st-toast-msg{animation:st-fade 3.5s forwards}
        @keyframes st-fade{0%{opacity:1;transform:translateX(-50%) translateY(0)}80%{opacity:1}100%{opacity:0;transform:translateX(-50%) translateY(-8px)}}
        </style>
        <div class="st-toast-msg">You have registered successfully.</div>
        '''
        st.markdown(toast_html, unsafe_allow_html=True)
        # clear so it only shows once
        st.session_state.registration_success = False

col1, col2 = st.columns(2)

if "show_register" not in st.session_state:
    st.session_state.show_register = False

if "registration_success" not in st.session_state:
    st.session_state.registration_success = False

def switch_to_register():
    st.session_state.show_register = True

def switch_to_login():
    st.session_state.show_register = False

with col1:
    st.markdown("### 👤 User Authentication")

    if not st.session_state.show_register:
        if st.session_state.registration_success:
            st.success("You have registered successfully.")
            st.session_state.registration_success = False

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="user@example.com")
            password = st.text_input("Password", type="password")
            sign_in = st.form_submit_button("Sign In")
            create_account_clicked = st.form_submit_button("Create Account")

        if sign_in:
            # Client-side checks
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
                st.session_state["user_id"] = data.get("user_id")
            else:
                st.error(data)

        if create_account_clicked:
            switch_to_register()
    else:
        with st.form("register_form"):
            reg_email = st.text_input("Email", placeholder="user@example.com")
            reg_password = st.text_input("Password", type="password")
            register_clicked = st.form_submit_button("Register")

        if register_clicked:
            # Client-side validation for email and password complexity
            def valid_password(p: str) -> tuple[bool, str | None]:
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

            if not reg_email:
                st.error("Please enter your email.")
            elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", reg_email):
                st.error("Please enter a valid email address.")
            elif not reg_password:
                st.error("Please enter a password.")
            else:
                ok_pw, err = valid_password(reg_password)
                if not ok_pw:
                    st.error(err)
                else:
                    ok, data = api_post("/register", {"email": reg_email, "password": reg_password})
                    if ok:
                        st.session_state.registration_success = True
                        switch_to_login()
                    else:
                        st.error(data)

        if st.button("Sign in"):
            switch_to_login()

with col2:
    st.markdown("### 👥 Profile Management")
    if "user_id" not in st.session_state:
        st.info("Please log in to update your profile.")
    else:
        with st.form("profile_form"):
            name = st.text_input("Name", placeholder="Your name")
            age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "50+"])
            language = st.radio("Language Preference", ["English", "हिंदी"], horizontal=True)
            save_profile = st.form_submit_button("Save Profile")

        if save_profile:
            ok, data = api_post(
                f"/profile/{st.session_state['user_id']}",
                {"name": name, "age_group": age_group, "language": language},
            )
            if ok:
                st.success("Profile updated successfully")
            else:
                st.error(data)

# Lightweight backend health indicator
try:
    _ping = requests.get(f"{API_URL}/ping", timeout=2)
    if _ping.ok:
        st.sidebar.success("Backend: online")
    else:
        st.sidebar.warning("Backend: unreachable")
except Exception:
    st.sidebar.warning("Backend: unreachable")
