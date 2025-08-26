import streamlit as st
import requests
import re

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
if "show_forgot" not in st.session_state:
    st.session_state.show_forgot = False
if "reset_token_prefill" not in st.session_state:
    st.session_state["reset_token_prefill"] = ""
if "reset_clear_pending" not in st.session_state:
    st.session_state["reset_clear_pending"] = False
if "reset_success" not in st.session_state:
    st.session_state["reset_success"] = False
if "reset_form_version" not in st.session_state:
    st.session_state["reset_form_version"] = 0

# If a token is present in the page URL, auto-open Forgot Password and prefill
try:
    qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
    if qp and "token" in qp:
        token_val = qp["token"] if isinstance(qp["token"], str) else (qp["token"][0] if qp["token"] else "")
        if token_val:
            st.session_state.show_forgot = True
            st.session_state["reset_token_prefill"] = token_val
except Exception:
    pass

# If a clear is pending from last submission, bump version and clear prefill BEFORE rendering widgets
if st.session_state.get("reset_clear_pending"):
    st.session_state["reset_token_prefill"] = ""
    st.session_state["reset_form_version"] += 1
    st.session_state["reset_clear_pending"] = False

def switch_to_register():
    st.session_state.show_register = True

def switch_to_login():
    st.session_state.show_register = False

with col1:
    st.markdown("### 👤 User Authentication")

    if st.session_state.get("show_forgot", False):
        # Forgot password flow first if toggled
        st.markdown("#### 🔐 Forgot Password")
        if st.session_state.get("reset_success"):
            st.success("Password has been reset successfully")
            st.session_state["reset_success"] = False
        with st.form("forgot_form"):
            fp_email = st.text_input("Email", placeholder="user@example.com")
            get_link = st.form_submit_button("Get Reset Token")
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
        st.markdown("##### Reset Password (use token from message)")
        with st.form("reset_form"):
            token = st.text_input(
                "Token",
                value=st.session_state.get("reset_token_prefill", ""),
                key=f"reset_token_input_v{st.session_state['reset_form_version']}",
            )
            new_pw = st.text_input(
                "New Password",
                type="password",
                key=f"reset_new_password_input_v{st.session_state['reset_form_version']}",
            )
            reset_clicked = st.form_submit_button("Reset Password")
        if reset_clicked:
            # Reuse client policy
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

            if not token:
                st.error("Please enter the reset token.")
            else:
                ok_pw, err = valid_password(new_pw)
                if not ok_pw:
                    st.error(err)
                else:
                    ok, data = api_post("/reset-password", {"token": token, "new_password": new_pw})
                    if ok:
                        # Show success after rerun and clear inputs safely on next cycle
                        st.session_state["reset_success"] = True
                        st.session_state["reset_clear_pending"] = True
                        # Also clear token from URL so we don't auto-open forgot view
                        try:
                            if hasattr(st, "query_params"):
                                st.query_params.clear()
                            else:
                                st.experimental_set_query_params()
                        except Exception:
                            pass
                        st.rerun()
                    else:
                        st.error(data)
        # Provide a back-to-login button
        if st.button("Back to Sign in"):
            st.session_state.show_forgot = False
            st.session_state.show_register = False
            # Clear any token in the URL so we don't re-open this view
            try:
                if hasattr(st, "query_params"):
                    st.query_params.clear()
                else:
                    st.experimental_set_query_params()
            except Exception:
                pass
            st.rerun()

    elif not st.session_state.show_register:
        if st.session_state.registration_success:
            st.success("You have registered successfully.")
            st.session_state.registration_success = False

        with st.form("login_form"):
            email = st.text_input("Email", placeholder="user@example.com")
            password = st.text_input("Password", type="password")
            cols = st.columns([1,1,1])
            with cols[0]:
                sign_in = st.form_submit_button("Sign In")
            with cols[1]:
                create_account_clicked = st.form_submit_button("Create Account")
            with cols[2]:
                forgot_clicked = st.form_submit_button("Forgot Password?")

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
                    st.session_state["token"] = data.get("access_token")
                else:
                    st.error(data)

        if create_account_clicked:
            switch_to_register()
        if 'forgot_clicked' in locals() and forgot_clicked:
            st.session_state.show_register = False
            st.session_state["show_forgot"] = True
            st.rerun()
    elif st.session_state.get("show_register", False):
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
                        st.rerun()
                    else:
                        st.error(data)

        if st.button("Sign in"):
            switch_to_login()
            st.session_state["show_forgot"] = False

with col2:
    st.markdown("### 👥 Profile Management")
    if "token" not in st.session_state or not st.session_state.get("token"):
        st.info("Please log in to update your profile.")
    else:
        ok_me, me = api_get("/me", token=st.session_state.get("token"))
        if ok_me and isinstance(me, dict):
            st.caption(f"Signed in as {me.get('email')}")
        with st.form("profile_form"):
            name = st.text_input("Name", placeholder="Your name")
            age_group = st.selectbox("Age Group", ["18-25", "26-35", "36-50", "50+"])
            language = st.radio("Language Preference", ["English", "हिंदी"], horizontal=True)
            save_profile = st.form_submit_button("Save Profile")

        if save_profile:
            ok, data = api_post(
                "/profile",
                {"name": name, "age_group": age_group, "language": language},
                token=st.session_state.get("token"),
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
