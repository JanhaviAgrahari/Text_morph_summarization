# Text Summary + Auth (FastAPI + Streamlit)

A minimal FastAPI backend with MySQL for user authentication and a Streamlit frontend for login, registration, and profile updates.

## Features
- Register and login with email/password
- Passwords hashed with bcrypt (passlib)
- Update simple profile fields (name, age group, language)
- MySQL database auto-created on first run (if server is reachable)
- Streamlit UI with clear error messages and backend health indicator

## Project layout
```
Backend/
  main.py        # FastAPI app and routes
  models.py      # SQLAlchemy models
  schemas.py     # Pydantic schemas
  crud.py        # DB operations and password hashing
  database.py    # SQLAlchemy engine/session and DB bootstrap
Frontend/
  app.py         # Streamlit UI calling the FastAPI endpoints
requirements.txt
```

## Prerequisites
- Windows PowerShell (default shell)
- Python 3.10 (matches the provided venv)
- MySQL server running locally on port 3306
- Database credentials are read from `Backend/.env` (loaded via `python-dotenv`). Example:


## Quick start (using provided venv)
Open PowerShell in the project root `text_summary1` and run:

```powershell
# 1) Start the FastAPI backend
./chatbot_env/Scripts/uvicorn.exe Backend.main:app --host 127.0.0.1 --port 8000 --reload
```

In a new PowerShell window, run the Streamlit frontend:

```powershell
# 2) Start the Streamlit frontend
./chatbot_env/Scripts/streamlit.exe run Frontend/app.py
```

Now open http://127.0.0.1:8501 and try Register, then Login. The sidebar should show Backend: online.

If you want to use your own environment instead of the bundled venv:

```powershell
# Create and activate a new venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Start backend
uvicorn Backend.main:app --host 127.0.0.1 --port 8000 --reload

# In another terminal, start frontend
streamlit run Frontend/app.py
```

## Configuration
- The backend loads DB credentials from `Backend/.env` using `python-dotenv`.
- The SQLAlchemy URL is constructed with password safely URL-encoded.
- CORS is enabled broadly for local development.
- To enable email-based Forgot Password, add SMTP settings to `Backend/.env`:

  ```env
  SMTP_HOST=smtp.gmail.com
  SMTP_PORT=587
  SMTP_USER=your_email@gmail.com
  SMTP_PASSWORD=your_app_password
  EMAIL_FROM=Your App <your_email@gmail.com>
  FRONTEND_RESET_URL=http://localhost:8501
  APP_NAME=TextSummary
  ```
  For Gmail, use an App Password (2FA required) and keep these secrets private.
- CORS is enabled broadly for local development.

## API (summary)
- `GET /ping` → `{ "message": "pong" }`
- `POST /register` → returns created user (id, email, optional profile fields)
- `POST /login` → `{ "message": "Login successful", "user_id": <id> }`
- `POST /profile/{user_id}` → returns updated user

Open Swagger docs: http://127.0.0.1:8000/docs

## Troubleshooting
- Backend unreachable in Streamlit:
  - Make sure uvicorn is running and `GET http://127.0.0.1:8000/ping` responds.
  - Verify MySQL is running and credentials in `Backend/.env` are correct.
- Register returns 400:
  - Email already exists in DB.
- Login returns 401:
  - Invalid email or password.

## Security note
This example is for learning/demo. For production, add proper auth (JWT sessions/tokens), HTTPS, secure secrets management, rate limiting, and DB migrations.
