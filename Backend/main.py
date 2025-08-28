from fastapi import FastAPI, Depends, HTTPException
from fastapi import BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, SessionLocal, Base
from .emailer import send_password_reset_email
from .auth import create_access_token, get_current_user
import os
import logging

Base.metadata.create_all(bind=engine)

# Root FastAPI app with basic auth and profile routes
app = FastAPI(title="User Authentication API")

# Allow local tools and browsers to call the API (useful if switching to browser-side requests)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/ping")
def ping():
    # Lightweight health check endpoint used by Streamlit sidebar
    return {"message": "pong"}

def get_db():
    # Dependency that yields a DB session and ensures cleanup
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.post("/register", response_model=schemas.UserOut)
def register(user: schemas.UserCreate, db: Session = Depends(get_db)):
    # Prevent duplicate registrations by email
    db_user = db.query(models.User).filter(models.User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    created = crud.create_user(db, user)
    return created

@app.post("/login", response_model=schemas.Token)
def login(user: schemas.UserLogin, db: Session = Depends(get_db)):
    # Verify credentials and issue a short-lived JWT
    db_user = crud.authenticate_user(db, user.email, user.password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(db_user.id)
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", response_model=schemas.UserOut)
def read_me(current_user: models.User = Depends(get_current_user)):
    # Returns current user based on Bearer token
    return current_user


@app.post("/profile", response_model=schemas.UserOut)
def update_profile(profile: schemas.UserProfile, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    # Update basic profile fields for the authenticated user
    db_user = crud.update_profile(db, current_user.id, profile)
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    return db_user


@app.post("/forgot-password", response_model=schemas.MessageOut)
def forgot_password(payload: schemas.ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Always return a generic message to avoid leaking which emails exist
    token, pr = crud.create_password_reset(db, payload.email)
    if pr and token:
        # Send email in the background to avoid blocking the HTTP response
        background_tasks.add_task(send_password_reset_email, payload.email, token)

    # Optional developer-friendly hint controlled by env
    debug = (os.getenv("EMAIL_DEBUG", "false").lower() in {"1", "true", "yes", "on"})
    if debug and token:
        logging.getLogger("uvicorn.error").info(
            f"[EMAIL_DEBUG] Password reset token for {payload.email}: {token}"
        )

    # Do not reveal if email exists or include any token in the response
    return {"message": "If the email exists, a reset link has been sent."}


@app.post("/reset-password", response_model=schemas.MessageOut)
def reset_password(payload: schemas.ResetPasswordRequest, db: Session = Depends(get_db)):
    # Validate token, ensure it's not expired/used, and set the new password
    ok = crud.reset_password(db, payload.token, payload.new_password)
    if not ok:
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")
    return {"message": "Password has been reset successfully"}
