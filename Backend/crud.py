from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext
from datetime import datetime, timedelta
import secrets

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_user(db: Session, user: schemas.UserCreate):
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not verify_password(password, user.password):
        return None
    return user

def update_profile(db: Session, user_id: int, profile: schemas.UserProfile):
    user = db.query(models.User).filter(models.User.id == user_id).first()
    if user:
        user.name = profile.name
        user.age_group = profile.age_group
        user.language = profile.language
        db.commit()
        db.refresh(user)
    return user


# --- Password reset helpers ---
def create_password_reset(db: Session, email: str, minutes_valid: int = 15) -> tuple[str, models.PasswordReset | None]:
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        return "", None
    token = secrets.token_urlsafe(32)
    expires = datetime.utcnow() + timedelta(minutes=minutes_valid)
    pr = models.PasswordReset(user_id=user.id, token=token, expires_at=expires, used=False)
    db.add(pr)
    db.commit()
    db.refresh(pr)
    return token, pr


def reset_password(db: Session, token: str, new_password: str) -> bool:
    pr = db.query(models.PasswordReset).filter(models.PasswordReset.token == token).first()
    if not pr or pr.used:
        return False
    if pr.expires_at < datetime.utcnow():
        return False
    user = db.query(models.User).filter(models.User.id == pr.user_id).first()
    if not user:
        return False
    user.password = get_password_hash(new_password)
    pr.used = True
    db.commit()
    return True
