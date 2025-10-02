from sqlalchemy.orm import Session
from . import models, schemas
from passlib.context import CryptContext
from datetime import datetime, timedelta
import secrets

# Configure passlib for password hashing/verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str):
    # Generate salted hash for persistent storage
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    # Check user-provided password against stored hash
    return pwd_context.verify(plain_password, hashed_password)

def create_user(db: Session, user: schemas.UserCreate, role: str = "user"):
    # Create a new user with a hashed password and role
    hashed_password = get_password_hash(user.password)
    db_user = models.User(email=user.email, password=hashed_password, role=role)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return db_user

def authenticate_user(db: Session, email: str, password: str):
    # Look up by email and verify password
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user or not verify_password(password, user.password):
        return None
    return user

def update_profile(db: Session, user_id: int, profile: schemas.UserProfile):
    # Update user profile fields and persist
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
    # Create one-time token for a user if email exists; return blank token if not
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
    # Validate token, ensure not expired/used; set new hashed password and mark token used
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


# --- History operations ---
def create_history_entry(db: Session, entry: schemas.CreateHistoryEntry):
    """Create a new history entry for a summary or paraphrase."""
    db_entry = models.History(
        user_id=entry.user_id,
        type=entry.type,
        original_text=entry.original_text,
        result_text=entry.result_text,
        model=entry.model,
        parameters=entry.parameters
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

def get_user_history(db: Session, user_id: int, limit: int = 100):
    """Get the history entries for a specific user, newest first."""
    return db.query(models.History)\
        .filter(models.History.user_id == user_id)\
        .order_by(models.History.created_at.desc())\
        .limit(limit)\
        .all()

def get_history_entry(db: Session, entry_id: int):
    """Get a specific history entry by ID."""
    return db.query(models.History).filter(models.History.id == entry_id).first()

def get_user_history_by_type(db: Session, user_id: int, type: str, limit: int = 100):
    """Get history entries for a user filtered by type (summary/paraphrase)."""
    return db.query(models.History)\
        .filter(models.History.user_id == user_id, models.History.type == type)\
        .order_by(models.History.created_at.desc())\
        .limit(limit)\
        .all()

def delete_history_entry(db: Session, entry_id: int, user_id: int) -> bool:
    """Delete a history entry, ensuring it belongs to the correct user."""
    db_entry = db.query(models.History)\
        .filter(models.History.id == entry_id, models.History.user_id == user_id)\
        .first()
    if not db_entry:
        return False
    db.delete(db_entry)
    db.commit()
    return True


def get_all_history(db: Session, limit: int = 1000, type_filter: str = None):
    """Get all history entries across all users (admin/mentor only)."""
    query = db.query(models.History).join(models.User)
    if type_filter:
        query = query.filter(models.History.type == type_filter)
    return query.order_by(models.History.created_at.desc()).limit(limit).all()
