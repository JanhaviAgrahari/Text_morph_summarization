from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from .database import Base

# ORM model representing application users
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # Unique login identifier; indexed for fast lookups
    email = Column(String(255), unique=True, index=True, nullable=False)
    # Hashed password (never store plaintext)
    password = Column(String(255), nullable=False)
    # Optional profile attributes saved via /profile endpoint
    name = Column(String(255), nullable=True)
    age_group = Column(String(50), nullable=True)
    language = Column(String(50), nullable=True)


# Tracks password reset tokens and their lifecycle
class PasswordReset(Base):
    __tablename__ = "password_resets"

    id = Column(Integer, primary_key=True, index=True)
    # FK to the user who requested the reset
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    # Random, URL-safe token sent via email; must be unique
    token = Column(String(255), unique=True, index=True, nullable=False)
    # Expiration timestamp (UTC)
    expires_at = Column(DateTime, nullable=False)
    # Mark token as used after successful password change (one-time use)
    used = Column(Boolean, default=False, nullable=False)
