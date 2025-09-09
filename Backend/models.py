from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, Enum
from sqlalchemy.orm import relationship
from .database import Base
from datetime import datetime

# ORM model representing application users
class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    # Unique login identifier; indexed for fast lookups
    email = Column(String, unique=True, index=True, nullable=False)
    # Hashed password (never store plaintext)
    password = Column(String, nullable=False)
    # Optional profile attributes saved via /profile endpoint
    name = Column(String, nullable=True)
    age_group = Column(String, nullable=True)
    language = Column(String, nullable=True)
    
    # Relationship to history entries
    history_entries = relationship("History", back_populates="user")


# Tracks password reset tokens and their lifecycle
class PasswordReset(Base):
    __tablename__ = "password_resets"

    id = Column(Integer, primary_key=True, index=True)
    # FK to the user who requested the reset
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    # Random, URL-safe token sent via email; must be unique
    token = Column(String, unique=True, index=True, nullable=False)
    # Expiration timestamp (UTC)
    expires_at = Column(DateTime, nullable=False)
    # Mark token as used after successful password change (one-time use)
    used = Column(Boolean, default=False, nullable=False)


# Stores all text transformations (summaries and paraphrases)
class History(Base):
    __tablename__ = "history"

    id = Column(Integer, primary_key=True, index=True)
    # FK to the user who created this entry
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    # Type of transformation (summary or paraphrase)
    type = Column(String, nullable=False, index=True)
    # Original text (before transformation)
    original_text = Column(Text, nullable=False)
    # Transformed text (after summarization/paraphrasing)
    result_text = Column(Text, nullable=False)
    # Model used for transformation
    model = Column(String, nullable=False)
    # Parameters used (JSON string with model-specific settings)
    parameters = Column(String, nullable=True)
    # Creation timestamp
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Relationship to the user
    user = relationship("User", back_populates="history_entries")
