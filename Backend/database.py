import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Define database directory and ensure it exists
DB_DIR = Path(__file__).parent.parent / "database"
DB_DIR.mkdir(exist_ok=True)

# SQLite database file path
SQLITE_DB_FILE = DB_DIR / "text_morph.db"

# Create the SQLite connection string
DATABASE_URL = f"sqlite:///{SQLITE_DB_FILE}"

# Create the SQLAlchemy engine
# Note: check_same_thread=False allows SQLite to be used with multiple threads
# which is necessary for FastAPI's concurrent request handling
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False},
    pool_pre_ping=True
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declarative base used by SQLAlchemy models
Base = declarative_base()

# Database dependency for route handlers
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()