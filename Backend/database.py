import os
from pathlib import Path
from urllib.parse import quote_plus

from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError, ProgrammingError

# Load environment variables from .env located next to this file
_env_path = Path(__file__).with_name('.env')
load_dotenv(dotenv_path=_env_path)

# Read from env with safe defaults (useful for local dev without .env)
MYSQL_USER = os.getenv("MYSQL_USER")
MYSQL_PASSWORD = os.getenv("MYSQL_PASSWORD")
MYSQL_HOST = os.getenv("MYSQL_HOST")
MYSQL_DB = os.getenv("MYSQL_DB")

# URL-encode password to support special characters
_pwd = quote_plus(MYSQL_PASSWORD or "")
DATABASE_URL = f"mysql+mysqlconnector://{MYSQL_USER}:{_pwd}@{MYSQL_HOST}:3306/{MYSQL_DB}"

# Try to create an engine bound to the target database. If the database doesn't exist,
# create it using a temporary connection to the server (no database) and retry.
try:
	engine = create_engine(DATABASE_URL, pool_pre_ping=True)
	# test connection
	with engine.connect() as conn:
		pass
except (OperationalError, ProgrammingError) as e:
	# If the database is missing, create it and retry
	tmp_url = f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:3306/"
	tmp_engine = create_engine(tmp_url)
	with tmp_engine.connect() as conn:
		# CREATE DATABASE IF NOT EXISTS
		conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {MYSQL_DB}"))
	engine = create_engine(DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
