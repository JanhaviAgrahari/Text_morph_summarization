from pydantic import BaseModel, EmailStr, field_validator
import re

class UserCreate(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        # Minimum 8 chars, at least one uppercase, one lowercase, one digit, one special char
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long.")
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must include at least one uppercase letter.")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must include at least one lowercase letter.")
        if not re.search(r"\d", v):
            raise ValueError("Password must include at least one number.")
        if not re.search(r"[^A-Za-z0-9]", v):
            raise ValueError("Password must include at least one special character.")
        return v

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class UserProfile(BaseModel):
    name: str
    age_group: str
    language: str

class UserOut(BaseModel):
    id: int
    email: str
    name: str | None
    age_group: str | None
    language: str | None

    class Config:
        from_attributes = True
