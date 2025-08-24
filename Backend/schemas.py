from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    email: EmailStr
    password: str

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
