from pydantic import BaseModel, EmailStr, field_validator
import re
from datetime import datetime

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
    role: str = "user"

    class Config:
        from_attributes = True


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str

    @field_validator("new_password")
    @classmethod
    def validate_new_password(cls, v: str) -> str:
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


class MessageOut(BaseModel):
    message: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# --- ROUGE evaluation schemas ---
class RougeEvalRequest(BaseModel):
    reference: str  # ground-truth / human written summary
    candidate: str  # model generated summary
    use_stemmer: bool = True
    metrics: list[str] | None = None  # e.g. ["rouge1","rouge2","rougeL"]

class RougeScore(BaseModel):
    precision: float
    recall: float
    f1: float

class RougeEvalResponse(BaseModel):
    scores: dict[str, RougeScore]
    reference_tokens: int
    candidate_tokens: int


# --- Paraphrasing schemas ---
class ParaphraseRequest(BaseModel):
    model_name: str
    text: str
    creativity: float = 0.3
    length: str = "medium"  # short | medium | long
    # Optional user email; if provided, backend may store history
    user_email: str | None = None
    # ROUGE evaluation options
    evaluate_rouge: bool = False
    rouge_metrics: list[str] | None = None  # e.g. ["rouge1","rouge2","rougeL"]
    use_stemmer: bool = True
    # Optional reference (gold standard) paraphrase to evaluate against
    reference_paraphrase: str | None = None
    
# --- Visualization schemas ---
class ComplexityVisualizationRequest(BaseModel):
    original_text: str
    comparison_texts: list[str]


# --- History schemas ---
class HistoryEntry(BaseModel):
    id: int
    type: str  # 'summary' or 'paraphrase'
    original_text: str
    result_text: str
    model: str
    created_at: datetime
    parameters: str | None = None
    feedback_rating: str | None = None  # 'thumbs_up', 'thumbs_down', or None
    feedback_comment: str | None = None
    feedback_at: datetime | None = None

    class Config:
        from_attributes = True

class HistoryResponse(BaseModel):
    entries: list[HistoryEntry]
    
class CreateHistoryEntry(BaseModel):
    user_id: int
    type: str  # 'summary' or 'paraphrase' 
    original_text: str
    result_text: str
    model: str
    parameters: str | None = None

# Create-by-email payload for external callers (frontend)
class CreateHistoryByEmail(BaseModel):
    email: str
    type: str  # 'summary' or 'paraphrase'
    original_text: str
    result_text: str
    model: str
    parameters: str | None = None


# Admin-only schema with user details
class AdminHistoryEntry(BaseModel):
    id: int
    user_id: int
    user_email: str
    user_name: str | None
    type: str  # 'summary' or 'paraphrase'
    original_text: str
    result_text: str
    model: str
    created_at: datetime
    parameters: str | None = None
    feedback_rating: str | None = None  # 'thumbs_up', 'thumbs_down', or None
    feedback_comment: str | None = None
    feedback_at: datetime | None = None

    class Config:
        from_attributes = True


class AdminHistoryResponse(BaseModel):
    entries: list[AdminHistoryEntry]
    total_count: int


# --- Admin edit schema ---
class UpdateHistoryEntry(BaseModel):
    result_text: str  # The edited summary/paraphrase


# --- Admin statistics schemas ---
class UserActivityStats(BaseModel):
    user_email: str
    user_name: str | None
    total_summaries: int
    total_paraphrases: int
    last_activity: datetime | None


class AdminStatistics(BaseModel):
    total_users: int
    total_summaries: int
    total_paraphrases: int
    total_documents: int
    active_users_count: int  # Users with at least one activity
    recent_activity: list[UserActivityStats]
    # Feedback statistics
    feedback_thumbs_up: int
    feedback_thumbs_down: int
    feedback_total: int
    feedback_rate: float  # Percentage of documents with feedback


# --- Feedback schemas ---
class SubmitFeedback(BaseModel):
    rating: str  # 'thumbs_up' or 'thumbs_down'
    comment: str | None = None
