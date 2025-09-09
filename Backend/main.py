from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, SessionLocal, Base
from .emailer import send_password_reset_email
from .auth import create_access_token, get_current_user
import os
import logging

# Local summarization and paraphrasing helpers
from .summarizer import summarize_text, MODELS
from .pdf_utils import extract_text_from_pdf
from .parahrase import paraphrase_text, analyze_text_complexity
from . import schemas as _schemas
from . import schemas as schemas

try:
    from rouge_score import rouge_scorer
except Exception:  # pragma: no cover - optional dependency
    rouge_scorer = None

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


UPLOAD_DIR = "Backend/uploaded_pdfs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/summarize")
@app.post("/summarize/")
async def summarize_endpoint(
    model_choice: str = Form("pegasus"),
    summary_length: str = Form("medium"),
    input_type: str = Form("text"),  # "text" or "pdf"
    text_input: str = Form(""),
    pdf_file: UploadFile = File(None)
):
    """Summarize plain text or an uploaded PDF using a selected local model.

    model_choice: one of MODELS keys (pegasus, bart, flan-t5)
    summary_length: short | medium | long (mapped internally in summarizer)
    input_type: 'text' or 'pdf'
    """
    try:
        model_choice = (model_choice or "pegasus").lower()
        if model_choice not in MODELS:
            raise HTTPException(status_code=422, detail=f"Invalid model_choice. Choose one of: {', '.join(MODELS.keys())}")

        summary_length = (summary_length or "medium").lower()
        if summary_length not in {"short", "medium", "long"}:
            raise HTTPException(status_code=422, detail="summary_length must be short, medium, or long")

        input_type = (input_type or "text").lower()
        if input_type not in {"text", "pdf"}:
            raise HTTPException(status_code=422, detail="input_type must be 'text' or 'pdf'")

        if input_type == "pdf":
            if not pdf_file:
                raise HTTPException(status_code=400, detail="No PDF uploaded")
            pdf_path = os.path.join(UPLOAD_DIR, pdf_file.filename)
            with open(pdf_path, "wb") as f:
                f.write(await pdf_file.read())
            extracted_text = extract_text_from_pdf(pdf_path)
            if not extracted_text:
                raise HTTPException(status_code=400, detail="Unable to extract text from PDF")
            text_to_summarize = extracted_text
        else:
            text_to_summarize = (text_input or "").strip()
            if not text_to_summarize:
                raise HTTPException(status_code=400, detail="No text provided")

        if len(text_to_summarize) < 30:
            raise HTTPException(status_code=400, detail="Text is too short to summarize (min ~30 characters)")

        summary = summarize_text(text_to_summarize, model_choice=model_choice, summary_length=summary_length)
        return {
            "summary": summary,
            "model": model_choice,
            "original_length": len(text_to_summarize.split()),
            "summary_length": len(summary.split())
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Summarization failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/evaluate/rouge", response_model=_schemas.RougeEvalResponse)
def evaluate_rouge(payload: _schemas.RougeEvalRequest):
    """Compute ROUGE scores between a reference (gold) and candidate summary.

    Requires optional dependency rouge-score. If not installed, returns 503.
    """
    if rouge_scorer is None:
        raise HTTPException(status_code=503, detail="rouge-score library not installed. Add 'rouge-score' to requirements.txt")

    ref = (payload.reference or "").strip()
    cand = (payload.candidate or "").strip()
    if not ref or not cand:
        raise HTTPException(status_code=422, detail="Both reference and candidate must be non-empty")

    metrics = payload.metrics or ["rouge1", "rouge2", "rougeL"]
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=payload.use_stemmer)
    score_dict = scorer.score(ref, cand)

    # Format scores with 4-decimal rounding
    formatted = {}
    for k, v in score_dict.items():
        formatted[k] = _schemas.RougeScore(
            precision=round(v.precision, 4),
            recall=round(v.recall, 4),
            f1=round(v.fmeasure, 4)
        )

    return _schemas.RougeEvalResponse(
        scores=formatted,
        reference_tokens=len(ref.split()),
        candidate_tokens=len(cand.split())
    )


# -------------------- Visualization Endpoints --------------------

@app.post("/visualize/complexity")
def visualize_complexity_endpoint(payload: _schemas.ComplexityVisualizationRequest):
    """Generate complexity charts for comparing text samples.
    
    Accepts a list of texts and returns base64-encoded visualization charts.
    """
    try:
        from .visualizations import generate_complexity_charts
        
        # Analyze each text
        original_metrics = analyze_text_complexity(payload.original_text)
        paraphrased_metrics = []
        
        for text in payload.comparison_texts:
            paraphrased_metrics.append(analyze_text_complexity(text))
            
        # Generate visualization
        visualizations = generate_complexity_charts(original_metrics, paraphrased_metrics)
        return visualizations
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Visualization failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate visualization: {e}")


# -------------------- Paraphrasing Endpoint --------------------

@app.post("/paraphrase")
@app.post("/paraphrase/")
def paraphrase_endpoint(paraphrase_request: schemas.ParaphraseRequest, db: Session = Depends(get_db)):
    """
    Paraphrases the given text. Optionally saves to history (not implemented here) and
    returns simple complexity metrics for original and paraphrased variants.
    """
    try:
        # Support friendly aliases (pegasus, bart, flan-t5) using summarizer MODELS mapping
        requested_name = paraphrase_request.model_name or ""
        key = requested_name.lower()
        model_name = MODELS.get(key, requested_name)
        
        text = (paraphrase_request.text or "").strip()
        if not text:
            raise HTTPException(status_code=422, detail="text must be non-empty")
            
        creativity = float(paraphrase_request.creativity or 0.3)
        length = (paraphrase_request.length or "medium").lower()
        user_email = paraphrase_request.user_email

        # Call the imported paraphrase_text function
        result = paraphrase_text(
            text=text,
            model_name=model_name,
            creativity=creativity,
            length=length
        )
        
        return result
    except ValueError as e:
        # Convert ValueErrors from parahrase.py to appropriate HTTP errors
        if "transformers not available" in str(e):
            raise HTTPException(status_code=503, detail="transformers not available on server")
        elif "Failed to load model" in str(e):
            raise HTTPException(status_code=500, detail=str(e))
        else:
            raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Paraphrase failed")
        raise HTTPException(status_code=500, detail=f"Failed to generate paraphrase: {e}")
