from fastapi import FastAPI, Depends, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from . import models, schemas, crud
from .database import engine, SessionLocal, Base, get_db
from .emailer import send_password_reset_email
from .auth import create_access_token, get_current_user, require_mentor_or_admin
import json
import os
import logging
from functools import partial

# Local summarization and paraphrasing helpers
from .summarizer import summarize_text, MODELS
from .pdf_utils import extract_text_from_pdf
from .parahrase import paraphrase_text, analyze_text_complexity
from .fine_tuned_summarizer import summarize_text_with_fine_tuned_model, summarize_text_with_model
from .metrics import calculate_all_metrics
from . import schemas as _schemas
from . import schemas as schemas
from .metrics_schemas import MetricsRequest, MetricsResponse

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
    pdf_file: UploadFile = File(None),
    user_email: str = Form(None),  # Optional user email to save in history
    db: Session = Depends(get_db)
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

        # Offload heavy summarization to a worker thread to keep event loop responsive
        summary = await run_in_threadpool(
            partial(summarize_text, text_to_summarize, model_choice=model_choice, summary_length=summary_length)
        )
        
        # Save to history if user is provided
        if user_email:
            user = db.query(models.User).filter(models.User.email == user_email).first()
            if user:
                # Create a history entry
                history_entry = schemas.CreateHistoryEntry(
                    user_id=user.id,
                    type="summary",
                    original_text=text_to_summarize,
                    result_text=summary,
                    model=model_choice,
                    parameters=json.dumps({"summary_length": summary_length})
                )
                crud.create_history_entry(db, history_entry)
        
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


@app.post("/summarize/fine-tuned/")
async def fine_tuned_summarize_endpoint(
    summary_length: str = Form("medium"),
    input_type: str = Form("text"),  # "text" or "pdf"
    text_input: str = Form(""),
    pdf_file: UploadFile = File(None),
    model_choice: str = Form("t5"),  # 't5' or 'bart'
    user_email: str = Form(None),  # Optional user email to save in history
    db: Session = Depends(get_db)
):
    """Summarize plain text or an uploaded PDF using a local fine-tuned model (T5 or BART).
    
    summary_length: short | medium | long (mapped internally in summarizer)
    input_type: 'text' or 'pdf'
    model_choice: 't5' or 'bart'
    """
    try:
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

        # Use selected local model for summarization
        model_choice_norm = (model_choice or "t5").lower()
        if model_choice_norm not in {"t5", "bart"}:
            raise HTTPException(status_code=422, detail="model_choice must be 't5' or 'bart'")

        # Offload heavy summarization to a worker thread to keep event loop responsive
        summary = await run_in_threadpool(
            partial(summarize_text_with_model, text_to_summarize, summary_length=summary_length, model_choice=model_choice_norm)
        )
        
        # Save to history if user is provided
        if user_email:
            user = db.query(models.User).filter(models.User.email == user_email).first()
            if user:
                # Create a history entry
                history_entry = schemas.CreateHistoryEntry(
                    user_id=user.id,
                    type="summary",
                    original_text=text_to_summarize,
                    result_text=summary,
                    model=f"fine-tuned-{model_choice_norm}",
                    parameters=json.dumps({"summary_length": summary_length})
                )
                crud.create_history_entry(db, history_entry)
        
        return {
            "summary": summary,
            "model": f"fine-tuned-{model_choice_norm}",
            "original_length": len(text_to_summarize.split()),
            "summary_length": len(summary.split())
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Fine-tuned summarization failed")
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


@app.post("/evaluate/metrics", response_model=MetricsResponse)
def evaluate_metrics(payload: MetricsRequest):
    """
    Calculate comprehensive metrics between a reference and candidate summary.
    
    Metrics include:
    - BLEU scores (1-4 grams)
    - Perplexity (returned for candidate and reference)
    - Readability delta (original vs candidate if original_text is provided)
    - Readability delta (reference vs candidate)
    """
    try:
        ref = (payload.reference or "").strip()
        cand = (payload.candidate or "").strip()
        original = payload.original_text.strip() if payload.original_text else None
        
        if not ref or not cand:
            raise HTTPException(status_code=422, detail="Both reference and candidate must be non-empty")
            
        # Calculate all metrics
        results = calculate_all_metrics(ref, cand, original)
        
        # Format and return the results
        return results
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Metrics evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))


# -------------------- History Create-by-email --------------------

@app.post("/history")
def create_history_by_email(payload: schemas.CreateHistoryByEmail, db: Session = Depends(get_db)):
    """Create a history entry using the user's email instead of user_id.

    This helps the frontend log original/result pairs for fine-tuned summarization and
    paraphrasing without needing to look up user_id.
    """
    user = db.query(models.User).filter(models.User.email == payload.email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # --- Duplicate guard --------------------------------------------------
    # Streamlit sometimes re-triggers requests on reruns causing back-to-back
    # identical history inserts. We defensively look for an existing entry
    # created in the last few seconds with the same (user, type, model, texts).
    try:
        from datetime import datetime, timedelta, timezone as _tz
        window_start = datetime.now(_tz.utc) - timedelta(seconds=6)
        existing = db.query(models.History) \
            .filter(
                models.History.user_id == user.id,
                models.History.type == payload.type,
                models.History.model == payload.model,
                models.History.created_at >= window_start,
            ) \
            .order_by(models.History.created_at.desc()) \
            .all()
        norm_orig = (payload.original_text or "").strip()
        norm_res = (payload.result_text or "").strip()
        for e in existing:
            if e.original_text.strip() == norm_orig and e.result_text.strip() == norm_res:
                return {"id": e.id, "message": "Duplicate ignored", "duplicate": True}
    except Exception:
        # Fail open (still create) if any unexpected error in duplicate check
        pass

    entry = schemas.CreateHistoryEntry(
        user_id=user.id,
        type=payload.type,
        original_text=payload.original_text,
        result_text=payload.result_text,
        model=payload.model,
        parameters=payload.parameters,
    )
    db_entry = crud.create_history_entry(db, entry)
    return {"id": db_entry.id, "message": "History saved"}


# -------------------- History Endpoints --------------------

@app.get("/history", response_model=schemas.HistoryResponse)
def get_history(
    email: str, 
    type: str = None,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    """Get user's history of summaries and paraphrases.
    
    Args:
        email: User's email
        type: Optional filter by type ('summary' or 'paraphrase')
        limit: Maximum number of history entries to return
    """
    # Get the user by email
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get the history entries
    if type:
        entries = crud.get_user_history_by_type(db, user.id, type, limit)
    else:
        entries = crud.get_user_history(db, user.id, limit)
    
    return {"entries": entries}

@app.delete("/history/{entry_id}")
def delete_history_entry(
    entry_id: int,
    db: Session = Depends(get_db),
    email: str = None
):
    """Delete a specific history entry."""
    # Get the user by email
    user = db.query(models.User).filter(models.User.email == email).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Delete the entry
    deleted = crud.delete_history_entry(db, entry_id, user.id)
    if not deleted:
        raise HTTPException(status_code=404, detail="History entry not found or doesn't belong to user")
    
    return {"message": "History entry deleted successfully"}


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
    Paraphrases the given text. Saves to history if user_email is provided and
    returns complexity metrics for original and paraphrased variants.
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
        
        # Optional ROUGE evaluation
        if paraphrase_request.evaluate_rouge:
            try:
                if rouge_scorer is None:
                    result["rouge_evaluation"] = {
                        "available": False,
                        "reason": "rouge-score library not installed"
                    }
                else:
                    metrics = paraphrase_request.rouge_metrics or ["rouge1", "rouge2", "rougeL"]
                    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=paraphrase_request.use_stemmer)
                    
                    # Get reference paraphrase if provided, otherwise use original text as reference
                    ref_paraphrase = paraphrase_request.reference_paraphrase
                    has_ref_paraphrase = ref_paraphrase and ref_paraphrase.strip()
                    
                    paraphrase_scores = []
                    for item in result.get("paraphrased_results", []):
                        candidate = (item.get("text") or "").strip()
                        if not candidate:
                            paraphrase_scores.append({
                                "vs_original": {"scores": {}, "reference_tokens": len(text.split()), "candidate_tokens": 0},
                                "vs_reference": None if has_ref_paraphrase else {}
                            })
                            continue
                            
                        # ROUGE score against original text
                        score_dict_orig = scorer.score(text, candidate)
                        formatted_orig = {}
                        for k, v in score_dict_orig.items():
                            formatted_orig[k] = {
                                "precision": round(v.precision, 4),
                                "recall": round(v.recall, 4),
                                "f1": round(v.fmeasure, 4)
                            }
                        
                        # Optional ROUGE against reference paraphrase
                        vs_reference = None
                        if has_ref_paraphrase:
                            score_dict_ref = scorer.score(ref_paraphrase.strip(), candidate)
                            formatted_ref = {}
                            for k, v in score_dict_ref.items():
                                formatted_ref[k] = {
                                    "precision": round(v.precision, 4),
                                    "recall": round(v.recall, 4),
                                    "f1": round(v.fmeasure, 4)
                                }
                            vs_reference = {
                                "scores": formatted_ref,
                                "reference_tokens": len(ref_paraphrase.strip().split()),
                                "candidate_tokens": len(candidate.split())
                            }
                            
                        paraphrase_scores.append({
                            "vs_original": {
                                "scores": formatted_orig,
                                "reference_tokens": len(text.split()),
                                "candidate_tokens": len(candidate.split())
                            },
                            "vs_reference": vs_reference
                        })
                        
                    # Add visualization of ROUGE scores if visualization module is available
                    rouge_visualizations = {}
                    try:
                        from .visualizations import generate_rouge_chart
                        rouge_visualizations = generate_rouge_chart(paraphrase_scores, metrics)
                    except Exception as viz_exc:
                        logging.getLogger("uvicorn.error").warning(f"ROUGE visualization failed: {viz_exc}")
                    
                    result["rouge_evaluation"] = {
                        "available": True,
                        "metric_names": metrics,
                        "results": paraphrase_scores,
                        "visualizations": rouge_visualizations
                    }
            except Exception as rouge_exc:
                logging.getLogger("uvicorn.error").warning(f"ROUGE evaluation failed: {rouge_exc}")
                result["rouge_evaluation"] = {"available": False, "reason": str(rouge_exc)}
        
        # Save to history if user email is provided
        if user_email:
            user = db.query(models.User).filter(models.User.email == user_email).first()
            if user:
                # Create a history entry
                parameters = {
                    "creativity": creativity,
                    "length": length
                }
                # Extract the first paraphrase text from result
                first_text = ""
                try:
                    pr = result.get("paraphrased_results") or []
                    if isinstance(pr, list) and pr:
                        first_text = pr[0].get("text", "")
                except Exception:
                    first_text = ""

                history_entry = schemas.CreateHistoryEntry(
                    user_id=user.id,
                    type="paraphrase",
                    original_text=text,
                    result_text=first_text,
                    model=model_name,
                    parameters=json.dumps(parameters)
                )
                crud.create_history_entry(db, history_entry)
        
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


@app.get("/admin/history", response_model=schemas.AdminHistoryResponse)
def get_all_user_history(
    type_filter: str = None,
    limit: int = 1000,
    current_user: models.User = Depends(require_mentor_or_admin),
    db: Session = Depends(get_db)
):
    """Get all history entries across all users (mentor/admin only).
    
    Args:
        type_filter: Optional filter by type ('summary' or 'paraphrase')
        limit: Maximum number of entries to return (default 1000)
    
    Returns:
        AdminHistoryResponse with entries including user details
    """
    try:
        # Get all history entries with user data
        history_entries = crud.get_all_history(db, limit=limit, type_filter=type_filter)
        
        # Transform to AdminHistoryEntry format
        admin_entries = []
        for entry in history_entries:
            admin_entry = schemas.AdminHistoryEntry(
                id=entry.id,
                user_id=entry.user_id,
                user_email=entry.user.email,
                user_name=entry.user.name,
                type=entry.type,
                original_text=entry.original_text,
                result_text=entry.result_text,
                model=entry.model,
                created_at=entry.created_at,
                parameters=entry.parameters
            )
            admin_entries.append(admin_entry)
        
        return schemas.AdminHistoryResponse(
            entries=admin_entries,
            total_count=len(admin_entries)
        )
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Failed to fetch admin history")
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {e}")
