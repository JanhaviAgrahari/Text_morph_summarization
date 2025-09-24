from pydantic import BaseModel

class MetricsRequest(BaseModel):
    reference: str
    candidate: str
    original_text: str | None = None  # Optional original text for readability delta (original vs candidate)

class MetricsResponse(BaseModel):
    # BLEU scores and cumulative BLEU
    bleu: dict
    # Backward-compatible candidate perplexity (same as perplexity_candidate)
    perplexity: dict
    # Optional expanded fields
    perplexity_candidate: dict | None = None
    perplexity_reference: dict | None = None
    # Readability comparisons
    readability: dict | None = None  # original vs candidate (if original_text provided)
    readability_ref_candidate: dict | None = None  # reference vs candidate delta