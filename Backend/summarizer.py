# Backend/summarizer.py
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Map friendly names to HF model ids
MODELS = {
    "pegasus": "google/pegasus-xsum",
    "bart": "facebook/bart-large-cnn",
    "flan-t5": "google/flan-t5-large"
}

# Cache loaded (tokenizer, model, device)
_loaded = {}

def load_model(model_key: str):
    """Load and cache model/tokenizer, move model to device."""
    if model_key not in MODELS:
        raise ValueError(f"Unsupported model: {model_key}. Choose one of {list(MODELS.keys())}")
    if model_key in _loaded:
        return _loaded[model_key]

    model_name = MODELS[model_key]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _loaded[model_key] = (tokenizer, model, device)
    return tokenizer, model, device

def _chunk_text_by_words(text: str, max_words: int = 800):
    """Yield chunks of text roughly max_words each (simple word-based chunking)."""
    words = text.split()
    if not words:
        return
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def summarize_text(text: str, model_choice: str = "pegasus", summary_length: str = "medium") -> str:
    """
    High-level summarization that:
      - loads/caches model
      - chunk long texts
      - summarizes each chunk and (if multiple chunks) produces a final concise summary
    Returns final summary string.
    """
    if not text or not text.strip():
        return "Error: no text to summarize."

    tokenizer, model, device = load_model(model_choice)

    # length map in tokens (approx)
    length_map = {
        "short": (30, 80),
        "medium": (80, 150),
        "long": (150, 300)
    }
    min_len, max_len = length_map.get(summary_length, length_map["medium"])

    # Collect chunk summaries
    chunk_summaries = []
    for chunk in _chunk_text_by_words(text, max_words=800):
        # tokenize and move to device
        inputs = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # generate
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=6,
            length_penalty=2.0,
            early_stopping=True,
        )
        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(chunk_summary.strip())

    # If multiple chunk summaries, summarize the concatenation to get a single coherent output
    if len(chunk_summaries) == 0:
        return "Error: nothing to summarize."
    elif len(chunk_summaries) == 1:
        return chunk_summaries[0]
    else:
        combined = " ".join(chunk_summaries)
        # Final pass to compress
        inputs = tokenizer(combined, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        final_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=max(min_len, 20),
            num_beams=6,
            length_penalty=2.0,
            early_stopping=True,
        )
        final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)
        return final_summary.strip()
