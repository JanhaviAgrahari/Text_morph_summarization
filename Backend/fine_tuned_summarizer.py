# Backend/fine_tuned_summarizer.py
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging

# Path to the fine-tuned model directory
FINE_TUNED_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(__file__)), "fine_tuned_t5_summarizer"))

# Check if model exists
if not os.path.isdir(FINE_TUNED_MODEL_PATH):
    raise FileNotFoundError(f"Fine-tuned model directory not found: {FINE_TUNED_MODEL_PATH}")

# Cache loaded model and tokenizer
_loaded_fine_tuned = None

def load_fine_tuned_model():
    """Load and cache the fine-tuned T5 model and tokenizer."""
    global _loaded_fine_tuned
    
    if _loaded_fine_tuned is not None:
        return _loaded_fine_tuned

    try:
        logging.info(f"Loading fine-tuned T5 model from: {FINE_TUNED_MODEL_PATH}")
        tokenizer = AutoTokenizer.from_pretrained(FINE_TUNED_MODEL_PATH)
        model = AutoModelForSeq2SeqLM.from_pretrained(FINE_TUNED_MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        _loaded_fine_tuned = (tokenizer, model, device)
        return tokenizer, model, device
    except Exception as e:
        logging.error(f"Failed to load fine-tuned model: {str(e)}")
        raise RuntimeError(f"Failed to load fine-tuned model: {str(e)}")

def _chunk_text_by_words(text: str, max_words: int = 800):
    """Yield chunks of text roughly max_words each (simple word-based chunking)."""
    words = text.split()
    if not words:
        return
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def summarize_text_with_fine_tuned_model(text: str, summary_length: str = "medium") -> str:
    """
    Summarize text using the T5 model.
    
    Args:
        text: Input text to summarize
        summary_length: 'short', 'medium', or 'long'
        
    Returns:
        Summarized text
    """
    if not text or not text.strip():
        return "Error: no text to summarize."

    tokenizer, model, device = load_fine_tuned_model()

    # Length map in tokens (approx)
    length_map = {
        "short": (30, 80),
        "medium": (80, 150),
        "long": (150, 300)
    }
    min_len, max_len = length_map.get(summary_length, length_map["medium"])

    # Collect chunk summaries
    chunk_summaries = []
    for chunk in _chunk_text_by_words(text, max_words=800):
        # Tokenize and move to device
        inputs = tokenizer(chunk, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Generate summary
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