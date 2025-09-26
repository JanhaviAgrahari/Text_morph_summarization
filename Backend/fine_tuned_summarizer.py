"""Utilities to load and run local fine-tuned/pretrained summarization models.

Supports:
- T5 (folder: fine_tuned_t5_summarizer)
- BART (folder: fine_tuned_bart_summarizer)
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# Local model directories
_ROOT = os.path.dirname(os.path.dirname(__file__))
T5_MODEL_DIR = os.path.abspath(os.path.join(_ROOT, "fine_tuned_t5_summarizer"))
BART_MODEL_DIR = os.path.abspath(os.path.join(_ROOT, "fine_tuned_bart_summarizer"))

# Cache of loaded models: {"t5": (tok, model, device), "bart": (...)}
_LOADED: dict[str, tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]] = {}


def _load_model(model_choice: str) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM, torch.device]:
    """Load and cache a local model by choice: 't5' or 'bart'."""
    model_choice = (model_choice or "t5").lower()
    if model_choice in _LOADED:
        return _LOADED[model_choice]

    if model_choice == "t5":
        model_dir = T5_MODEL_DIR
    elif model_choice == "bart":
        model_dir = BART_MODEL_DIR
    else:
        raise ValueError("model_choice must be 't5' or 'bart'")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found for {model_choice}: {model_dir}")

    logging.info(f"Loading {model_choice.upper()} model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    _LOADED[model_choice] = (tokenizer, model, device)
    return tokenizer, model, device

def _chunk_text_by_words(text: str, max_words: int = 800):
    """Yield chunks of text roughly max_words each (simple word-based chunking)."""
    words = text.split()
    if not words:
        return
    for i in range(0, len(words), max_words):
        yield " ".join(words[i:i + max_words])

def summarize_text_with_model(text: str, summary_length: str = "medium", model_choice: str = "t5") -> str:
    """Summarize text using a local model ('t5' or 'bart')."""
    if not text or not text.strip():
        return "Error: no text to summarize."

    tokenizer, model, device = _load_model(model_choice)
    
    # Use fixed token length bounds similar to the T5-small behavior
    length_map = {
        "short": (30, 80),
        "medium": (80, 150),
        "long": (150, 300)
    }
    min_len, max_len = length_map.get(summary_length, length_map["medium"])

    def _too_similar(src: str, cand: str) -> bool:
        """Heuristic to detect near-copy outputs."""
        if not src or not cand:
            return False
        s = src.strip().lower()
        c = cand.strip().lower()
        if s == c:
            return True
        # length-based check
        src_words = s.split()
        cand_words = c.split()
        if not cand_words:
            return False
        len_ratio = min(len(cand_words) / max(len(src_words), 1), 1.0)
        # token overlap (Jaccard over unique words)
        set_src = set(src_words)
        set_cand = set(cand_words)
        overlap = len(set_src & set_cand) / max(len(set_cand), 1)
        return (len_ratio > 0.95) or (overlap > 0.95)

    def _generate_once(inp_text: str, min_l: int, max_l: int) -> str:
        # Tokenize and move to device
        inputs = tokenizer(inp_text, truncation=True, padding="longest", return_tensors="pt", max_length=1024)
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        gen_kwargs = dict(
            max_length=max_l,
            min_length=min_l,
            num_beams=6,
            length_penalty=2.0,
            early_stopping=True,
        )
        # Apply BART-specific anti-copy constraints
        if model_choice.lower() == "bart":
            gen_kwargs.update({
                "no_repeat_ngram_size": 3,
                "encoder_no_repeat_ngram_size": 3,
                "repetition_penalty": 1.1,
            })
            if getattr(tokenizer, "bos_token_id", None) is not None:
                gen_kwargs["forced_bos_token_id"] = tokenizer.bos_token_id

        summary_ids = model.generate(inputs["input_ids"], **gen_kwargs)
        out = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()

        # If looks copied (BART sometimes copies), retry with stricter settings
        if model_choice.lower() == "bart" and _too_similar(inp_text, out):
            stricter = dict(
                max_length=max(int(max_l * 0.9), min_l + 5),
                min_length=min_l,
                num_beams=4,
                length_penalty=2.5,
                no_repeat_ngram_size=4,
                encoder_no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                early_stopping=True,
            )
            if getattr(tokenizer, "bos_token_id", None) is not None:
                stricter["forced_bos_token_id"] = tokenizer.bos_token_id
            summary_ids = model.generate(inputs["input_ids"], **stricter)
            out = tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
        return out

    # Collect chunk summaries
    chunk_summaries = []
    for chunk in _chunk_text_by_words(text, max_words=800):
        chunk_summary = _generate_once(chunk, min_len, max_len)
        chunk_summaries.append(chunk_summary)

    # If multiple chunk summaries, summarize the concatenation to get a single coherent output
    if len(chunk_summaries) == 0:
        return "Error: nothing to summarize."
    elif len(chunk_summaries) == 1:
        return chunk_summaries[0]
    else:
        combined = " ".join(chunk_summaries)
        # Final pass to compress with the same anti-copy safeguards
        min_len_final = max(min_len, 20)
        max_len_final = max_len
        final_summary = _generate_once(combined, min_len_final, max_len_final)
        return final_summary.strip()


# Backward-compatible name used by existing code (defaults to T5)
def summarize_text_with_fine_tuned_model(text: str, summary_length: str = "medium") -> str:
    return summarize_text_with_model(text, summary_length=summary_length, model_choice="t5")