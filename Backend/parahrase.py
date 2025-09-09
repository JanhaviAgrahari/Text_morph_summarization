"""
Paraphrasing module using HuggingFace transformers for text paraphrasing.
Supports both T5 and Pegasus models with appropriate prompt handling.
"""
import logging
from typing import Dict, Tuple, List, Optional

# Optional imports with fallbacks
try:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError:
    AutoModelForSeq2SeqLM = None
    AutoTokenizer = None

try:
    import textstat as _textstat
except ImportError:
    _textstat = None

# Import visualization module
try:
    from .visualizations import generate_complexity_charts
except ImportError:
    generate_complexity_charts = None

# Cache loaded models/tokenizers to avoid reloading
paraphrasing_pipelines: Dict[str, Tuple[object, object]] = {}

def analyze_text_complexity(text: str) -> dict:
    """Return simple readability metrics for a given text. Falls back gracefully."""
    if not _textstat:
        return {"available": False}
    try:
        return {
            "flesch_reading_ease": float(_textstat.flesch_reading_ease(text)),
            "gunning_fog": float(_textstat.gunning_fog(text)),
            "smog_index": float(_textstat.smog_index(text)),
            "word_count": len(text.split()),
        }
    except Exception:
        return {"available": False}

def clamp(v, lo, hi):
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, int(v)))

def paraphrase_text(
    text: str, 
    model_name: str, 
    creativity: float = 0.3, 
    length: str = "medium"
) -> dict:
    """
    Paraphrases the given text using the specified model.
    
    Args:
        text: Text to paraphrase
        model_name: HuggingFace model name/path
        creativity: Controls temperature and top_p (0.0-1.0)
        length: Target length ("short", "medium", "long")
    
    Returns:
        Dict with original_text_analysis and paraphrased_results
    
    Raises:
        ValueError: If parameters or transformers are invalid
    """
    if AutoModelForSeq2SeqLM is None or AutoTokenizer is None:
        raise ValueError("transformers not available")

    text = (text or "").strip()
    if not text:
        raise ValueError("text must be non-empty")
        
    # Lazy-load HF model/tokenizer
    if model_name not in paraphrasing_pipelines:
        try:
            # This can be slow on first run while weights are downloaded
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            paraphrasing_pipelines[model_name] = (model, tokenizer)
        except Exception as e:
            raise ValueError(f"Failed to load model: {e}")

    model, tokenizer = paraphrasing_pipelines[model_name]

    # Determine target lengths for proper paraphrasing (should match original more closely)
    original_word_count = max(1, len(text.split()))
    # For paraphrasing, we want lengths closer to original (not summarizing)
    # approximate tokens ~ words * 1.3, then clamp to safe bounds
    if length == "short":
        max_new = clamp(original_word_count * 1.0 * 1.3, 32, 512)
        min_new = clamp(original_word_count * 0.8 * 1.3, 24, 384)
    elif length == "long":
        max_new = clamp(original_word_count * 1.5 * 1.3, 64, 768) 
        min_new = clamp(original_word_count * 1.2 * 1.3, 48, 512)
    else:  # medium
        max_new = clamp(original_word_count * 1.2 * 1.3, 48, 640)
        min_new = clamp(original_word_count * 1.0 * 1.3, 32, 512)

    temperature = 0.5 + creativity
    top_p = min(0.99, 0.85 + (creativity / 10.0))

    try:
        # Add appropriate prefix depending on model type for proper paraphrasing 
        model_name_lower = (model_name or "").lower()
        
        # Different models need different prompting techniques for full paraphrasing
        if "t5" in model_name_lower:
            prompt = "paraphrase: " + text
        elif "bart" in model_name_lower:
            prompt = "Paraphrase the following while keeping the original meaning: " + text
        elif "flan" in model_name_lower:
            prompt = "Rewrite the following text in a different way but keep the same meaning: " + text
        else:
            # Pegasus and default case - requires more careful param tuning to produce proper paraphrases
            prompt = text
            
        inputs = tokenizer([prompt], return_tensors="pt", truncation=True, padding="longest")
        
        # Generate first paraphrase with specified parameters
        output_ids_1 = model.generate(
            **inputs,
            max_new_tokens=int(max_new),
            min_new_tokens=int(min_new),
            num_return_sequences=1,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            length_penalty=1.0,  # Favor completeness (>1.0 = longer outputs)
            no_repeat_ngram_size=3,  # Avoid repetition
            repetition_penalty=1.2,  # Penalize repeated tokens
            early_stopping=True,
        )
        
        # Generate second paraphrase with a different approach - using beam search for diversity
        # We'll use num_beams with num_beam_groups and diversity_penalty for the second generation
        output_ids_2 = model.generate(
            **inputs,
            max_new_tokens=int(max_new),
            min_new_tokens=int(min_new),
            num_beams=4,                   # Use beam search
            num_beam_groups=4,             # With diverse beam groups
            diversity_penalty=1.0,         # Apply diversity penalty between groups
            do_sample=False,               # Must be False when using diversity_penalty
            num_return_sequences=1,
            length_penalty=1.2,            # Slightly different length preference
            no_repeat_ngram_size=2,        # Different repetition settings
            repetition_penalty=1.3,        # Stronger repetition penalty
            early_stopping=True,
        )
        
        # Combine both outputs
        output_ids = [output_ids_1[0], output_ids_2[0]]
        paraphrased_texts = [
            tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in output_ids
        ]

        # Fallback retry with conservative settings if output is empty, too short, or generation failed
        retry = False
        if not any(p.strip() for p in paraphrased_texts):
            retry = True
        else:
            # Check if paraphrases are too short (< 70% of original)
            original_token_count = len(text.split())
            if all(len(p.split()) < original_token_count * 0.7 for p in paraphrased_texts):
                retry = True
            
            # Check if paraphrases are too similar to each other (indicating lack of diversity)
            # Simple heuristic: check if more than 80% of words are the same
            if len(paraphrased_texts) >= 2:
                words1 = set(paraphrased_texts[0].lower().split())
                words2 = set(paraphrased_texts[1].lower().split())
                common_words = words1.intersection(words2)
                
                # If the paraphrases share too many words, force a retry
                if len(words1) > 0 and len(common_words) / len(words1) > 0.8:
                    retry = True
                
        if retry:
            logging.info("Paraphrase too short or empty - retrying with more conservative settings")
            # Try again with more conservative settings and more explicit prompt
            if "t5" in model_name_lower:
                alternative_prompt = "paraphrase this completely: " + text
            elif "bart" in model_name_lower or "flan" in model_name_lower:
                alternative_prompt = "Provide a complete paraphrase of this text with similar length: " + text
            else:
                alternative_prompt = text
                
            alt_inputs = tokenizer([alternative_prompt], return_tensors="pt", truncation=True, padding="longest")
            
            # First retry with conservative settings
            output_ids_1 = model.generate(
                **alt_inputs,
                max_new_tokens=int(max(original_word_count * 1.5, 64)),
                min_new_tokens=int(max(original_word_count * 0.8, 24)),
                num_return_sequences=1,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                length_penalty=1.5,  # Strongly favor longer outputs
                no_repeat_ngram_size=2,
                repetition_penalty=1.1,
                early_stopping=True,
            )
            
            # Second retry with beam search for diversity
            output_ids_2 = model.generate(
                **alt_inputs,
                max_new_tokens=int(max(original_word_count * 1.5, 64)),
                min_new_tokens=int(max(original_word_count * 0.8, 24)),
                num_return_sequences=1,
                do_sample=False,            # Must be False when using diversity_penalty
                num_beams=4,                # Use beam search
                num_beam_groups=4,          # With diverse beam groups
                diversity_penalty=1.0,      # Apply diversity penalty between groups
                length_penalty=1.7,         # Even stronger favor for longer outputs
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                early_stopping=True,
            )
            
            # Combine the outputs
            output_ids = [output_ids_1[0], output_ids_2[0]]
            paraphrased_texts = [
                tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for g in output_ids
            ]

        # Additional diversity check and post-processing
        if len(paraphrased_texts) >= 2:
            # If we still have too similar outputs, try to modify one of them
            words1 = set(paraphrased_texts[0].lower().split())
            words2 = set(paraphrased_texts[1].lower().split())
            common_words = words1.intersection(words2)
            
            if len(words1) > 0 and len(words2) > 0 and len(common_words) / min(len(words1), len(words2)) > 0.75:
                # Use a different model-specific approach for each variant
                if "t5" in model_name_lower:
                    # For T5 models - try a completely different approach with beam search
                    final_prompt = "rewrite completely differently: " + text
                    alt_inputs = tokenizer([final_prompt], return_tensors="pt", truncation=True, padding="longest")
                    alt_output_ids = model.generate(
                        **alt_inputs,
                        max_new_tokens=int(max_new),
                        min_new_tokens=int(min_new),
                        num_return_sequences=1,
                        do_sample=False,            # Switch to beam search for diversity
                        num_beams=5,                # Use beam search
                        num_beam_groups=5,          # With diverse beam groups
                        diversity_penalty=1.5,      # Strong diversity penalty
                        no_repeat_ngram_size=2,
                        early_stopping=True,
                    )
                    paraphrased_texts[1] = tokenizer.decode(alt_output_ids[0], skip_special_tokens=True)
                elif "bart" in model_name_lower:
                    # For BART models
                    final_prompt = "Completely rephrase the following with different words: " + text
                    alt_inputs = tokenizer([final_prompt], return_tensors="pt", truncation=True, padding="longest")
                    alt_output_ids = model.generate(
                        **alt_inputs,
                        max_new_tokens=int(max_new),
                        min_new_tokens=int(min_new),
                        num_return_sequences=1,
                        do_sample=False,              # Switch to beam search for diversity
                        num_beams=5,                  # Use beam search
                        num_beam_groups=5,            # With diverse beam groups
                        diversity_penalty=1.5,        # Strong diversity penalty
                        length_penalty=1.0,
                        early_stopping=True,
                    )
                    paraphrased_texts[1] = tokenizer.decode(alt_output_ids[0], skip_special_tokens=True)
        
        original_analysis = analyze_text_complexity(text)
        paraphrased_results = []
        paraphrased_metrics_list = []
        
        for p_text in paraphrased_texts:
            complexity = analyze_text_complexity(p_text)
            paraphrased_results.append({
                "text": p_text,
                "complexity": complexity,
            })
            paraphrased_metrics_list.append(complexity)

        # Generate visualization charts if available
        visualizations = {}
        if generate_complexity_charts is not None:
            try:
                visualizations = generate_complexity_charts(
                    original_analysis, 
                    paraphrased_metrics_list
                )
            except Exception as e:
                logging.warning(f"Failed to generate complexity charts: {e}")
        
        return {
            "original_text_analysis": original_analysis,
            "paraphrased_results": paraphrased_results,
            "visualizations": visualizations,
        }
    except Exception as e:
        logging.getLogger("uvicorn.error").exception("Paraphrase failed")
        raise ValueError(f"Failed to generate paraphrase: {e}")
