# Add this block after ROUGE evaluation but before the outer exception handler
# Calculate additional metrics (BLEU, perplexity, readability delta)
with st.spinner("Calculating additional metrics..."):
    try:
        metrics_payload = {
            "reference": reference_input.strip(),
            "candidate": result["summary"],
            "original_text": original_text_display if original_text_display else None
        }
        
        metrics_resp = requests.post(f"{BACKEND_URL}/evaluate/metrics", json=metrics_payload, timeout=120)
        metrics_json = metrics_resp.json()
        
        if "error" not in metrics_json:
            st.markdown("### Additional Quality Metrics")
            
            # BLEU scores
            if "bleu" in metrics_json and "error" not in metrics_json["bleu"]:
                bleu_scores = metrics_json["bleu"]
                st.markdown("#### BLEU Scores")
                bleu_cols = st.columns(5)
                
                with bleu_cols[0]:
                    st.metric("BLEU-1", f"{bleu_scores['bleu_1']:.4f}")
                with bleu_cols[1]:
                    st.metric("BLEU-2", f"{bleu_scores['bleu_2']:.4f}")
                with bleu_cols[2]:
                    st.metric("BLEU-3", f"{bleu_scores['bleu_3']:.4f}")
                with bleu_cols[3]:
                    st.metric("BLEU-4", f"{bleu_scores['bleu_4']:.4f}")
                with bleu_cols[4]:
                    st.metric("BLEU", f"{bleu_scores['bleu']:.4f}")
                
                st.caption("BLEU scores measure the similarity between the generated summary and the reference. Higher scores (0-1) indicate better matches.")
            
            # Perplexity
            if "perplexity" in metrics_json and "error" not in metrics_json["perplexity"]:
                perplexity = metrics_json["perplexity"]["perplexity"]
                ngram = metrics_json["perplexity"]["ngram"]
                st.markdown("#### Perplexity")
                st.metric("Perplexity", f"{perplexity:.2f}")
                st.caption(f"Perplexity measures how 'surprised' the model is by the text. Lower values indicate more fluent text. (Using {ngram}-gram model)")
            
            # Readability delta
            if "readability" in metrics_json and "error" not in metrics_json["readability"] and metrics_json["readability"].get("delta"):
                delta = metrics_json["readability"]["delta"]
                original = metrics_json["readability"]["original"]
                summary = metrics_json["readability"]["summary"]
                
                st.markdown("#### Readability Metrics")
                
                import pandas as pd
                
                # Create a dataframe with the readability metrics
                metrics_df = pd.DataFrame({
                    "Metric": [
                        "Flesch Reading Ease",
                        "Flesch-Kincaid Grade",
                        "Gunning Fog",
                        "SMOG Index",
                        "Automated Readability",
                        "Coleman-Liau Index",
                        "Dale-Chall Score"
                    ],
                    "Original": [
                        original["flesch_reading_ease"],
                        original["flesch_kincaid_grade"],
                        original["gunning_fog"],
                        original["smog_index"],
                        original["automated_readability_index"],
                        original["coleman_liau_index"],
                        original["dale_chall_readability_score"]
                    ],
                    "Summary": [
                        summary["flesch_reading_ease"],
                        summary["flesch_kincaid_grade"],
                        summary["gunning_fog"],
                        summary["smog_index"],
                        summary["automated_readability_index"],
                        summary["coleman_liau_index"],
                        summary["dale_chall_readability_score"]
                    ],
                    "Delta": [
                        delta["flesch_reading_ease"],
                        delta["flesch_kincaid_grade"],
                        delta["gunning_fog"],
                        delta["smog_index"],
                        delta["automated_readability_index"],
                        delta["coleman_liau_index"],
                        delta["dale_chall_readability_score"]
                    ]
                })
                
                # Format the delta column with a + sign for positive values
                metrics_df["Delta"] = metrics_df["Delta"].apply(lambda x: f"+{x:.2f}" if x > 0 else f"{x:.2f}")
                
                # Display the dataframe
                st.dataframe(
                    metrics_df,
                    column_config={
                        "Original": st.column_config.NumberColumn(format="%.2f"),
                        "Summary": st.column_config.NumberColumn(format="%.2f")
                    }
                )
                
                st.caption("Readability metrics compare the complexity of the original text vs. the summary. For Flesch Reading Ease, higher scores mean easier readability. For all others, lower scores indicate easier readability.")
        else:
            st.warning(f"Failed to calculate additional metrics: {metrics_json.get('error', 'Unknown error')}")
    except Exception as metrics_err:
        st.warning(f"Failed to calculate additional metrics: {metrics_err}")