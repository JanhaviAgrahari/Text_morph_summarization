import re
import os

# Define paths
app_path = r"c:\Users\janha\Desktop\Infosys Springboard 6.0 (Project)\Text_morph_summarization\Frontend\app.py"
metrics_path = r"c:\Users\janha\Desktop\Infosys Springboard 6.0 (Project)\Text_morph_summarization\additional_metrics.py"

# Read files
with open(app_path, 'r', encoding='utf-8') as f:
    app_content = f.read()
    
with open(metrics_path, 'r', encoding='utf-8') as f:
    metrics_content = f.read()

# Find the pattern in fine-tuned summarization tab and insert our metrics code
pattern = r'(                                except Exception as er:\s+                                    st\.error\(f"ROUGE evaluation failed: \{er\}"\)\s+)(\s+                except Exception as e:)'

# We need to find the one in the fine-tuned summarization tab
parts = app_content.split('# Tab 7 - Fine-tuned Summarization')
if len(parts) >= 2:
    # Look in second part only
    second_part = parts[1]
    
    # Replace in second part
    match = re.search(pattern, second_part)
    if match:
        modified_second_part = second_part.replace(
            match.group(0), 
            match.group(1) + "\n" + metrics_content + "\n" + match.group(2)
        )
        
        # Reconstruct full content
        modified_content = parts[0] + '# Tab 7 - Fine-tuned Summarization' + modified_second_part
        
        # Write back to file
        with open(app_path, 'w', encoding='utf-8') as f:
            f.write(modified_content)
        
        print("Successfully added metrics code")
    else:
        print("Pattern not found in Fine-tuned Summarization tab")
else:
    print("Couldn't find Fine-tuned Summarization tab")