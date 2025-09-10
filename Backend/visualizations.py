"""
Visualization module for creating complexity analysis charts.
"""
import base64
import io
from typing import List, Dict, Any, Optional, Tuple

try:
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from matplotlib.figure import Figure
except ImportError:
    plt = None
    pd = None
    np = None

def categorize_complexity(score: float, metric: str = "flesch_reading_ease") -> str:
    """
    Categorize text complexity based on readability metrics.
    
    Args:
        score: The score value
        metric: The type of metric (default: flesch_reading_ease)
        
    Returns:
        Category: "Beginner", "Intermediate", or "Advanced"
    """
    if metric == "flesch_reading_ease":
        if score >= 80:
            return "Beginner"
        elif score >= 50:
            return "Intermediate"
        else:
            return "Advanced"
    elif metric in ("gunning_fog", "smog_index"):
        if score <= 8:
            return "Beginner"
        elif score <= 12:
            return "Intermediate"
        else:
            return "Advanced"
    else:
        # Default categorization
        return "Intermediate"

def generate_rouge_chart(rouge_results: List[Dict[str, Any]], 
                    metric_names: List[str] = None) -> Dict[str, str]:
    """
    Generate visualization chart comparing ROUGE scores of paraphrased texts.
    
    Args:
        rouge_results: List of ROUGE results for each paraphrased version
        metric_names: List of ROUGE metric names to visualize
        
    Returns:
        Dictionary with base64 encoded PNG image for ROUGE scores
    """
    if plt is None or pd is None or np is None:
        return {"error": "Visualization libraries not available"}
    
    if not metric_names:
        metric_names = ["rouge1", "rouge2", "rougeL"]
    
    # Extract ROUGE scores
    data = []
    for i, result in enumerate(rouge_results):
        option_name = f"Option {i+1}"
        
        # Get scores vs original or reference
        vs_data = result.get("vs_original", {})
        if result.get("vs_reference") is not None:
            # Prioritize reference if available
            vs_data = result.get("vs_reference", {})
            
        scores = vs_data.get("scores", {})
        for metric in metric_names:
            if metric in scores:
                # Use F1 score (most common ROUGE metric)
                f1_score = scores[metric].get("f1", 0)
                data.append({
                    "option": option_name,
                    "metric": metric.upper(),
                    "value": f1_score
                })
    
    if not data:
        return {"error": "No valid ROUGE data available"}
    
    df = pd.DataFrame(data)
    
    # Generate ROUGE comparison chart
    rouge_img = io.BytesIO()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor('#1E1E1E')  # Dark background
    ax.set_facecolor('#1E1E1E')
    
    # Transform data for grouped bar chart
    pivot_df = df.pivot(index="metric", columns="option", values="value").reset_index()
    
    # Set metric as index for plotting
    pivot_df = pivot_df.set_index("metric")
    
    # Create color palette
    option_colors = {
        "Option 1": "#9BD0F5",  # Light blue
        "Option 2": "#1F77B4",  # Blue
        "Option 3": "#FFA5A5"   # Light red
    }
    
    # Create a list of colors matching columns
    colors = [option_colors.get(col, f"C{i}") for i, col in enumerate(pivot_df.columns)]
    
    # Plot the bar chart
    pivot_df.plot(kind="bar", ax=ax, color=colors, width=0.7, 
                 edgecolor='black', linewidth=0.5)
    
    # Customize the chart
    ax.set_title("ROUGE Score Comparison", color='white', fontsize=14, pad=20)
    ax.set_xlabel("Metric", color='white', fontsize=12)
    ax.set_ylabel("F1-Score (Higher is Better)", color='white', fontsize=12)
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    
    # Add a legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper right", title="Option", frameon=False,
             title_fontsize=12, fontsize=10, labelcolor='white')
    plt.setp(ax.get_legend().get_title(), color='white')
    
    # Remove grid and spines
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    fig.savefig(rouge_img, format='png', bbox_inches='tight')
    rouge_img.seek(0)
    
    # Convert image to base64
    rouge_base64 = base64.b64encode(rouge_img.read()).decode('utf-8')
    
    return {'rouge_chart': rouge_base64}

def generate_complexity_charts(original_metrics: Dict[str, float], 
                              paraphrased_metrics_list: List[Dict[str, float]]) -> Dict[str, str]:
    """
    Generate visualization charts comparing original and paraphrased text complexity.
    
    Args:
        original_metrics: Dictionary of metrics for the original text
        paraphrased_metrics_list: List of metric dictionaries for paraphrased versions
        
    Returns:
        Dictionary with two base64 encoded PNG images:
        - 'breakdown': Bar chart showing complexity breakdown
        - 'profile': Line chart showing complexity profile
    """
    if plt is None or pd is None or np is None:
        return {"error": "Visualization libraries not available"}
    
    # Prepare data for visualization
    data = []
    sources = ["Original"] + [f"Option {i+1}" for i in range(len(paraphrased_metrics_list))]
    
    # Extract metrics for each text version
    all_metrics = [original_metrics] + paraphrased_metrics_list
    
    # Create DataFrame for visualization
    for i, metrics in enumerate(all_metrics):
        if not metrics or metrics.get("available") is False:
            continue
            
        source = sources[i]
        flesch = metrics.get("flesch_reading_ease", 0)
        fog = metrics.get("gunning_fog", 0)
        smog = metrics.get("smog_index", 0)
        
        # Categorize based on Flesch reading ease score
        category = categorize_complexity(flesch)
        
        data.append({
            "source": source, 
            "metric": "flesch_reading_ease", 
            "value": flesch,
            "category": category
        })
        data.append({
            "source": source, 
            "metric": "gunning_fog", 
            "value": fog,
            "category": category
        })
        data.append({
            "source": source, 
            "metric": "smog_index", 
            "value": smog,
            "category": category
        })
    
    if not data:
        return {"error": "No valid metrics data available"}
    
    df = pd.DataFrame(data)
    
    # Create complexity category distribution
    category_df = pd.DataFrame([
        {"source": source, 
         "category": categorize_complexity(metrics.get("flesch_reading_ease", 50)), 
         "percentage": 100}
        for source, metrics in zip(sources, all_metrics)
        if metrics and metrics.get("available") is not False
    ])
    
    # Generate breakdown chart (stacked bar chart)
    breakdown_img = io.BytesIO()
    fig1, ax1 = plt.subplots(figsize=(10, 6), dpi=100)
    fig1.patch.set_facecolor('#1E1E1E')  # Dark background
    ax1.set_facecolor('#1E1E1E')
    
    categories = ["Beginner", "Intermediate", "Advanced"]
    colors = {"Beginner": "#4B71CA", "Intermediate": "#3CB44B", "Advanced": "#E74C3C"}
    
    # Calculate percentages for each source and category
    result_data = []
    for source in sources:
        if source not in category_df["source"].values:
            continue
            
        cat = category_df[category_df["source"] == source]["category"].iloc[0]
        for category in categories:
            value = 100 if category == cat else 0
            result_data.append({"source": source, "category": category, "percentage": value})
    
    result_df = pd.DataFrame(result_data)
    pivot_df = result_df.pivot(index="source", columns="category", values="percentage")
    pivot_df = pivot_df.fillna(0)
    
    # Ensure all categories exist
    for cat in categories:
        if cat not in pivot_df.columns:
            pivot_df[cat] = 0
    
    # Sort columns in desired order
    pivot_df = pivot_df[categories]
    
    # Create the stacked bar chart
    pivot_df.plot(kind="bar", stacked=True, ax=ax1, color=[colors[c] for c in pivot_df.columns], 
                 width=0.8, edgecolor='black', linewidth=0.5)
    
    # Customize the chart
    ax1.set_title("Complexity Breakdown", color='white', fontsize=14, pad=20)
    ax1.set_xlabel("Source", color='white', fontsize=12)
    ax1.set_ylabel("% of Sentences", color='white', fontsize=12)
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', colors='white')
    ax1.set_ylim(0, 100)
    
    # Add a legend
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles, labels, loc="upper right", title="Level", frameon=False, 
             title_fontsize=12, fontsize=10, labelcolor='white')
    plt.setp(ax1.get_legend().get_title(), color='white')
    
    # Remove grid and spines
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    fig1.savefig(breakdown_img, format='png', bbox_inches='tight')
    breakdown_img.seek(0)
    
    # Generate profile chart (line chart)
    profile_img = io.BytesIO()
    fig2, ax2 = plt.subplots(figsize=(10, 6), dpi=100)
    fig2.patch.set_facecolor('#1E1E1E')  # Dark background
    ax2.set_facecolor('#1E1E1E')
    
    # Create a profile dataset that shows % of text in each complexity category
    profile_data = []
    for source in sources:
        source_data = category_df[category_df["source"] == source]
        if len(source_data) == 0:
            continue
            
        cat = source_data["category"].iloc[0]
        for category in categories:
            percentage = 100 if category == cat else 0
            if category == cat:
                profile_data.append({
                    "source": source,
                    "category": category,
                    "percentage": percentage
                })
    
    profile_df = pd.DataFrame(profile_data)
    
    # Create a pivot table for easier plotting
    if not profile_df.empty:
        categories_encoded = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
        profile_df["category_code"] = profile_df["category"].map(categories_encoded)
        
        # Plot lines for each source
        for i, source in enumerate(sources):
            source_data = profile_df[profile_df["source"] == source]
            if len(source_data) == 0:
                continue
                
            color = f"C{i}"
            marker = "o"
            if source == "Original":
                color = "#6495ED"  # Light blue for original
                marker = "s"  # Square marker
                
            # For each source, plot points at each category level
            for category in categories:
                category_code = categories_encoded[category]
                if category == source_data["category"].iloc[0]:
                    ax2.scatter([category_code], [100], color=color, marker=marker, s=80, 
                              label=source if category == source_data["category"].iloc[0] else None)
                else:
                    ax2.scatter([category_code], [0], color=color, marker=marker, s=80)
                    
            # Connect the points with lines
            category_codes = [0, 1, 2]  # Beginner, Intermediate, Advanced
            values = [100 if categories[code] == source_data["category"].iloc[0] else 0 for code in range(3)]
            ax2.plot(category_codes, values, color=color, linestyle='-', linewidth=2)
    
    # Customize the chart
    ax2.set_title("Complexity Profile", color='white', fontsize=14, pad=20)
    ax2.set_xlabel("Complexity Level", color='white', fontsize=12)
    ax2.set_ylabel("% of Sentences", color='white', fontsize=12)
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(categories)
    ax2.tick_params(axis='x', colors='white')
    ax2.tick_params(axis='y', colors='white')
    ax2.set_ylim(0, 100)
    
    # Add a legend
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, labels, loc="upper right", title="Source", frameon=False, 
             title_fontsize=12, fontsize=10, labelcolor='white')
    plt.setp(ax2.get_legend().get_title(), color='white')
    
    # Remove grid and spines
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    fig2.savefig(profile_img, format='png', bbox_inches='tight')
    profile_img.seek(0)
    
    # Convert images to base64
    breakdown_base64 = base64.b64encode(breakdown_img.read()).decode('utf-8')
    profile_base64 = base64.b64encode(profile_img.read()).decode('utf-8')
    
    return {
        'breakdown': breakdown_base64,
        'profile': profile_base64
    }
