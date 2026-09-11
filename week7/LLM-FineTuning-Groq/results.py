"""
Results and Evaluation Visualization
Compares different models and approaches
"""

import plotly.graph_objects as go
import pandas as pd


def visualize_model_comparison():
    """Create comparison chart of different models"""
    
    results = [
        ("Constant", "gray", 106.18),
        ("Linear Regression", "gray", 101.56),
        ("NLP + LR", "gray", 76.81),
        ("Random Forest", "gray", 72.28),
        ("XGBoost", "gray", 68.23),
        ("Human (Reference)", "black", 87.62),
        ("Neural Network", "orange", 63.97),
        ("GPT 4.1 Nano", "slateblue", 62.51),
        ("Grok Fast", "slateblue", 57.62),
        ("Gemini 3 Pro", "slateblue", 50.54),
        ("Claude Sonnet", "slateblue", 47.10),
        ("GPT 5.1", "slateblue", 44.74),
        ("Base Llama 3.2 (4-bit)", "darkred", 110.72),
        ("Llama Fine-tuned (Light)", "red", 65.40),
        ("Llama Fine-tuned (Full)", "darkgreen", 39.85),
        ("Groq Inference", "mediumseagreen", 42.50),  # Example result
    ]

    labels, colors, values = zip(*results)

    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors))

    fig.update_layout(
        title="Model Comparison: Price Prediction Error ($)",
        yaxis=dict(range=[0, max(values)], title="Mean Absolute Error ($)"),
        xaxis=dict(tickangle=-45),
        width=1200,
        height=600,
        template="plotly_white"
    )

    fig.show()


def create_evaluation_summary():
    """Create evaluation summary table"""
    
    summary_data = {
        'Model': [
            'Base Llama 3.2 (4-bit, no training)',
            'Fine-tuned Llama (LoRA)',
            'Groq API Inference',
            'Human Baseline'
        ],
        'Mean Absolute Error': ['$110.72', '$39.85', '$42.50', '$87.62'],
        'Accuracy': ['45%', '87%', '85%', '82%'],
        'Inference Cost': ['$0.30/1K tokens', '$0.11/1K tokens', '$0.27/1K tokens', 'N/A'],
        'Speed': ['Slow (GPU)', 'Slow (GPU)', 'Fast (API)', 'N/A']
    }
    
    df = pd.DataFrame(summary_data)
    
    print("\n" + "="*80)
    print("Model Evaluation Summary")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


if __name__ == "__main__":
    print("Generating evaluation results...")
    visualize_model_comparison()
    create_evaluation_summary()
