import pandas as pd
import numpy as np
from scipy.stats import entropy

def to_numeric(pred):
    mapping = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    return mapping.get(pred.strip().capitalize(), None)

def compute_ic(predictions):
    counts = predictions.value_counts(normalize=True)
    return entropy(counts)

files = {
    'gemma': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_ic_results.csv',
    'qwen': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_ic_results.csv',
    'llama': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_ic_results.csv',
}

axes = ['order_axis', 'social_axis', 'temporal_axis']
axis_names = {
    'order_axis': 'Political Order',
    'social_axis': 'Social Orientation',
    'temporal_axis': 'Temporal Orientation'
}

all_ic_results = []
all_polarity_results = []

for model_name, path in files.items():
    df = pd.read_csv(path)
    
    # Clean predictions
    df = df.dropna(subset=['predicted'])
    df['predicted_clean'] = df['predicted'].str.strip().str.capitalize()
    df = df[df['predicted_clean'].isin(['Positive', 'Negative', 'Neutral'])]
    df['predicted_numeric'] = df['predicted_clean'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

    for axis in axes:
        axis_label = axis_names[axis]

        ic_rows = []
        for template_id in df['template_id'].unique():
            template_df = df[df['template_id'] == template_id]
            ic_score = compute_ic(template_df['predicted_clean'])
            ic_rows.append({
                'model': model_name,
                'axis': axis_label,
                'template_id': template_id,
                'ic_score': round(ic_score, 4)
            })
        
        avg_ic = round(np.mean([r['ic_score'] for r in ic_rows]), 4)
        all_ic_results.append({
            'model': model_name,
            'axis': axis_label,
            'avg_ic_score': avg_ic
        })

        print(f"{model_name} | {axis_label} | Avg IC Score: {avg_ic}")

        for level in sorted(df[axis].unique()):
            level_df = df[df[axis] == level]
            avg_polarity = round(level_df['predicted_numeric'].mean(), 4)
            all_polarity_results.append({
                'model': model_name,
                'axis': axis_label,
                'axis_level': level,
                'avg_polarity': avg_polarity,
                'count': len(level_df)
            })
            print(f"  Level {level}: Avg Polarity = {avg_polarity}")

ic_df = pd.DataFrame(all_ic_results)
ic_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\ic_summary.csv', index=False)

polarity_df = pd.DataFrame(all_polarity_results)
polarity_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\polarity_summary.csv', index=False)

print("\nDone!")