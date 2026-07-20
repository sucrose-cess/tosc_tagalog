import pandas as pd
import numpy as np
from scipy.stats import entropy

def compute_ic(predictions):
    counts = predictions.value_counts(normalize=True)
    return entropy(counts)

files = {
    'gemma': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_ic_nicknames.csv',
    'qwen': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_ic_nicknames.csv',
    'llama': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_ic_nicknames.csv',
}

axes = ['order_axis', 'social_axis', 'temporal_axis']
axis_names = {
    'order_axis': 'Political Order',
    'social_axis': 'Social Orientation',
    'temporal_axis': 'Temporal Orientation'
}

all_ic_results = []
all_polarity_results = []
nickname_comparison = []

for model_name, path in files.items():
    df = pd.read_csv(path)

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
            ic_rows.append(ic_score)

        avg_ic = round(np.mean(ic_rows), 4)
        all_ic_results.append({
            'model': model_name,
            'axis': axis_label,
            'avg_ic_score': avg_ic
        })

        print(f"{model_name} | {axis_label} | Avg IC Score: {avg_ic}")

        # Average polarity per axis level
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

    for entity_id in df['entity_id'].unique():
        entity_df = df[df['entity_id'] == entity_id]
        official_name = entity_df['official_name'].iloc[0]
        for nickname in entity_df['nickname_used'].unique():
            nick_df = entity_df[entity_df['nickname_used'] == nickname]
            avg_polarity = round(nick_df['predicted_numeric'].mean(), 4)
            nickname_comparison.append({
                'model': model_name,
                'entity_id': entity_id,
                'official_name': official_name,
                'nickname': nickname,
                'avg_polarity': avg_polarity
            })

ic_df = pd.DataFrame(all_ic_results)
ic_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\ic_nicknames_summary.csv', index=False)

polarity_df = pd.DataFrame(all_polarity_results)
polarity_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\polarity_nicknames_summary.csv', index=False)

nickname_df = pd.DataFrame(nickname_comparison)
nickname_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\nickname_vs_official_comparison.csv', index=False)

print("\nDone! Saved all nickname IC results!")