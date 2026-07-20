import pandas as pd
import numpy as np
from scipy.stats import entropy

def compute_ic(predictions):
    counts = predictions.value_counts(normalize=True)
    return entropy(counts)

files = {
    'gemma': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_ic_results_5runs.csv',
    'qwen': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_ic_results_5runs.csv',
    'llama': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_ic_results_5runs.csv',
}

all_results = []

for model_name, path in files.items():
    df = pd.read_csv(path)
    df = df.dropna(subset=['predicted'])
    df['predicted_clean'] = df['predicted'].str.strip().str.capitalize()
    df = df[df['predicted_clean'].isin(['Positive', 'Negative', 'Neutral'])]
    df['predicted_numeric'] = df['predicted_clean'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

    for entity_id in df['entity_id'].unique():
        entity_df = df[df['entity_id'] == entity_id]
        entity_name = entity_df['entity_name'].iloc[0]
        order_axis = entity_df['order_axis'].iloc[0]
        social_axis = entity_df['social_axis'].iloc[0]
        temporal_axis = entity_df['temporal_axis'].iloc[0]

        avg_polarity = round(entity_df['predicted_numeric'].mean(), 4)

        # IC score per politician across all templates
        ic_scores = []
        for template_id in entity_df['template_id'].unique():
            template_df = entity_df[entity_df['template_id'] == template_id]
            if len(template_df) > 1:
                ic_scores.append(compute_ic(template_df['predicted_clean']))

        avg_ic = round(np.mean(ic_scores), 4) if ic_scores else None

        all_results.append({
            'model': model_name,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'order_axis': order_axis,
            'social_axis': social_axis,
            'temporal_axis': temporal_axis,
            'avg_polarity': avg_polarity,
            'avg_ic_score': avg_ic,
            'total_predictions': len(entity_df)
        })

        print(f"{model_name} | {entity_name} | Polarity: {avg_polarity} | IC: {avg_ic}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\ic_per_politician5runs.csv', index=False)
print("\nDone! Saved to ic_per_politician5runs.csv")