import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

files = {
    'gemma': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_ic_results_5runs.csv',
    'qwen': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_ic_results_5runs.csv',
    'llama': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_ic_results_5runs.csv',
}

for model_name, path in files.items():
    df = pd.read_csv(path)
    df = df.dropna(subset=['predicted'])
    df['predicted_clean'] = df['predicted'].str.strip().str.capitalize()
    df = df[df['predicted_clean'].isin(['Positive', 'Negative', 'Neutral'])]
    df['predicted_numeric'] = df['predicted_clean'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1})

    # Build sentiment matrix — rows = templates, columns = politicians
    # Average across runs per template-politician combo first
    avg_df = df.groupby(['template_id', 'entity_id', 'entity_name'])['predicted_numeric'].mean().reset_index()
    pivot = avg_df.pivot(index='template_id', columns='entity_name', values='predicted_numeric').fillna(0)

    # Compute cosine similarity between politicians (transpose so politicians are rows)
    politicians = pivot.columns.tolist()
    matrix = cosine_similarity(pivot.T)

    # Convert to percentage like the paper (scale to 100)
    matrix_pct = np.round(matrix * 100, 1)

    sim_df = pd.DataFrame(matrix_pct, index=politicians, columns=politicians)

    print(f"\n{'='*60}")
    print(f"Similarity Matrix — {model_name.upper()}")
    print('='*60)
    print(sim_df.to_string())

    sim_df.to_csv(
        rf'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\similarity_matrix_{model_name}.csv'
    )

print("\nDone! Saved similarity matrices for all models.")