import pandas as pd
from sklearn.metrics import f1_score, accuracy_score

files = {
    'gemma_without_calamancy_gold': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_without_calamancy_gold.csv',
    'gemma_without_calamancy_stress': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_without_calamancy_stress.csv',
    'qwen_without_calamancy_gold': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_without_calamancy_gold.csv',
    'qwen_without_calamancy_stress': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_without_calamancy_stress.csv',
    'llama_without_calamancy_gold': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_without_calamancy_gold.csv',
    'llama_without_calamancy_stress': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_without_calamancy_stress.csv',
    'gemma_with_calamancy_gold': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_with_calamancy_gold_entities.csv',
    'gemma_with_calamancy_stress': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_with_calamancy_stress_entities.csv',
    'qwen_with_calamancy_gold': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_with_calamancy_gold_entities.csv',
    'qwen_with_calamancy_stress': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_with_calamancy_stress_entities.csv',
    'llama_with_calamancy_gold': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_with_calamancy_gold_entities.csv',
    'llama_with_calamancy_stress': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_with_calamancy_stress_entities.csv',
}

summary = []

for name, path in files.items():
    try:
        df = pd.read_csv(path)

        if 'true_label' in df.columns:
            true_col = 'true_label'
        elif 'Final_Label' in df.columns:
            true_col = 'Final_Label'
        else:
            print(f"No label column found in {name}, columns are: {df.columns.tolist()}")
            continue

        df = df.dropna(subset=[true_col, 'predicted'])

        true = df[true_col].astype(str).str.strip().str.capitalize()
        pred = df['predicted'].astype(str).str.strip().str.capitalize()

        valid = pred.isin(['Positive', 'Negative', 'Neutral'])
        true = true[valid]
        pred = pred[valid]

        f1 = f1_score(true, pred, average='weighted', labels=['Positive', 'Negative', 'Neutral'], zero_division=0)
        acc = accuracy_score(true, pred)

        summary.append({
            'experiment': name,
            'weighted_f1': round(f1, 4),
            'accuracy': round(acc, 4),
            'valid_rows': len(true),
            'skipped_rows': len(df) - len(true)
        })

        print(f"{name}: F1={round(f1,4)} | Accuracy={round(acc,4)}")

    except Exception as e:
        print(f"Error with {name}: {e}")

results_df = pd.DataFrame(summary)
results_df.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\evaluation_summary.csv', index=False)
print("\nDone! Saved to evaluation_summary.csv")