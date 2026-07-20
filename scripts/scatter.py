import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. LOAD DATA
# Using r"" strings for Windows paths to avoid issues with backslashes
base_path = r"C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results"
eval_df = pd.read_csv(os.path.join(base_path, "evaluation_summary.csv"))
ic_df = pd.read_csv(os.path.join(base_path, "ic_summary.csv"))
nick_df = pd.read_csv(os.path.join(base_path, "ic_nicknames_summary.csv"))

# 2. DATA PREPARATION
# 2a. Extract 'model' from 'experiment' column (e.g., 'gemma_without_calamancy_gold' -> 'gemma')
def extract_model(exp_name):
    return exp_name.split('_')[0]

eval_df['model'] = eval_df['experiment'].apply(extract_model)

# 2b. Add entity types to IC data
ic_df['entity_type'] = 'Official'
nick_df['entity_type'] = 'Nickname'
combined_ic = pd.concat([ic_df, nick_df])

# 2c. Merge datasets on 'model'
# Note: This will create rows for every axis. 
# We average 'avg_ic_score' across axes for a cleaner plot.
combined_ic_avg = combined_ic.groupby(['model', 'entity_type'])['avg_ic_score'].mean().reset_index()
plot_data = pd.merge(eval_df, combined_ic_avg, on='model')

# 3. GENERATE PLOT
plt.figure(figsize=(10, 6))
sns.set_theme(style="whitegrid")

scatter = sns.scatterplot(
    data=plot_data, 
    x='weighted_f1', 
    y='avg_ic_score', 
    hue='model', 
    style='entity_type', # Differentiates Official vs Nickname
    size='entity_type',  # Changes marker size
    palette='viridis',
    sizes=(100, 250),
    edgecolor='black'
)

# 4. FORMATTING
plt.title('Model Reliability: Performance (F1) vs Inconsistency (IC)', fontsize=14, fontweight='bold')
plt.xlabel('Weighted F1 Score (Performance)', fontsize=12)
plt.ylabel('Average Inconsistency (IC) Score', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('reliability_scatter.png', dpi=300)
print("Successfully saved: reliability_scatter.png")
plt.show()