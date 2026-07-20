import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# ── CONFIG ─────────────────────────────────────────────────────────────────────
IDENTITIES_PATH = r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\entity.csv.csv'
files = {
    'Gemma 2 2B': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\gemma_ic_results.csv',
    'Qwen 2.5 3B': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\qwen_ic_results.csv',
    'Llama 3.2 3B': r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_ic_results.csv',
}

# ── DATA PROCESSING ──────────────────────────────────────────────────────────
identities = pd.read_csv(IDENTITIES_PATH)
identities.columns = identities.columns.str.strip().str.lower()

def get_model_data(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    # Ensure mapping handles 'Positive'/'Negative' consistently
    df['predicted_numeric'] = df['predicted'].str.strip().str.capitalize().map(
        {'Positive': 1, 'Neutral': 0, 'Negative': -1}
    )
    return df.groupby(['entity_name', 'order_axis', 'social_axis', 'temporal_axis'])['predicted_numeric'].mean().reset_index()

model_data = {model: get_model_data(path) for model, path in files.items()}

# ── PLOTTING FUNCTION ─────────────────────────────────────────────────────────
def plot_compass(data_dict, x_col, y_col, x_label, y_label, filename):
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    norm = Normalize(vmin=-0.5, vmax=0.5)
    cmap = plt.get_cmap('RdYlGn')

    for ax, (model_name, df) in zip(axes, data_dict.items()):
        ax.set_facecolor('#f8f8f8')
        
        # Plot points
        sc = ax.scatter(df[x_col], df[y_col], c=df['predicted_numeric'], 
                        cmap=cmap, norm=norm, s=150, edgecolors='#333333', zorder=3)
        
        # Add labels
        for _, row in df.iterrows():
            ax.text(row[x_col], row[y_col] + 0.1, row['entity_name'], 
                    ha='center', fontsize=7, bbox=dict(boxstyle='round,pad=0.1', facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_title(model_name, fontweight='bold')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True, linestyle='--', alpha=0.6)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label('Avg Sentiment')
    plt.savefig(filename, dpi=180)
    print(f"Saved: {filename}")

# ── GENERATE PLOTS ─────────────────────────────────────────────────────────────
# 1. Order vs Social
plot_compass(model_data, 'order_axis', 'social_axis', 
             'Political Order (1= Extreme Discipline, 5= Extreme Rights)', 
             'Social Orientation (1=Extreme Self, 5= Extreme Community)', 
             'compass_order_social.png')

# 2. Order vs Temporal
plot_compass(model_data, 'order_axis', 'temporal_axis', 
             'Political Order (1=Extreme Discipline, 5=Extreme Rights)', 
             'Temporal Axis (1=Extreme Present, 5=Extreme Future)', 
             'compass_order_temporal.png')

# 3. Social vs Temporal
plot_compass(model_data, 'social_axis', 'temporal_axis', 
             'Social Orientation (1=Extreme Self, 5= Extreme Community)', 
             'Temporal Axis (1=Extreme Present, 5=Extreme Future)', 
             'compass_social_temporal.png')