"""
Mann-Whitney U Tests for IC Score and Polarity Comparisons
===========================================================
Light theme version — white backgrounds, dark text.
"""

import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, entropy
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

DEMO_MODE = True

IC_FILES = {
    'Gemma 2 2B':   r'/mnt/user-data/uploads/similarity_matrix_gemma.csv',
    'Qwen 2.5 3B':  r'/mnt/user-data/uploads/similarity_matrix_qwen.csv',
    'Llama 3.2 3B': r'/mnt/user-data/uploads/similarity_matrix_llama.csv',
}

# ── Theme constants ───────────────────────────────────────────────────────────
BG        = 'white'
PANEL_BG  = '#f8fafc'
TEXT      = '#0f172a'
SUBTEXT   = '#475569'
ACCENT    = '#1d4ed8'
GRID_LINE = '#e2e8f0'

# ── Demo data ─────────────────────────────────────────────────────────────────
if DEMO_MODE:
    rng = np.random.default_rng(42)
    politicians = [
        'Alan Peter Cayetano', 'Antonio Trillanes', 'Benigno Aquino III',
        'Chiz Escudero', 'Grace Poe', 'Jejomar Binay',
        'Juan Ponce Enrile', 'Leni Robredo', 'Mar Roxas',
        'Miriam Defensor Santiago', 'Rodrigo Duterte',
    ]
    axis_scores = {
        'Alan Peter Cayetano':      (2, 2, 3),
        'Antonio Trillanes':        (2, 3, 4),
        'Benigno Aquino III':       (3, 4, 4),
        'Chiz Escudero':            (3, 3, 3),
        'Grace Poe':                (3, 4, 3),
        'Jejomar Binay':            (2, 2, 2),
        'Juan Ponce Enrile':        (1, 1, 1),
        'Leni Robredo':             (4, 5, 5),
        'Mar Roxas':                (4, 4, 4),
        'Miriam Defensor Santiago': (3, 3, 3),
        'Rodrigo Duterte':          (1, 1, 2),
    }
    bias_prior = {
        'Gemma 2 2B':   {p: 0.0  for p in politicians},
        'Qwen 2.5 3B':  {p: -0.3 for p in politicians},
        'Llama 3.2 3B': {p: (0.4 if axis_scores[p][0] >= 4 else -0.4)
                          for p in politicians},
    }
    templates  = [f'T{i:03d}' for i in range(1, 51)]
    sentiments = ['Positive', 'Neutral', 'Negative']
    frames = {}
    for model, priors in bias_prior.items():
        rows = []
        for t in templates:
            for p in politicians:
                prior = priors[p]
                probs = np.array([0.33 + prior*0.4, 0.33, 0.33 - prior*0.4])
                probs = np.clip(probs, 0.05, 0.95)
                probs /= probs.sum()
                pred = rng.choice(sentiments, p=probs)
                o, s, tm = axis_scores[p]
                rows.append({'template_id': t, 'entity_id': politicians.index(p),
                             'official_name': p, 'order_axis': o,
                             'social_axis': s, 'temporal_axis': tm,
                             'predicted': pred})
        frames[model] = pd.DataFrame(rows)
    IC_DATA = frames
else:
    IC_DATA = {m: pd.read_csv(p) for m, p in IC_FILES.items()}

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — COMPUTE IC SCORES
# ══════════════════════════════════════════════════════════════════════════════

def compute_ic_per_template(df):
    def _ic(preds):
        counts = preds.value_counts(normalize=True)
        return entropy(counts, base=np.e)
    return df.groupby('template_id')['predicted'].apply(_ic)

ic_per_template = {}
for model, df in IC_DATA.items():
    df['predicted'] = df['predicted'].str.strip().str.capitalize()
    df = df[df['predicted'].isin(['Positive', 'Neutral', 'Negative'])]
    IC_DATA[model] = df
    ic_per_template[model] = compute_ic_per_template(df)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3A — MODEL vs MODEL
# ══════════════════════════════════════════════════════════════════════════════

models = list(IC_DATA.keys())
mw_model_pvals = pd.DataFrame(np.nan, index=models, columns=models)

for m1 in models:
    for m2 in models:
        if m1 == m2:
            continue
        x = ic_per_template[m1].values
        y = ic_per_template[m2].values
        stat, p = mannwhitneyu(x, y, alternative='two-sided')
        mw_model_pvals.loc[m1, m2] = round(p, 4)

print("\n═══ Part A: Model vs Model ═══")
print(mw_model_pvals.to_string())

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3B — AXIS LEVEL PAIRS
# ══════════════════════════════════════════════════════════════════════════════

num_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}

AXES = {
    'Political Order':      'order_axis',
    'Social Orientation':   'social_axis',
    'Temporal Orientation': 'temporal_axis',
}

all_pval_tables = {}

for model, df in IC_DATA.items():
    df = df.copy()
    df['pred_num'] = df['predicted'].map(num_map)
    for ax_label, ax_col in AXES.items():
        levels   = sorted(df[ax_col].unique())
        pval_mat = pd.DataFrame(np.nan, index=levels, columns=levels)
        for li in levels:
            for lj in levels:
                if li == lj:
                    continue
                xi = df[df[ax_col] == li]['pred_num'].values
                xj = df[df[ax_col] == lj]['pred_num'].values
                if len(xi) < 2 or len(xj) < 2:
                    continue
                stat, p = mannwhitneyu(xi, xj, alternative='two-sided')
                pval_mat.loc[li, lj] = round(p, 4)
        all_pval_tables[(model, ax_label)] = pval_mat

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — HEATMAP HELPER  (light theme)
# ══════════════════════════════════════════════════════════════════════════════

def pval_heatmap(ax, mat, title, row_labels, col_labels, sig_thresh=0.05):
    n_r, n_c = len(row_labels), len(col_labels)
    arr      = mat.values.astype(float)

    # Green (significant) → white → red (not significant)
    cmap = plt.cm.RdYlGn   # green=low p, red=high p

    for r in range(n_r):
        for c in range(n_c):
            val = arr[r, c]

            if np.isnan(val):
                cell_color = GRID_LINE
            else:
                # invert: low p → green end of RdYlGn
                norm = 1.0 - np.clip(val, 0, 1)
                cell_color = cmap(norm)

            ax.add_patch(plt.Rectangle([c-0.5, r-0.5], 1, 1,
                                        color=cell_color, zorder=0))

            if not np.isnan(val):
                # Dark text on light cells, light text on very saturated cells
                brightness = 0.299*cell_color[0] + 0.587*cell_color[1] + 0.114*cell_color[2]
                txt_color  = TEXT if brightness > 0.45 else 'white'
                ax.text(c, r, f'{val:.4f}', ha='center', va='center',
                        fontsize=8.5, color=txt_color,
                        fontweight='bold' if val < sig_thresh else 'normal',
                        fontfamily='monospace')
            else:
                ax.text(c, r, '—', ha='center', va='center',
                        fontsize=10, color=SUBTEXT, fontfamily='monospace')

    ax.set_xlim(-0.5, n_c-0.5)
    ax.set_ylim(n_r-0.5, -0.5)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, rotation=35, ha='right',
                       fontsize=8.5, color=TEXT)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_labels, fontsize=8.5, color=TEXT)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight='bold', pad=7)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID_LINE)
        spine.set_linewidth(0.8)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — FIGURE A: model vs model
# ══════════════════════════════════════════════════════════════════════════════

fig_a, ax_a = plt.subplots(figsize=(7, 5))
fig_a.patch.set_facecolor(BG)
ax_a.set_facecolor(PANEL_BG)

pval_heatmap(ax_a, mw_model_pvals,
             'Model vs. Model — IC Score Significance (Mann-Whitney U, two-sided)',
             models, models)

fig_a.text(0.5, 0.01,
    'Green / bold = p < 0.05 (significant difference in inconsistency between models)',
    ha='center', fontsize=8, color=SUBTEXT)

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\mw_model_comparison_light.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: mw_model_comparison_light.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — FIGURE B: 3 axes × 3 models
# ══════════════════════════════════════════════════════════════════════════════

ax_labels_list = list(AXES.keys())
fig_b, axes_b  = plt.subplots(3, 3, figsize=(20, 16))
fig_b.patch.set_facecolor(BG)
fig_b.suptitle(
    'Mann-Whitney p-values: Polarity across Contreras Axis Levels\n'
    'Row = reference group  |  Col = comparison group  '
    '|  Green / bold = p < 0.05',
    color=TEXT, fontsize=13, fontweight='bold', y=0.99)

for col_i, model in enumerate(models):
    for row_i, ax_label in enumerate(ax_labels_list):
        ax   = axes_b[row_i, col_i]
        pmat = all_pval_tables[(model, ax_label)]
        lvls = list(pmat.index)

        if ax_label == 'Political Order':
            desc = {1:'Extreme\nDiscipline', 2:'Lean\nDiscipline',
                    3:'Moderate',            4:'Lean\nRights',
                    5:'Extreme\nRights'}
        elif ax_label == 'Social Orientation':
            desc = {1:'Extreme\nSelf',  2:'Lean\nSelf',
                    3:'Neutral',         4:'Lean\nComm.',
                    5:'Extreme\nComm.'}
        else:
            desc = {1:'Traditionalist', 2:'Lean Trad.',
                    3:'Moderate',        4:'Lean Prog.',
                    5:'Progressive'}

        lvl_labels = [desc.get(l, str(l)) for l in lvls]
        pval_heatmap(ax, pmat, f'{model}\n{ax_label}', lvl_labels, lvl_labels)

        if col_i == 0:
            ax.set_ylabel(ax_label, color=ACCENT, fontsize=9,
                          fontweight='bold')

# Light separator lines between panels
for ax_row in axes_b:
    for ax in ax_row:
        ax.set_facecolor(PANEL_BG)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\mw_axis_pvalues_light.png',
            dpi=180, bbox_inches='tight', facecolor=BG)
plt.close()
print("Saved: mw_axis_pvalues_light.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — EXPORT CSV
# ══════════════════════════════════════════════════════════════════════════════

rows = []
for (model, ax_label), pmat in all_pval_tables.items():
    for li in pmat.index:
        for lj in pmat.columns:
            val = pmat.loc[li, lj]
            rows.append({
                'model': model, 'axis': ax_label,
                'level_i': li, 'level_j': lj, 'p_value': val,
                'significant_p05': 'YES' if (not np.isnan(val) and val < 0.05) else 'NO',
            })
pd.DataFrame(rows).to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\mann_whitney_all_pvalues.csv', index=False)
print("Saved: mann_whitney_all_pvalues.csv")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n═══ SUMMARY ═══")
print("\nModel IC comparison (Part A):")
for m1 in models:
    for m2 in models:
        if m1 >= m2:
            continue
        p   = mw_model_pvals.loc[m1, m2]
        sig = "SIGNIFICANT" if p < 0.05 else "not significant"
        print(f"  {m1} vs {m2}: p = {p:.4f}  ->  {sig}")

print("\nPolarity axis comparisons (Part B) — significant pairs (p < 0.05):")
for (model, ax_label), pmat in all_pval_tables.items():
    sigs  = [f"Level {li} vs {lj} (p={pmat.loc[li,lj]:.4f})"
             for li in pmat.index for lj in pmat.columns
             if not np.isnan(pmat.loc[li,lj]) and pmat.loc[li,lj] < 0.05]
    label = f"\n  {model} | {ax_label}:"
    print(label + (" no significant pairs" if not sigs else ""))
    for s in sigs:
        print(f"    {s}")

print("\nAll done.")