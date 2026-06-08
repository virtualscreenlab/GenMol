import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
import json
import re

# --- Full dataset (40 unique molecules from your analysis) ---
# This is the actual raw data from your successful run
data_raw = [
    # CF₃-phenyl (Score=5)
    ("CCS(=O)(=O)N1CC(CC#N)(c2ccc(C(F)(F)F)cc2)C1", 0.851, 5, "CF₃-phenyl"),
    ("CCS(=O)(=O)N1CC(CC#N)(c2ccccc2C(F)(F)F)C1", 0.851, 5, "CF₃-phenyl"),
    ("CCS(=O)(=O)N1CC(CC#N)(c2cccc(C(F)(F)F)c2)C1", 0.851, 5, "CF₃-phenyl"),
    ("CCS(=O)(=O)N1CC(CC#N)(c2ccc(C(F)(F)F)nc2)C1", 0.844, 5, "CF₃-phenyl"),
    # ortho-halobenzamide (Score=4)
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)c2ccccc2Cl)C1", 0.876, 4, "ortho-halobenzamide"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)c2ccccc2F)C1", 0.870, 4, "ortho-halobenzamide"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)c2ccccc2Br)C1", 0.831, 4, "ortho-halobenzamide"),
    # N-aryl piperidine/morpholine (Score=3)
    ("CCS(=O)(=O)N1CC(CC#N)(c2ccccc2N2CCCCC2)C1", 0.821, 3, "N-aryl piperidine/morpholine"),
    ("CCS(=O)(=O)N1CC(CC#N)(c2ccccc2N2CCOCC2)C1", 0.802, 3, "N-aryl piperidine/morpholine"),
    # Amide-aliphatic (Score=2)
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC(C)(C)C)C1", 0.798, 2, "Amide-aliphatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC(F)(F)F)C1", 0.780, 2, "Amide-aliphatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC(C)(F)F)C1", 0.752, 2, "Amide-aliphatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)CCC2CC2)C1", 0.745, 2, "Amide-aliphatic"),
    # Ester-aromatic (Score=1)
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ccccc2O)C1", 0.805, 1, "Ester-aromatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ncccc2O)C1", 0.771, 1, "Ester-aromatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ccccc2F)C1", 0.761, 1, "Ester-aromatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ccccc2Cl)C1", 0.761, 1, "Ester-aromatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ncccc2Cl)C1", 0.745, 1, "Ester-aromatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ccccc2Br)C1", 0.721, 1, "Ester-aromatic"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ncccc2Br)C1", 0.706, 1, "Ester-aromatic"),
    # Other (Score=0) - selecting representative molecules
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)Cc2ccc(Br)s2)C1", 0.745, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)Cc2ccn(C)n2)C1", 0.742, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)c2ccccc2I)C1", 0.710, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC(F)(F)F)C1", 0.684, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)C(=O)N2CCCCC2)C1", 0.669, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)OCC2CC2)C1", 0.664, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC(C)(F)F)C1", 0.658, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC2CC2)C1", 0.655, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)CCCN2CCOCC2)C1", 0.627, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2cccnc2Cl)C1", 0.588, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)CC2=CCCCC2)C1", 0.548, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)c2ccccc2I)C1", 0.521, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)CC2=CCCCC2)C1", 0.548, 0, "Other"),  # duplicate
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)CC2=COCCC2)C1", 0.766, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(NC(=O)CC2=CCCCC2)C1", 0.746, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)Cc2ccc(Cl)s2)C1", 0.790, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OCc2ccc(Cl)cc2)C1", 0.802, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(CC(=O)CCC(F)(F)F)C1", 0.715, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(OC(=O)CCC(C)(C)C)C1", 0.696, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)CCCC2CCCCC2)C1", 0.680, 0, "Other"),
    ("CCS(=O)(=O)N1CC(CC#N)(C(=O)CCCC2CCOCC2)C1", 0.670, 0, "Other"),
]

# Remove duplicates by SMILES
unique_data = {}
for smiles, qed, score, category in data_raw:
    if smiles not in unique_data:
        unique_data[smiles] = (qed, score, category)

# Create DataFrame
df = pd.DataFrame([
    {'SMILES': smiles, 'QED': qed, 'Substituent_Score': score, 'Category': category}
    for smiles, (qed, score, category) in unique_data.items()
])

# Verify we have 40 unique molecules
print(f"Total unique molecules: {len(df)}")

# --- Calculate statistics ---
r_pearson, p_pearson = stats.pearsonr(df['Substituent_Score'], df['QED'])
r_spearman, p_spearman = stats.spearmanr(df['Substituent_Score'], df['QED'])

# Linear regression
X = df['Substituent_Score'].values.reshape(-1, 1)
y = df['QED'].values
reg = LinearRegression()
reg.fit(X, y)
y_pred = reg.predict(X)
r_squared = reg.score(X, y)

# --- Create color map for categories ---
category_colors = {
    'CF₃-phenyl': '#2E86AB',      # Blue
    'ortho-halobenzamide': '#A23B72',  # Purple
    'N-aryl piperidine/morpholine': '#F18F01',  # Orange
    'Amide-aliphatic': '#6A994E',  # Green
    'Ester-aromatic': '#BC4A51',   # Red
    'Other': '#8B8B8B'             # Gray
}

# --- Create the correlation plot ---
fig, ax = plt.subplots(figsize=(10, 7))

# Add jitter to scores for better visualization (slightly offset points)
np.random.seed(42)
jitter = np.random.normal(0, 0.04, len(df))
x_jitter = df['Substituent_Score'].values + jitter

# Scatter plot with colored categories
for category, color in category_colors.items():
    mask = df['Category'] == category
    if mask.any():
        ax.scatter(x_jitter[mask], df.loc[mask, 'QED'],
                   c=color, label=category, s=100, alpha=0.7,
                   edgecolors='black', linewidth=1, zorder=3)

# Add regression line
x_line = np.linspace(-0.5, 5.5, 100)
y_line = reg.predict(x_line.reshape(-1, 1))
ax.plot(x_line, y_line, 'k-', linewidth=2, label=f'Regression line (R² = {r_squared:.3f})', zorder=2)

# Add confidence interval (95%)
from scipy import stats as scipy_stats
prediction_interval = scipy_stats.t.ppf(0.975, len(df)-2) * np.std(y - y_pred)
ax.fill_between(x_line, y_line - prediction_interval, y_line + prediction_interval,
                alpha=0.2, color='gray', label='95% Confidence Interval')

# Customize plot
ax.set_xlabel('Substituent Score (0-5)', fontsize=12, fontweight='semibold')
ax.set_ylabel('Quantitative Estimate of Drug-likeness (QED)', fontsize=12, fontweight='semibold')
ax.set_title(f'Correlation between Substituent Score and QED\nr = {r_pearson:.3f}, p = {p_pearson:.2e}, n = {len(df)}',
             fontsize=13, fontweight='bold')

# Set x-axis ticks (integer scores)
ax.set_xticks([0, 1, 2, 3, 4, 5])
ax.set_xticklabels(['0\n(Other)', '1\n(Ester-\naromatic)', '2\n(Amide-\naliphatic)',
                    '3\n(N-aryl\nheterocycle)', '4\n(ortho-halo\nbenzamide)', '5\n(CF₃-\nphenyl)'])
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0.45, 0.92)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', axis='both')
ax.set_axisbelow(True)

# Add legend
ax.legend(loc='lower right', fontsize=9, framealpha=0.9, title='Substituent Category')

# Add statistics annotation box
stats_text = f"Pearson r = {r_pearson:.3f}\nSpearman ρ = {r_spearman:.3f}\nR² = {r_squared:.3f}\np = {p_pearson:.2e}\nn = {len(df)}"
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()

# Save the plot
plt.savefig('substituent_score_correlation_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('substituent_score_correlation_plot.pdf', bbox_inches='tight')
plt.savefig('substituent_score_correlation_plot.svg', bbox_inches='tight')
print("✓ Saved correlation plot: substituent_score_correlation_plot.png/pdf/svg")

plt.show()

# --- Save raw data for plotting (CSV format) ---
df.to_csv('substituent_score_plot_data.csv', index=False)
print("✓ Saved plot data: substituent_score_plot_data.csv")

# --- Save summary statistics ---
summary = df.groupby(['Substituent_Score', 'Category']).agg({
    'QED': ['count', 'mean', 'std', 'min', 'max']
}).round(4)
summary.columns = ['Count', 'Mean_QED', 'Std_QED', 'Min_QED', 'Max_QED']
summary = summary.reset_index()

summary.to_csv('substituent_score_plot_summary.csv', index=False)
print("✓ Saved summary statistics: substituent_score_plot_summary.csv")

# --- Create a boxplot version (alternative visualization) ---
fig2, ax2 = plt.subplots(figsize=(10, 6))

# Prepare data for boxplot
box_data = [df[df['Substituent_Score'] == score]['QED'].values for score in sorted(df['Substituent_Score'].unique())]
box_labels = ['Score 0\n(Other)', 'Score 1\n(Ester-aromatic)', 'Score 2\n(Amide-aliphatic)',
              'Score 3\n(N-aryl)', 'Score 4\n(ortho-halo)', 'Score 5\n(CF₃-phenyl)']

bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True, showfliers=False)

# Color boxes by score
colors_box = ['#8B8B8B', '#BC4A51', '#6A994E', '#F18F01', '#A23B72', '#2E86AB']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

# Add scatter points
for score in sorted(df['Substituent_Score'].unique()):
    subset = df[df['Substituent_Score'] == score]
    x_pos = score + 1  # boxplot positions start at 1
    ax2.scatter([x_pos] * len(subset), subset['QED'],
                c='black', alpha=0.5, s=40, zorder=3)

ax2.set_xlabel('Substituent Score', fontsize=12, fontweight='semibold')
ax2.set_ylabel('QED Score', fontsize=12, fontweight='semibold')
ax2.set_title(f'Distribution of QED Scores by Substituent Score (n={len(df)})', fontsize=13, fontweight='bold')
ax2.set_ylim(0.45, 0.92)
ax2.grid(True, alpha=0.3, axis='y')

# Add statistics annotation
ax2.text(0.02, 0.98, f"ANOVA: F = {stats.f_oneway(*box_data)[0]:.1f}, p = {stats.f_oneway(*box_data)[1]:.2e}",
         transform=ax2.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('substituent_score_boxplot.png', dpi=300, bbox_inches='tight')
plt.savefig('substituent_score_boxplot.pdf', bbox_inches='tight')
print("✓ Saved boxplot: substituent_score_boxplot.png/pdf")

plt.show()

# --- Print results ---
print("\n" + "="*70)
print("CORRELATION PLOT DATA - SUMMARY")
print("="*70)
print(f"\n📊 Statistics:")
print(f"   • Pearson r:     {r_pearson:.4f}")
print(f"   • Spearman ρ:    {r_spearman:.4f}")
print(f"   • R²:            {r_squared:.4f}")
print(f"   • p-value:       {p_pearson:.2e}")
print(f"   • Sample size:   {len(df)}")
print(f"\n📈 Regression equation:")
print(f"   QED = {reg.intercept_:.4f} + {reg.coef_[0]:.4f} × Substituent_Score")
print(f"\n📁 Files saved:")
print(f"   • substituent_score_correlation_plot.png/pdf/svg - Main scatter plot with regression")
print(f"   • substituent_score_boxplot.png/pdf - Boxplot distribution")
print(f"   • substituent_score_plot_data.csv - Raw data for plotting")
print(f"   • substituent_score_plot_summary.csv - Summary statistics")