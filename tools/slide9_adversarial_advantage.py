"""
Slide 9 Adversarial Advantage Graphics Generator
-------------------------------------------------
Run locally: python slide9_adversarial_advantage.py

Produces 4 honest, rigorous PNGs:
  Slide9_1_DataSplit.png
  Slide9_2_FrozenParams.png
  Slide9_3_Efficiency.png
  Slide9_4_Complexity.png

All math is grounded in the v16 physical constraints:
  - N = 69,831 total images
  - Hybrid Neural Update: 10% data (6,983), 43% params (4.85M), 1 epoch
  - Standard DL Update: 100% data (69,831), 100% params (11.18M), 3 epochs
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── THEME ────────────────────────────────────────────────────────────────────
BG      = '#3b3644'
PINK    = '#ff4d6d'
PURPLE  = '#7a6a91'
LITE    = '#a99bbf'
WHITE   = '#ffffff'
DIM     = '#5a5266'
GRID    = '#4a4454'

plt.rcParams.update({
    'axes.facecolor': BG,    'figure.facecolor': BG,
    'axes.edgecolor': WHITE, 'axes.labelcolor':  WHITE,
    'xtick.color':    WHITE, 'ytick.color':      WHITE,
    'text.color':     WHITE, 'grid.color':       GRID,
    'font.family':    'sans-serif', 'font.size': 12,
    'axes.titlesize': 14,    'axes.titleweight': 'bold',
    'savefig.facecolor': BG, 'legend.facecolor': BG,
    'legend.edgecolor': PURPLE,
})
OUT = './'


# ══════════════════════════════════════════════════════════════════════════════
# 1. TRAINING DATA SPLIT
# ══════════════════════════════════════════════════════════════════════════════
def graph_1_data_split():
    TOTAL  = 69831
    NEURAL = 6983
    ML     = TOTAL - NEURAL

    fig, ax = plt.subplots(figsize=(10, 4.5))
    bar_h = 0.45
    y_std = 1.0
    y_hyb = 0.0

    ax.barh(y_std, TOTAL, height=bar_h, color=DIM, zorder=2)
    ax.text(TOTAL / 2, y_std,
            f'All {TOTAL:,} images — backpropagation on 100% of data',
            ha='center', va='center', color=WHITE, fontsize=10, fontweight='bold')

    ax.barh(y_hyb, NEURAL, height=bar_h, color=PINK,   zorder=2)
    ax.barh(y_hyb, ML,     height=bar_h, left=NEURAL,  color=PURPLE, zorder=2)
    
    # Label the small neural slice
    ax.annotate(f'{NEURAL:,}\nimages\n(10%)',
                xy=(NEURAL / 2, y_hyb + bar_h / 2),
                xytext=(NEURAL / 2, y_hyb + bar_h + 0.22),
                ha='center', color=PINK, fontsize=9, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PINK, lw=1.2))
    
    ax.text(NEURAL + ML / 2, y_hyb,
            f'{ML:,} images — Forward pass only (Zero gradient cost)',
            ha='center', va='center', color=WHITE, fontsize=9, fontweight='bold')

    ax.set_yticks([y_hyb, y_std])
    ax.set_yticklabels(['Hybrid Approach', 'Standard Deep Learning'], fontsize=11)
    ax.set_xlabel('Number of Images')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_xlim(0, TOTAL * 1.02)
    ax.set_ylim(-0.3, 1.75)
    ax.set_title('Training Strategy: Selective Gradient Updates', pad=14, color=PINK)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=1)

    plt.tight_layout()
    path = os.path.join(OUT, 'Slide9_1_DataSplit.png')
    plt.savefig(path, bbox_inches='tight', dpi=180)
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 2. FROZEN vs TRAINABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
def graph_2_frozen_params():
    FROZEN    = 6_323_776
    TRAINABLE = 4_853_762
    TOTAL     = FROZEN + TRAINABLE

    f_pct = 100 * FROZEN    / TOTAL
    t_pct = 100 * TRAINABLE / TOTAL

    fig, ax = plt.subplots(figsize=(10, 4.5))
    h = 0.4
    y = 0.0

    ax.barh(y, FROZEN,    height=h, color=PURPLE, zorder=2)
    ax.barh(y, TRAINABLE, height=h, left=FROZEN,  color=PINK, zorder=2)

    ax.annotate(f'Frozen Majority ({f_pct:.1f}%)\n{FROZEN/1e6:.1f}M params (conv1→layer3)\nLocked Permanently',
                xy=(FROZEN / 2, y + h / 2),
                xytext=(FROZEN / 2, y + h + 0.38),
                ha='center', color=PURPLE, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PURPLE, lw=1.5))

    ax.annotate(f'Trainable Head ({t_pct:.1f}%)\n{TRAINABLE/1e6:.1f}M params (layer4+fc)\nOnly 1-Epoch Update',
                xy=(FROZEN + TRAINABLE / 2, y - h / 2),
                xytext=(FROZEN + TRAINABLE / 2, y - h - 0.38),
                ha='center', color=PINK, fontsize=10, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=PINK, lw=1.5))

    ax.set_title(f'Parameter Economy: Freezing the Majority', pad=14, color=PINK)
    ax.set_xlabel('Parameter Count')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1e6:.1f}M'))
    ax.set_xlim(0, TOTAL * 1.02)
    ax.set_ylim(-1.0, 1.1)
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=1)

    plt.tight_layout()
    path = os.path.join(OUT, 'Slide9_2_FrozenParams.png')
    plt.savefig(path, bbox_inches='tight', dpi=180)
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 3. RESOURCE EFFICIENCY (NEW)
# Replacing Caching with an honest comparison of total training compute.
# ══════════════════════════════════════════════════════════════════════════════
def graph_3_efficiency():
    # Relative Energy/Compute Units
    # Standard: 100% data * 100% params * 3 epochs = 300 units
    # Hybrid: 10% data * 43% params * 1 epoch = 4.3 units
    unit_std = 100 * 100 * 3
    unit_hyb = 10 * 43.4 * 1
    
    labels = ['Standard\nDeep Learning', 'Hybrid Approach\n(V16)']
    values = [unit_std, unit_hyb]
    colors = [DIM, PINK]
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values, color=colors, width=0.5, zorder=2)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 5,
                f'{val:,.1f} relative units',
                ha='center', va='bottom', color=WHITE,
                fontsize=11, fontweight='bold')

    # Savings callout
    savings = (1 - (unit_hyb/unit_std)) * 100
    ax.text(1, unit_std*0.5, f'{savings:.1f}% Reduction\nin Compute Budget',
            ha='center', color=PINK, fontsize=12, fontweight='bold',
            bbox=dict(facecolor=BG, edgecolor=PINK, boxstyle='round,pad=0.5', lw=2))

    ax.set_title('Total Training Compute: The Cost of Protection', pad=14, color=PINK)
    ax.set_ylabel('Relative Compute Cost Index')
    ax.set_ylim(0, unit_std * 1.25)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.3, zorder=1)

    plt.tight_layout()
    path = os.path.join(OUT, 'Slide9_3_Efficiency.png')
    plt.savefig(path, bbox_inches='tight', dpi=180)
    print(f"Saved: {path}")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# 4. TIME COMPLEXITY (Honest Comparison)
# Including the Few-Shot Update cost in the Hybrid side.
# ══════════════════════════════════════════════════════════════════════════════
def graph_4_complexity():
    # Setup Parameters
    N = 69831;   N_few = 6983
    P = 11177538; P_train = 4853762
    E_std = 3;   E_few = 1
    
    # Standard DL Cost
    cost_std = N * P * E_std
    
    # Hybrid Stack costs
    # 1. Neural Update (Few-Shot)
    cost_neural_update = N_few * P_train * E_few
    # 2. K-Means
    cost_km  = N * 5 * 300
    # 3. Isolation Forest / Random Forest
    cost_mining = N * np.log2(N) * 200 # 100 trees for each
    
    NORM = 1e9
    
    labels = [
        'Standard DL\nFull Retraining',
        'Hybrid Approach:\nFew-Shot Update',
        'Hybrid Approach:\nSub-population Mining'
    ]
    # We combine IF/KM/RF into "Mining"
    values = np.array([cost_std, cost_neural_update, cost_km + cost_mining]) / NORM
    colors = [DIM, PINK, PURPLE]
    
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(labels, values, color=colors, height=0.5, zorder=2)
    
    for bar, val in zip(bars, values):
        txt = f'{val/1000:.2f}T ops' if val >= 1000 else f'{val:.2f}B ops'
        ax.text(val + max(values) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                txt, va='center', color=WHITE, fontsize=10, fontweight='bold')

    # Real Speedup
    total_hybrid = values[1] + values[2]
    speedup = values[0] / total_hybrid
    
    ax.text(max(values) * 0.5, -0.8,
            f'Total System is ~{speedup:,.0f}× faster than standard re-training',
            ha='center', color=PINK, fontsize=12, fontweight='bold',
            bbox=dict(facecolor=BG, edgecolor=PINK, boxstyle='round,pad=0.5', lw=1.8))

    ax.set_title(f'Operational Multiplier: Scientific Time Complexity', pad=14, color=PINK)
    ax.set_xlabel('Relative Floating Point Operations')
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.0f}T' if x >= 1000 else f'{x:.0f}B'))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.grid(axis='x', linestyle='--', alpha=0.3, zorder=1)
    ax.invert_yaxis()
    ax.set_ylim(top=-1.2)

    plt.tight_layout()
    path = os.path.join(OUT, 'Slide9_4_Complexity.png')
    plt.savefig(path, bbox_inches='tight', dpi=180)
    print(f"Saved: {path}")
    plt.close()


if __name__ == '__main__':
    print('Generating Honest Adversarial Advantage Graphics...\n')
    graph_1_data_split()
    graph_2_frozen_params()
    graph_3_efficiency()
    graph_4_complexity()
    print('\nDone. All 4 assets created.')
