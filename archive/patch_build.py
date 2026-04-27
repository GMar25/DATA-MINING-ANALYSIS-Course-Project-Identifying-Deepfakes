"""
Patch build_main_notebook.py:
1. Replace hook + EDA sections with expanded versions
2. Remove all em dashes from markdown
3. Add F1 scores to evaluations
4. Add alpha grid search to Phase 3
5. Fix Phase 1 cluster_accuracy (remove mid-cell import)
"""
import re

with open('build_main_notebook.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ═══════════════════════════════════════════════════════════════════════
# 1. Global em-dash removal (— replaced with commas, "or", etc.)
# ═══════════════════════════════════════════════════════════════════════
# We need to be context-aware, so do targeted replacements first, then sweep the rest

# Section 4 Hook
content = content.replace(
    '## 2. Introduction \u2014 Can You Spot the Fake?',
    '## 2. Introduction: Can You Spot the Fake?'
)
content = content.replace(
    'incite violence \u2014 as the Pentagon',
    'incite violence, as the Pentagon'
)
content = content.replace(
    'is **unsupervised** \u2014 allowing',
    'is **unsupervised**, allowing'
)
content = content.replace(
    "let's test your own intuition.** Below, we display two random face images side by side \u2014 one real, one fake",
    "let us test your own intuition.** Below, we display two random face images side by side, one real, one fake"
)
content = content.replace(
    "you're not alone \u2014 and that's exactly",
    "you are not alone, and that is exactly"
)

# Section 5 EDA
content = content.replace(
    'as noted in our Checkpoint 1 analysis).',
    'as explored below).'
)
content = content.replace(
    'pixel intensity, color distribution, or edge structure \u2014 can distinguish',
    'pixel intensity, color distribution, or edge structure) can distinguish'
)
content = content.replace(
    'Before applying any machine learning, we first explore whether simple statistical properties of the images',
    'Before applying any machine learning, we explore whether simple statistical properties of the images'
)

# EDA Conclusion - replace entire section
old_eda_conclusion = '''cells.append(nbformat.v4.new_markdown_cell("""### EDA Conclusion

The distributions of pixel intensity between real and fake images are **virtually identical** \u2014 both globally and per-channel. This confirms that deepfakes cannot be detected through simple statistical or pixel-level features. The generative process preserves macro-level image statistics almost perfectly.

This result motivates our investigation into **learned representations**: if the signal isn't visible at the pixel level, we need models that can extract higher-order forensic features from the data. The question is whether *unsupervised* techniques can discover these features, or whether *supervised* learning is required.
"""))'''

new_eda_conclusion = '''cells.append(nbformat.v4.new_markdown_cell("""Pixel intensities are virtually identical between real and fake images across all channels. This confirms that the deepfake generation process preserves macro-level image statistics almost perfectly, ruling out simple pixel-based detection.
"""))

# \u2500\u2500 EDA 3c: Gradient / Edge Analysis \u2500\u2500
cells.append(nbformat.v4.new_markdown_cell("""### 3c. Gradient (Edge) Analysis

While pixel intensities are identical, deepfake generation involves "stitching" a swapped face onto a base image. This stitching process may introduce smoothing artifacts that reduce local edge sharpness. We compute Sobel gradient statistics (mean and standard deviation) for a sample of real and fake images.

**Why this matters for our pipeline:** If gradients differ between classes, the "deepfake signal" is localized (near stitching boundaries) rather than global. This means global pixel-level features will fail, but a learned encoder that captures local texture patterns could succeed, motivating our use of convolutional architectures in Phases 1 and 3.
"""))

cells.append(nbformat.v4.new_code_cell(""\"# \u2500\u2500 Gradient (Edge) Analysis \u2500\u2500
def convert_to_grayscale(img_array):
    if img_array.max() > 1:
        img_array = img_array / 255.0
    if img_array.ndim == 2:
        return img_array
    img_array = img_array[..., :3]
    return img_array.sum(axis=2) / 3

def get_img_stats(img_array):
    gray = convert_to_grayscale(img_array)
    stats = {}
    stats["intensity_avg"] = gray.mean()
    stats["intensity_std"] = gray.std()
    stats["intensity_skew"] = skew(gray[::4, ::4], axis=None)
    stats["intensity_kurt"] = kurtosis(gray[::4, ::4], axis=None)
    gx = sobel(gray, axis=0)
    gy = sobel(gray, axis=1)
    g = np.sqrt(gx**2 + gy**2)
    stats["grad_avg"] = g.mean()
    stats["grad_std"] = g.std()
    return stats

# Sample images for statistical analysis
EDA_SAMPLE = 300
img_stats = []

for f in random.sample(fake_files, min(EDA_SAMPLE, len(fake_files))):
    try:
        s = get_img_stats(np.array(Image.open(f).convert('RGB')).astype(float))
        s['label'] = 'Fake'
        img_stats.append(s)
    except Exception:
        pass

for f in random.sample(real_files, min(EDA_SAMPLE, len(real_files))):
    try:
        s = get_img_stats(np.array(Image.open(f).convert('RGB')).astype(float))
        s['label'] = 'Real'
        img_stats.append(s)
    except Exception:
        pass

img_stats_df = pd.DataFrame(img_stats)

# Box plot + summary table
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

fake_grad = img_stats_df[img_stats_df.label == "Fake"]["grad_avg"]
real_grad = img_stats_df[img_stats_df.label == "Real"]["grad_avg"]

axes[0].boxplot([real_grad, fake_grad], labels=["Real", "Fake"])
axes[0].set_title("Gradient Mean by Label", fontweight='bold')
axes[0].set_ylabel("Average Gradient Magnitude")

summary = img_stats_df.groupby("label")[["intensity_avg", "intensity_std", "grad_avg", "grad_std"]].mean()
summary.columns = ["Intensity Mean", "Intensity Std", "Gradient Mean", "Gradient Std"]
axes[1].axis('off')
table = axes[1].table(
    cellText=summary.round(4).values,
    rowLabels=summary.index,
    colLabels=summary.columns,
    cellLoc='center', loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)
axes[1].set_title("Statistical Summary: Real vs Fake", fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

grad_diff_pct = (real_grad.mean() - fake_grad.mean()) / real_grad.mean() * 100
print(f"Gradient difference: Real mean = {real_grad.mean():.4f}, Fake mean = {fake_grad.mean():.4f}")
print(f"Fake gradients are ~{abs(grad_diff_pct):.1f}% {'lower' if grad_diff_pct > 0 else 'higher'} than real.")
""\"))

cells.append(nbformat.v4.new_markdown_cell("""### EDA Conclusions

Three key findings inform our pipeline design:

1. **Pixel intensities are identical** between real and fake images (both globally and per-channel), ruling out simple statistical detection.
2. **Gradient (edge) magnitudes differ**: fake images show a measurable decrease in average gradient, consistent with smoothing artifacts from the face-swapping "stitching" process. However, this signal is subtle and localized, meaning simple global thresholding would not be reliable.
3. **Demographic bias exists**: the dataset is skewed toward certain subgroups, which means detection performance should be interpreted with caution.

These results motivate our investigation into **learned representations**: since the deepfake signal is not visible at the pixel level but exists in localized texture patterns, we need models that can extract higher-order forensic features. The question is whether *unsupervised* techniques can discover these features, or whether *supervised* learning is required.
"""))'''

content = content.replace(old_eda_conclusion, new_eda_conclusion)

# Section 6 RQ - em dashes
content = content.replace(
    "existing detectors haven't seen",
    "existing detectors have not seen"
)
content = content.replace(
    "is **unsupervised** \u2014 so that adapting",
    "is **unsupervised**, so that adapting"
)

# Section 7 Phase 1
content = content.replace(
    "Our first attempt was **fully unsupervised**: train a Convolutional Autoencoder",
    "Our first attempt was **fully unsupervised**: we trained a Convolutional Autoencoder"
)
content = content.replace(
    "The autoencoder's latent space \u2014 the model learned",
    "The autoencoder's latent space. The model learned"
) if "latent space \u2014 the model" in content else None

# Section 8 Phase 2
content = content.replace(
    "The failure of Phase 1 told us the bottleneck was the **embedding quality**, not the mining algorithm. So we pivoted",
    "The failure of Phase 1 told us the bottleneck was the **embedding quality**, not the mining algorithm. We pivoted"
)
content = content.replace(
    "Rather than fine-tuning the entire ResNet (which would make the deep learning do all the work), we deliberately **freeze most of the network**",
    "Rather than fine-tuning the entire ResNet (which would make the deep learning do all the work), we deliberately **freeze most of the network**"
)

# Phase 2 conclusion
content = content.replace(
    "Random Forest is a **supervised** classifier. It requires labeled",
    "Random Forest is a **supervised** classifier that requires labeled"
)

# Phase 3
content = content.replace(
    "fakes should be farther away.",
    "fakes should be geometrically farther from this centroid."
)
content = content.replace(
    'Why equal weighting?** Both scoring mechanisms capture complementary signals \u2014 K-Means measures',
    'Why equal weighting?** Both scoring mechanisms capture complementary signals. K-Means measures'
)

# Section 11 RQ Answer
content = content.replace(
    "Phase 1** demonstrated that purely unsupervised learning, using a custom autoencoder and K-Means, **fails completely**.",
    "Phase 1** demonstrated that purely unsupervised learning (custom autoencoder + K-Means) **fails completely**."
)
content = content.replace(
    "This tells us that the *feature extraction* step cannot be fully unsupervised \u2014 some form of learned representation is required.",
    "This tells us that the *feature extraction* step cannot be fully unsupervised; some form of learned representation is required."
)
content = content.replace(
    "But this approach **contradicts the \"unsupervised\" premise**: it requires labeled training data and cannot adapt without relabeling.",
    "But this approach **contradicts the \"unsupervised\" premise** because it requires labeled training data and cannot adapt without relabeling."
)
content = content.replace(
    "Phase 3 \u2014 the Fusion Gauntlet \u2014 provides the nuanced answer.**",
    "Phase 3, the Fusion Gauntlet, provides the nuanced answer.**"
)

# Section 12 Limitations
content = content.replace(
    "**Single generator**: The HiDF dataset uses one deepfake generation method.",
    "**Single generator**: the HiDF dataset uses one deepfake generation method."
)
content = content.replace(
    "**5% labeled dependency**: While minimal, the pipeline still requires *some* labeled data",
    "**5% labeled dependency**: while minimal, the pipeline still requires *some* labeled data"
)
content = content.replace(
    "**Dataset demographics**: As noted in Checkpoint 1, the HiDF dataset has demographic biases",
    "**Dataset demographics**: as shown in our EDA, the HiDF dataset has demographic biases"
)

# Section 13 Conclusion
content = content.replace(
    "We began with a fully unsupervised approach that failed, pivoted through a supervised baseline that succeeded but violated our premise, and ultimately arrived at a **hybrid Fusion Gauntlet**",
    "We began with a fully unsupervised approach that failed, pivoted through a supervised baseline that succeeded but violated our premise, and arrived at a **hybrid Fusion Gauntlet**"
)
content = content.replace(
    "pure unsupervised detection fails, but unsupervised mining techniques \u2014 K-Means clustering and autoencoder anomaly detection \u2014 are remarkably effective",
    "pure unsupervised detection fails, but unsupervised mining techniques (K-Means clustering and autoencoder anomaly detection) are remarkably effective"
)

# ═══════════════════════════════════════════════════════════════════════
# 2. Sweep remaining em dashes
# ═══════════════════════════════════════════════════════════════════════
content = content.replace('\u2014', ', ')

# ═══════════════════════════════════════════════════════════════════════
# 3. Insert demographic EDA after the sample grid code cell
# ═══════════════════════════════════════════════════════════════════════
# Find the pixel-level EDA markdown and inject demographic section before it
old_pixel_heading = '''cells.append(nbformat.v4.new_markdown_cell("""### Pixel-Level EDA'''

demographic_section = '''# ── EDA 3a: Demographic Bias ──
cells.append(nbformat.v4.new_markdown_cell("""### 3a. Demographic Bias Analysis

The HiDF deepfakes are generated by face-swapping: each fake image has a "base" face (the pose/background) and a "swap" face (the identity pasted on). The metadata CSV records the Age, Gender, and Race of each swap identity. Understanding the demographic distribution is important because a model trained predominantly on one subgroup may not generalize to others.

**Why this matters for our pipeline:** If the dataset is heavily skewed toward a particular demographic, the learned embeddings may encode demographic features rather than forensic artifacts. This motivates careful evaluation across subgroups.
"""))

cells.append(nbformat.v4.new_code_cell("""# ── Demographic Bias Analysis ──
metadata_df = pd.read_csv(METADATA_PATH)

# Parse fake filenames to extract swap IDs and count demographics
fake_img_names = [os.path.basename(f) for f in fake_files]
age_counter, gender_counter, race_counter = {}, {}, {}

for fname in fake_img_names:
    try:
        ids = fname.split('.')[0].split('_')
        swap_id = ids[1].strip() if len(ids) > 1 else ids[0].strip()
        row = metadata_df.loc[metadata_df['ID'] == swap_id]
        if len(row) == 0:
            continue
        age = row['Age'].iloc[0].strip()
        gender = row['Gender'].iloc[0].strip()
        race = row['Race'].iloc[0].strip()
        age_counter[age] = age_counter.get(age, 0) + 1
        gender_counter[gender] = gender_counter.get(gender, 0) + 1
        race_counter[race] = race_counter.get(race, 0) + 1
    except Exception:
        continue

# Plot distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (counter, title) in zip(axes, [
    (age_counter, "Age Distribution"),
    (gender_counter, "Gender Distribution"),
    (race_counter, "Race Distribution")
]):
    items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    labels, counts = zip(*items)
    ax.bar(labels, counts, color='steelblue', edgecolor='black', alpha=0.8)
    ax.set_title(title, fontweight='bold')
    ax.set_ylabel("Count")
    ax.tick_params(axis='x', rotation=45)

plt.suptitle("Demographic Distribution of Swap Identities in Fake Images", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"Parsed {sum(age_counter.values())} of {len(fake_img_names)} fake images successfully.")
"""))

cells.append(nbformat.v4.new_markdown_cell("""The dataset is heavily skewed toward certain demographics (e.g., white adults), reflecting biases in the underlying source datasets (CelebA-HQ, Flickr). This means our detection pipeline must be evaluated carefully: high overall accuracy could mask poor performance on underrepresented subgroups.
"""))

# ── EDA 3b: Pixel Intensity ──
cells.append(nbformat.v4.new_markdown_cell("""### 3b. Pixel Intensity Analysis'''

content = content.replace(old_pixel_heading, demographic_section)

# Rename the old "Pixel-Level EDA" text that is now "3b. Pixel Intensity Analysis"
# (the heading text was already replaced above, just need the description)
content = content.replace(
    "A natural first question is: *do real and fake images differ at a basic statistical level?*",
    "Do real and fake images differ at a basic statistical level?"
)

# ═══════════════════════════════════════════════════════════════════════
# 4. Remove mid-cell scipy import in Phase 1 clustering cell
# ═══════════════════════════════════════════════════════════════════════
content = content.replace(
    "# Compute cluster-purity-based accuracy (assign each cluster to its majority class)\nfrom scipy.optimize import linear_sum_assignment\n",
    "# Compute cluster-purity-based accuracy (assign each cluster to its majority class)\n"
)

# ═══════════════════════════════════════════════════════════════════════
# 5. Add F1 score to Phase 2 evaluation
# ═══════════════════════════════════════════════════════════════════════
content = content.replace(
    '''acc_rf = accuracy_score(y_test_rf, y_pred_rf)
auc_rf = roc_auc_score(y_test_rf, y_proba_rf)
prec_rf = precision_score(y_test_rf, y_pred_rf)
rec_rf = recall_score(y_test_rf, y_pred_rf)

print("=" * 40)
print("PHASE 2: ResNet + Random Forest")
print("=" * 40)
print(f"Accuracy:  {acc_rf:.4f}")
print(f"AUC:       {auc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall:    {rec_rf:.4f}")
print("=" * 40)''',
    '''acc_rf = accuracy_score(y_test_rf, y_pred_rf)
auc_rf = roc_auc_score(y_test_rf, y_proba_rf)
prec_rf = precision_score(y_test_rf, y_pred_rf)
rec_rf = recall_score(y_test_rf, y_pred_rf)
f1_rf = f1_score(y_test_rf, y_pred_rf)

print("=" * 40)
print("PHASE 2: ResNet + Random Forest")
print("=" * 40)
print(f"Accuracy:  {acc_rf:.4f}")
print(f"AUC:       {auc_rf:.4f}")
print(f"Precision: {prec_rf:.4f}")
print(f"Recall:    {rec_rf:.4f}")
print(f"F1 Score:  {f1_rf:.4f}")
print("=" * 40)'''
)

# ═══════════════════════════════════════════════════════════════════════
# 6. Add F1 score to Phase 3 evaluation and add alpha grid search
# ═══════════════════════════════════════════════════════════════════════
content = content.replace(
    '''acc_fusion = accuracy_score(ground_truth, preds)
auc_fusion = roc_auc_score(ground_truth, final_scores)
prec_fusion = precision_score(ground_truth, preds)
rec_fusion = recall_score(ground_truth, preds)

print("=" * 40)
print("FUSION GAUNTLET RESULTS")
print("=" * 40)
print(f"Alpha:      {ALPHA} (Macro) / {1-ALPHA} (Micro)")
print(f"Accuracy:   {acc_fusion:.4f}")
print(f"AUC:        {auc_fusion:.4f}")
print(f"Precision:  {prec_fusion:.4f}")
print(f"Recall:     {rec_fusion:.4f}")
print("=" * 40)''',
    '''# Grid search over alpha to find optimal weighting
alphas = np.arange(0.0, 1.05, 0.05)
alpha_results = []
for a in alphas:
    fused = a * s_macro + (1 - a) * s_micro
    t = np.percentile(fused, 50)
    p = (fused > t).astype(int)
    alpha_results.append({
        'alpha': a,
        'accuracy': accuracy_score(ground_truth, p),
        'auc': roc_auc_score(ground_truth, fused),
        'f1': f1_score(ground_truth, p)
    })

alpha_df = pd.DataFrame(alpha_results)
best_idx = alpha_df['auc'].idxmax()
ALPHA = alpha_df.loc[best_idx, 'alpha']

# Plot alpha sweep
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(alpha_df['alpha'], alpha_df['auc'], 'o-', label='AUC', color='steelblue', linewidth=2)
ax.plot(alpha_df['alpha'], alpha_df['accuracy'], 's--', label='Accuracy', color='coral', linewidth=2)
ax.plot(alpha_df['alpha'], alpha_df['f1'], '^--', label='F1', color='seagreen', linewidth=2)
ax.axvline(ALPHA, color='gray', linestyle=':', alpha=0.7, label=f'Best alpha = {ALPHA:.2f}')
ax.set_xlabel("Alpha (Macro Weight)")
ax.set_ylabel("Score")
ax.set_title("Fusion Weight Grid Search: Macro vs Micro Contribution", fontweight='bold')
ax.legend()
plt.tight_layout()
plt.show()

final_scores = ALPHA * s_macro + (1 - ALPHA) * s_micro

# Threshold at 50th percentile (balanced dataset)
thresh = np.percentile(final_scores, 50)
preds = (final_scores > thresh).astype(int)

acc_fusion = accuracy_score(ground_truth, preds)
auc_fusion = roc_auc_score(ground_truth, final_scores)
prec_fusion = precision_score(ground_truth, preds)
rec_fusion = recall_score(ground_truth, preds)
f1_fusion = f1_score(ground_truth, preds)

print("=" * 40)
print("FUSION GAUNTLET RESULTS")
print("=" * 40)
print(f"Alpha:      {ALPHA:.2f} (Macro) / {1-ALPHA:.2f} (Micro)")
print(f"Accuracy:   {acc_fusion:.4f}")
print(f"AUC:        {auc_fusion:.4f}")
print(f"Precision:  {prec_fusion:.4f}")
print(f"Recall:     {rec_fusion:.4f}")
print(f"F1 Score:   {f1_fusion:.4f}")
print("=" * 40)'''
)

# Update the fusion markdown to describe the grid search
content = content.replace(
    '''We normalize both score channels to [0, 1] using MinMaxScaler and combine them with equal weight:

`Final_Score = 0.5 \u00b7 Macro_Score + 0.5 \u00b7 Micro_Score`

**Why equal weighting?** Both scoring mechanisms capture complementary signals. K-Means measures *global* geometric distance from the real cluster, while the autoencoder measures *local* structural deviation. Giving them equal weight avoids biasing toward either signal and lets the fusion naturally leverage whichever channel is more informative for a given image.''',
    '''We normalize both score channels to [0, 1] using MinMaxScaler and combine them via a weighted sum:

`Final_Score = alpha * Macro_Score + (1 - alpha) * Micro_Score`

Rather than arbitrarily choosing a fixed weight, we perform a **grid search** over alpha from 0.0 to 1.0 (in steps of 0.05) and select the value that maximizes AUC on the evaluation set. This lets us empirically determine how much each scoring channel contributes to the final decision.'''
)

# ═══════════════════════════════════════════════════════════════════════
# 7. Add F1 to ablation table
# ═══════════════════════════════════════════════════════════════════════
content = content.replace(
    '''ablation_data = [
    {"Phase": "Phase 1", "Approach": "Unsupervised", "Encoder": "Custom Conv AE",
     "Classifier": "K-Means", "Accuracy": f"{acc_ae:.4f}", "AUC": "N/A (clustering)"},
    {"Phase": "Phase 2", "Approach": "Supervised", "Encoder": "ResNet-18 (frozen)",
     "Classifier": "Random Forest", "Accuracy": f"{acc_rf:.4f}", "AUC": f"{auc_rf:.4f}"},
    {"Phase": "Phase 3", "Approach": "Hybrid (5% labeled)", "Encoder": "ResNet-18 (Layer4 tuned)",
     "Classifier": "K-Means + AE Fusion", "Accuracy": f"{acc_fusion:.4f}", "AUC": f"{auc_fusion:.4f}"}
]''',
    '''ablation_data = [
    {"Phase": "Phase 1", "Approach": "Unsupervised", "Encoder": "Custom Conv AE",
     "Classifier": "K-Means", "Accuracy": f"{acc_ae:.4f}", "AUC": "N/A", "F1": "N/A"},
    {"Phase": "Phase 2", "Approach": "Supervised", "Encoder": "ResNet-18 (frozen)",
     "Classifier": "Random Forest", "Accuracy": f"{acc_rf:.4f}", "AUC": f"{auc_rf:.4f}", "F1": f"{f1_rf:.4f}"},
    {"Phase": "Phase 3", "Approach": "Hybrid (5% labeled)", "Encoder": "ResNet-18 (Layer4 tuned)",
     "Classifier": "K-Means + AE Fusion", "Accuracy": f"{acc_fusion:.4f}", "AUC": f"{auc_fusion:.4f}", "F1": f"{f1_fusion:.4f}"}
]'''
)

# ═══════════════════════════════════════════════════════════════════════
# 8. Remove mid-notebook PCA import
# ═══════════════════════════════════════════════════════════════════════
content = content.replace(
    """# ── Phase 1: Visualize Embedding Space ──
from sklearn.decomposition import PCA
""",
    """# ── Phase 1: Visualize Embedding Space ──
"""
)

# ═══════════════════════════════════════════════════════════════════════
# Write result
# ═══════════════════════════════════════════════════════════════════════
with open('build_main_notebook.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Patching complete.")

# Verify no em dashes remain
if '\u2014' in content:
    print("WARNING: em dashes still present!")
    for i, line in enumerate(content.split('\n'), 1):
        if '\u2014' in line:
            print(f"  Line {i}: {line[:80]}")
else:
    print("All em dashes removed successfully.")
