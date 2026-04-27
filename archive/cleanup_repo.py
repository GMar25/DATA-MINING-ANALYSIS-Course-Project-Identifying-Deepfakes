"""Repo cleanup script: rename, organize, and move files."""
import os
import shutil

ROOT = os.path.dirname(os.path.abspath(__file__))

# Create directories
for d in ['archive', 'checkpoints', 'tools']:
    os.makedirs(os.path.join(ROOT, d), exist_ok=True)

# --- 1. Move checkpoints ---
for f in ['Checkpoint_1.ipynb', 'Checkpoint_2.ipynb']:
    src = os.path.join(ROOT, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(ROOT, 'checkpoints', f))
        print(f'  Moved {f} -> checkpoints/')

# --- 2. Rename Final_Project notebooks (strip extra identifiers) ---
renames = {
    'Final_Project_v14_Final.ipynb': 'Final_Project_v14.ipynb',
    'Final_Project_v15_DataMining.ipynb': 'Final_Project_v15.ipynb',
    'Final_Project_v16_ExtremeMining.ipynb': 'Final_Project_v16.ipynb',
    'Final_Project_v17_SMOTE_Optimization.ipynb': 'Final_Project_v17.ipynb',
    'Final_Project_v18_PCA_Guillotine.ipynb': 'Final_Project_v18.ipynb',
    'Final_Project_v19_Classical_Nuclear.ipynb': 'Final_Project_v19.ipynb',
    'Final_Project_v20_Dual_Pipeline.ipynb': 'Final_Project_v20.ipynb',
    'Final_Project_v21_Isolation.ipynb': 'Final_Project_v21.ipynb',
    'Final_Project_v22_Weak_GMM.ipynb': 'Final_Project_v22.ipynb',
    'Final_Project_v23_Imbalance.ipynb': 'Final_Project_v23.ipynb',
    'Final_Project_v24_Forced_GMM.ipynb': 'Final_Project_v24.ipynb',
    'Final_Project_v25_PCA_Forest.ipynb': 'Final_Project_v25.ipynb',
    'Final_Project_v26_KMeans_IForest.ipynb': 'Final_Project_v26.ipynb',
    'Final_Project_v27_Crippled_Anomaly.ipynb': 'Final_Project_v27.ipynb',
    'Final_Project_v28_Mahalanobis.ipynb': 'Final_Project_v28.ipynb',
    'Final_Project_v29_Anchored_KMeans.ipynb': 'Final_Project_v29.ipynb',
    'Final_Project_v29_LedoitWolf.ipynb': 'Final_Project_v29b.ipynb',
    'Final_Project_v32_Balanced_LOF.ipynb': 'Final_Project_v32.ipynb',
    'Final_Project_v33_Dual_Refinement.ipynb': 'Final_Project_v33.ipynb',
    'Final_Project_v34_Fusion_Gauntlet.ipynb': 'Final_Project_v34.ipynb',
}

for old, new in renames.items():
    src = os.path.join(ROOT, old)
    if os.path.exists(src):
        os.rename(src, os.path.join(ROOT, new))
        print(f'  Renamed {old} -> {new}')

# --- 3. Move all Final_Project_v*.ipynb to archive ---
for f in os.listdir(ROOT):
    if f.startswith('Final_Project_v') and f.endswith('.ipynb'):
        shutil.move(os.path.join(ROOT, f), os.path.join(ROOT, 'archive', f))
        print(f'  Moved {f} -> archive/')

# --- 4. Move old build scripts to archive ---
build_scripts = [
    'build_v12.py', 'build_v13.py', 'build_v13_resnet.py', 'build_v14.py',
    'build_v15.py', 'build_v16.py', 'build_v17.py', 'build_v18.py',
    'build_v19.py', 'build_v20_dual_pipeline.py', 'build_v21_isolation.py',
    'build_v22_weak_gmm.py', 'build_v23_imbalance.py', 'build_v24_forced_k.py',
    'build_v25_pca_forest.py', 'build_v26_kmeans_iforest.py',
    'build_v27_crippled_anomaly.py', 'build_v28_mahalanobis.py',
    'build_v29_ledoit_wolf.py', 'build_v32_one_class_lof.py',
    'build_v33_dual_refinement.py', 'build_v34_fusion_gauntlet.py',
]
for f in build_scripts:
    src = os.path.join(ROOT, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(ROOT, 'archive', f))
        print(f'  Moved {f} -> archive/')

# --- 5. Move scratch/helper files to archive ---
scratch_files = [
    'scratch_eval.py', 'scratch_eval2.py', 'scratch_test.py',
    'inspect_notebooks.py',
]
for f in scratch_files:
    src = os.path.join(ROOT, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(ROOT, 'archive', f))
        print(f'  Moved {f} -> archive/')

# --- 6. Move presentation/tools files to tools/ ---
tools_files = [
    'slide9_adversarial_advantage.py',
    'Image_Sampler.ipynb',
    'Presentation_Graphics_Generator.ipynb',
    'Presentation_Graphics_Generator_Live.ipynb',
]
for f in tools_files:
    src = os.path.join(ROOT, f)
    if os.path.exists(src):
        shutil.move(src, os.path.join(ROOT, 'tools', f))
        print(f'  Moved {f} -> tools/')

# Move Slide9 PNGs to tools/
for f in os.listdir(ROOT):
    if f.startswith('Slide9_') and f.endswith('.png'):
        shutil.move(os.path.join(ROOT, f), os.path.join(ROOT, 'tools', f))
        print(f'  Moved {f} -> tools/')

print('\nDone! Repo cleaned up.')
