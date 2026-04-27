# Identifying Deepfakes: A Data Mining Approach

Deepfakes, AI-generated images indistinguishable from genuine photographs, pose a growing threat to information integrity, financial security, and personal identity. This project investigates whether **unsupervised data mining techniques** can serve as the backbone of a quickly adaptable deepfake detection system, using the [HiDF (Human-Indistinguishable Deepfake) dataset](https://dl.acm.org/doi/epdf/10.1145/3711896.3737399).

The project traces a three-phase experimental journey: from a fully unsupervised approach that fails, through a supervised baseline that succeeds but contradicts the premise, to a hybrid **"Fusion Gauntlet"** pipeline combining deep neural embeddings with classical K-Means clustering and autoencoder anomaly detection.

## Start Here

**Main deliverable:** [`main_notebook.ipynb`](main_notebook.ipynb)

**Project video:** [YouTube Link](https://youtube.com) *(placeholder)*

## Research Questions

> **Can unsupervised data mining techniques be used to build a quickly adaptable and accurate deepfake detector?**

Investigated through three phases:
1. **Phase 1:** Pure unsupervised (Custom Autoencoder + K-Means) -- does unsupervised work at all?
2. **Phase 2:** Supervised pivot (ResNet-18 + Random Forest) -- is the problem solvable with better features?
3. **Phase 3:** Hybrid fusion (ResNet embeddings + K-Means + Anomaly Detection) -- can we keep the features but drop the supervised classifier?

## Data

- **Dataset:** [HiDF (Human-Indistinguishable Deepfake)](https://zenodo.org/records/16140829)
- **Source:** Zenodo (downloaded automatically by the notebook)
- **Contents:** Balanced set of real and AI-generated face images, plus demographic metadata
- **Preprocessing:** Images are resized to 224x224 and normalized using ImageNet statistics for ResNet-18 feature extraction

## How to Reproduce

This project was built and tested in **Google Colab with a T4 GPU**.

1. Open `main_notebook.ipynb` in Google Colab
2. Enable GPU runtime: `Runtime > Change runtime type > T4 GPU`
3. Run all cells sequentially. The dataset downloads automatically from Zenodo on first run.
4. See `requirements.txt` for the full Colab environment snapshot.

## Key Dependencies

| Package | Version |
|---------|---------|
| Python | 3.11+ |
| PyTorch | 2.2+ |
| torchvision | 0.17+ |
| scikit-learn | 1.4+ |
| pandas | 2.2+ |
| numpy | 1.26+ |
| matplotlib | 3.8+ |
| seaborn | 0.13+ |
| scipy | 1.12+ |

Full dependency list: [`requirements.txt`](requirements.txt) (generated from Colab at the end of the notebook)

## Repository Structure

```
.
├── main_notebook.ipynb          # Main deliverable (start here)
├── build_main_notebook.py       # Script that generates the notebook
├── README.md
├── Presentation.pdf             # Project presentation slides
├── requirements.txt             # Colab environment snapshot
├── .gitignore
├── checkpoints/
│   ├── Checkpoint_1.ipynb       # Dataset selection and EDA
│   └── Checkpoint_2.ipynb       # Research question formation
├── archive/                     # Previous experiment versions (v1-v34)
│   ├── Final_Project_v1.ipynb
│   ├── ...
│   └── Final_Project_v34.ipynb
├── tools/                       # Presentation graphics generators
│   ├── Presentation_Graphics_Generator.ipynb
│   └── ...
└── sample_projects/             # Reference sample projects
```

## Results Summary

| Phase | Approach | Accuracy | AUC |
|-------|----------|----------|-----|
| Phase 1 | Unsupervised (Custom AE + K-Means) | ~50% | N/A |
| Phase 2 | Supervised (ResNet + Random Forest) | High | High |
| Phase 3 | Hybrid Fusion (5% labeled) | Competitive | Competitive |

Pure unsupervised detection fails, but unsupervised mining techniques (K-Means + autoencoder anomaly detection) are effective *when paired with competent feature extraction*. The supervised component can be minimized to a 5% data handicap and a single unfrozen layer, making the system rapidly adaptable to new threats.

## Author

**Gage Mariano** | Data Mining & Analysis Course Project
