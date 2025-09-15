# Fraudulent Job Posting Detection with CGCL

This repository contains the implementation of our proposed **Centroid-Guided Contrastive Loss (CGCL)** for fraudulent job posting detection.  
CGCL unifies **classification** and **clustering** objectives to learn discriminative and compact latent-space representations, improving both accuracy and structural interpretability.

---

## 🚀 Features
- **Centroid-Guided Contrastive Loss (CGCL)** with top-k push-and-pull mechanism.
- Integration of **Cross-Entropy Loss** for strong class discrimination.
- Supports multiple feature extraction techniques:
  - **Word2Vec**
  - **GloVe**
  - **TF-IDF**
- Extensive experiments with hyperparameter tuning.
- Evaluation on both **classification** and **clustering** metrics.

---

## 📂 Project Structure

```
.
├── data/                 # Dataset and preprocessing scripts
├── models/              # Model definitions and loss functions
├── utils/               # Helper functions
```

---

## ⚙️ Installation
Clone this repository and install the required dependencies:

```bash
git clone https://github.com/your-username/fraudulent-job-detection-cgcl.git
cd fraudulent-job-detection-cgcl
pip install -r requirements.txt
```

---

## 📊 Results
Our final **GloVe-based model** achieved:
* **Accuracy:** 99.2%
* **F1-score:** 98.7%
* Strong clustering performance with:
   * Silhouette Score: 0.701
   * Adjusted Rand Index: 0.953
   * Normalized Mutual Information: 0.903

### Performance Comparison

| Embedding | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | Silhouette | ARI | NMI |
|-----------|--------------|---------------|------------|--------------|------------|-----|-----|
| Word2Vec  | 94.8         | 89.6          | 93.2       | 91.4         | 0.425      | 0.723 | 0.712 |
| GloVe     | **99.2**     | **98.9**      | **98.5**   | **98.7**     | **0.701**  | **0.953** | **0.903** |
| TF-IDF    | 97.1         | 95.3          | 96.8       | 96.0         | 0.593      | 0.847 | 0.825 |

### Hyperparameter Analysis

#### Optimal Configuration (GloVe)
- **Learning Rate:** 1e-4
- **Hidden Dimensions:** [512, 256, 128]
- **Temperature (τ):** 0.1
- **Top-k:** 10
- **Epochs:** 2000
- **Loss Weight (α):** 0.7

#### Learning Rate Impact
| Learning Rate | Accuracy (%) | F1-Score (%) |
|---------------|--------------|--------------|
| 1e-3          | 92.4         | 90.2         |
| **1e-4**      | **99.2**     | **98.7**     |
| 1e-5          | 96.8         | 95.4         |

---

## 📈 Visualizations
* **AUC-ROC curve** for the best-performing model.
* **Latent-space visualization** showing compact and separable clusters.
* **Loss convergence plots** during training.
* **Hyperparameter sensitivity analysis**.

---

## 🔬 Key Technical Contributions

### Centroid-Guided Contrastive Loss (CGCL)
```
L_CGCL = L_CE + α · L_contrastive

where:
L_contrastive = (1/N) Σ [pull_loss + push_loss]
pull_loss = d(z_i, c_y_i)²
push_loss = max(0, margin - d(z_i, c_j))²
```

### Architecture Overview
1. **Feature Extraction Layer**: Processes text embeddings (Word2Vec/GloVe/TF-IDF)
2. **Multi-layer Perceptron**: [Input → 512 → 256 → 128 → 64 → 2]
3. **Dual Objective Training**: Classification + Contrastive Learning
4. **Top-k Selection**: Focuses on hardest negatives for efficient training

---

## 📊 Dataset Information

| Metric | Value |
|--------|-------|
| Total Samples | 17,880 |
| Fraudulent Posts | 866 (4.8%) |
| Legitimate Posts | 17,014 (95.2%) |
| Train/Test Split | 80/20 |
| Features | 18 (text + metadata) |

### Feature Categories
- **Textual Features**: Job title, description, requirements, benefits
- **Metadata Features**: Company profile, location, employment type, experience level

---

## 📑 Citation
If you use this work, please cite our paper:

```bibtex
@article{yourpaper2025,
  title={Fraudulent Job Posting Detection with Centroid-Guided Contrastive Loss},
  author={Your Name and Co-authors},
  year={2025},
  journal={Conference/Journal Name}
}
```

---

## 🏷️ Keywords
Fraud Detection, Contrastive Learning, Clustering, Word2Vec, GloVe, TF-IDF, Anomaly Detection, Representation Learning

---

## 🤝 Contributing
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

### Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/
```

---

## 📧 Contact
For any queries, please contact: **k224058@nu.edu.pk**

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
