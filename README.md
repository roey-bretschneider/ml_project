# DDoS Attack Detection and Characterization

### Machine Learning for Networks (ML4N) Project — January 2026

This project develops a complete Machine Learning pipeline to automate the detection and analysis of network flows generated during **Distributed Denial of Service (DDoS)** attacks. Using flow-level data extracted via `CICFlowMeter-V3`, the pipeline addresses both supervised classification and unsupervised clustering tasks to identify malicious traffic patterns.



---

## 📂 Project Structure

### 🛠 Core Python Modules

* **`supervised.py`**: The main entry point for the classification task. It handles data loading, preprocessing, initial model screening, and final evaluation across different dataset splits.
* **`clustering.py`**: A class-based framework for unsupervised learning. It implements logic for -evaluation (Elbow/Silhouette), hyperparameter grid searches, and visualization of "natural groupings" in traffic.
* **`utils.py`**: A shared utility library containing functions for normalization, correlation-based feature dropping, PCA dimensionality reduction, and automated plotting (Confusion Matrices, ECDFs, and Heatmaps).
* **`analyze_misclassification_features.py`**: A diagnostic script used to investigate "feature overlap." It compares misclassified samples against correct ones to identify why certain attack types are confused by the model.

### ⚙️ Configuration

* **`params.json`**: Centralized configuration for the supervised pipeline, including model hyperparameters and columns to exclude from scaling.
* **`clustering_params.json`**: (Auto-generated) Stores the optimized parameters for K-Means, GMM, and DBSCAN.

---

## 🚀 Workflow

### 1. Data Pre-processing

The pipeline performs extensive feature engineering and cleaning:

* 
**Feature Engineering**: Generates volumetric features such as total packets and total bytes per flow.


* **Redundancy Removal**: Identifies and drops features with a correlation threshold higher than 0.90 to mitigate multicollinearity.
* **Dimensionality Reduction**: Utilizes **Principal Component Analysis (PCA)** to reduce the feature space while retaining 95% of the variance.

### 2. Supervised Learning (Classification)

We evaluate four primary classifiers: **Logistic Regression, Random Forest, Decision Tree, and Gaussian Naive Bayes**.

* 
**Splitting Strategies**: Models are tested using a **Per-Flow Split** (random shuffle) and a **Per-Time Split** (chronological) to evaluate robustness against concept drift.


* 
**Key Findings**: The **Decision Tree** emerged as the best model, offering high performance with significantly lower training and inference costs compared to Random Forest.



### 3. Unsupervised Learning (Clustering)

The goal is to understand if attacks naturally group into "families" independent of labels.

* 
**Algorithms**: K-Means, Gaussian Mixture Models (GMM), and DBSCAN.


* 
**Evaluation**: Internal quality is measured via **Silhouette Score**, while alignment with ground truth is verified using **Adjusted Rand Index (ARI)**.


* 
**Key Findings**: Attack families are better defined by **density continuity** (DBSCAN) than by distance from a centroid (K-Means).



---

## 📊 Summary of Results

| Task | Best Performing Method | Key Observation |
| --- | --- | --- |
| **Classification** | **Decision Tree** | Best balance of accuracy and inference speed.

 |
| **Clustering** | **DBSCAN / GMM** | Attack patterns are better represented by density or Gaussian distributions than spherical clusters.

 |

---

## 💻 How to Run

1. **Configure Environment**: Ensure `scikit-learn`, `pandas`, `seaborn`, and `scikit-learn-extra` are installed.
2. **Supervised Pipeline**: Execute `python supervised.py` to generate classification reports and plots in the `plots/` folder.
3. **Clustering Pipeline**: Execute `python clustering.py` to evaluate cluster counts and generate PCA visualizations.
4. **Misclassification Analysis**: Run `python analyze_misclassification_features.py` to diagnose model errors between specific attack classes.

---

## 🏫 Project Context

This project was developed for the **Machine Learning for Networks** course at **Politecnico di Torino**.

