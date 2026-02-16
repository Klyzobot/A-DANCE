# A-DANCE  
## Alzheimer’s Disease Analysis Network of Co-expression  

A-DANCE is a systems-level computational framework for transcriptomic classification of Alzheimer’s disease (AD). The pipeline integrates network-informed gene selection, multi-model benchmarking, class balancing, and automated machine learning (AutoML) optimization to identify predictive molecular signatures from RNA-Seq data.

---

## Abstract

Alzheimer’s disease is characterized by complex, system-wide transcriptional dysregulation rather than isolated gene perturbations. A-DANCE models this dysregulation using a structured machine learning pipeline built upon network-selected gene features.

Using curated expression profiles (n = 280 genes), the framework evaluates classical and ensemble classifiers, applies class balancing (SMOTE), and employs TPOT-based AutoML to identify an optimized predictive model.

Final selected classifier:
- Logistic Regression  
- Accuracy ≈ 84%  
- ROC-AUC ≈ 0.94–0.95  

This repository contains the full computational pipeline, benchmarking outputs, and trained model artifacts.

---

## Study Design

### Data Source
- Public transcriptomic datasets (GEO)
- Brain tissue samples
- Binary classification: Control vs Alzheimer’s Disease

### Feature Space
- 280 network-selected genes
- Derived from prior co-expression network analysis
- Centrality-informed feature prioritization

---

## Computational Framework

### 1. Data Preprocessing
- Expression matrix loading
- Metadata alignment
- Binary label encoding (0 = Control, 1 = AD)
- Stratified 80/20 train–test split

### 2. Class Imbalance Correction
- SMOTE (Synthetic Minority Oversampling Technique)
- Applied to training data only
- Balanced datasets exported for reproducibility

### 3. Model Benchmarking

The following classifiers were implemented and evaluated:

- Logistic Regression  
- Random Forest  
- Support Vector Machine  
- K-Nearest Neighbors  
- Naive Bayes  
- Multi-layer Perceptron  
- XGBoost  
- LightGBM  
- CatBoost  
- TabNet  

Evaluation Metrics:
- Accuracy
- ROC-AUC
- Classification report
- Confusion matrices

---

### 4. Automated Machine Learning (AutoML)

TPOT (Tree-based Pipeline Optimization Tool) was used for automated pipeline search and hyperparameter tuning.

Configuration:
- Generations: 5  
- Population size: 50  
- Stratified cross-validation  

Best-performing model selected by TPOT:
Logistic Regression  

The optimized pipeline was exported and serialized for reproducibility.

---

## Repository Structure

```
A-DANCE/
│
├── selected_expression_matrix.csv
├── metadata_binary.csv
├── balanced_training_data.csv
├── testing_data.csv
├── best_tpot_model.pkl
├── Project_Report.pdf
└── README.md
```

---

## Reproducibility

To reproduce results:

1. Install required dependencies.
2. Run the main analysis script.
3. Confusion matrices and model artifacts will be generated automatically.

### Dependencies

- Python 3.x
- Scikit-learn
- TPOT
- imbalanced-learn
- XGBoost
- LightGBM
- CatBoost
- PyTorch TabNet
- Pandas
- NumPy
- Matplotlib

---

## Detailed Documentation

A comprehensive description of methodology, network modeling strategy, benchmarking analysis, and full experimental results is available in:

👉 [Project_Report.pdf](./Project_Report.pdf)

---

## Research Significance

A-DANCE demonstrates:

- Integration of network biology with supervised learning  
- Benchmarking of classical and ensemble ML models  
- AutoML-driven optimization in biomedical transcriptomics  
- Exportable and reusable trained pipeline  

The framework is modular and extendable to additional neurodegenerative or transcriptomic datasets.

---

## Future Directions

- External cohort validation  
- Multi-omics integration  
- Explainable AI (SHAP / LIME)  
- Network robustness and perturbation analysis  

---

## Author

Nakshatra Malhotra  
Undergraduate Researcher – Systems & Computational Biology  
SRM University, AP