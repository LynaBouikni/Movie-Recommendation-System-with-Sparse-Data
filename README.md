# 🎬 Movie Recommendation System with Sparse Data

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/LynaBouikni/Movie-Recommendation-System-with-Sparse-Data.svg)](https://github.com/LynaBouikni/Movie-Recommendation-System-with-Sparse-Data)
[![Repo Size](https://img.shields.io/github/repo-size/LynaBouikni/Movie-Recommendation-System-with-Sparse-Data.svg)](https://github.com/LynaBouikni/Movie-Recommendation-System-with-Sparse-Data)
[![Status](https://img.shields.io/badge/Project-Complete-green.svg)](https://github.com/LynaBouikni)

> Collaborative Filtering · Matrix Factorization · SVD · KNN · Ensemble Modeling


> Collaborative Filtering · Matrix Factorization · SVD · KNN · Ensemble Modeling

---

## 📌 Overview

This project tackles one of the core challenges in building movie recommendation systems: **extremely sparse rating data**. Using a dataset with over **98% missing values**, we systematically explored and compared several approaches for predicting user ratings:

- Matrix Factorization with Gradient Descent (baseline)
- Matrix Factorization Ensembling (20 models)
- k-Nearest Neighbors with Cosine Similarity
- Singular Value Decomposition (SVD)
- An **ensemble method** combining MF, SVD, and KNN for robust predictions

All methods were benchmarked on RMSE and accuracy, with thoughtful trade-offs between prediction quality and computational cost.

---

## 🎯 Objective

> How can we effectively address the challenge of sparse data in recommendation systems?

This project aims to:

- Understand the limitations of traditional recommender algorithms on sparse data  
- Test different models and aggregation techniques to improve prediction quality  
- Compare runtime, RMSE, and accuracy across all methods  
- Analyze overfitting tendencies and generalization capabilities  

---

## 🗂 Dataset

- **Source**: Provided as part of an academic assignment at **Université Paris Dauphine - PSL / M2 IASD**
- **Files**:
  - `ratings_train.npy`: user-movie rating matrix (train set)
  - `ratings_test.npy`: test set
  - `names_genre.npy`: genre labels

- **Shape**: 610 users × 4980 movies  
- **Missing values**: ~98% of the matrix is unobserved  
- **Sparsity**: Data strongly follows **Zipf’s Law** → ideal for low-rank approximations

---

## ⚙️ Methods

### 🧩 1. Matrix Factorization (Baseline)
- Optimized using **Gradient Descent**  
- Regularization to avoid overfitting  
- Tuned: latent dimension K = 5

### 🧪 2. Matrix Factorization Ensembling
- Aggregated predictions from **20 parallel MF models**  
- Averaging yielded better results than voting

### 🔍 3. Singular Value Decomposition (SVD)
- Tuned on number of singular values (K = 5)  
- Trade-off observed: lower RMSE vs increased overfitting

### 🧑‍🤝‍🧑 4. k-Nearest Neighbors (KNN)
- Cosine similarity for user-user matching  
- Explored multiple values of k (best ~50)

### 🎛️ 5. Final Ensemble Method
- Combined MF, SVD, and KNN in a **fold-wise aggregation pipeline**  
- Aggregation via RMSE-optimal averaging

---

## 📊 Results

| Model                    | RMSE   | Accuracy (%) | Time (s) |
|--------------------------|--------|---------------|----------|
| MF (baseline)            | 0.969  | 24.4          | 188.15   |
| MF (ensemble, 20x)       | **0.89** | 24.89         | 101.37   |
| MF + SVD + KNN Ensemble  | 0.992  | **27.73**     | **81.65**|

⚖️ **Trade-Offs Observed**:

- MF Ensembling had the lowest RMSE.  
- The full ensemble had the best **accuracy** and **speed**.  
- SVD offered compact solutions but overfit easily at higher ranks.

---

## 🛠 Tech Stack

- **Languages**: Python  
- **Core Libraries**:
  - `numpy==1.25.1`
  - `scikit-learn==1.3.0`
  - `scipy==1.11.1`
  - `torch==2.0.1` *(for future enhancements or benchmarking)*

---

## 🧾 Hyperparameters

- `K (latent factors)` = 5  
- `Learning rate` = 1.4e-4  
- `Regularization λ, μ` = 0.1, 1  
- `Parallel MF models` = 20  
- `KNN (k)` = 50  
- `SVD (components)` = 5  

---

## 🧠 Key Learnings

- Sparse matrix completion requires strong regularization and efficient masking.  
- Ensembling helps **stabilize predictions** across folds and methods.  
- Voting is computationally expensive and often less effective than averaging.  
- Overfitting risks increase with richer representations (e.g., SVD).  
- Trade-offs between accuracy, runtime, and complexity must be carefully balanced.

---

## 👥 Authors

**Matrix Brigade**  
- Lyna Bouikni  
- Insaf Medjaouri  
- Roger Marius  

📅 Oct 2023 | 🏫 M2 IASD — ENS, Dauphine, Mines Paris - PSL

---

## 🤝 Acknowledgements

- Project for the **Data Science Lab** (M2 IASD)  
- Inspired by collaborative filtering literature and matrix completion theory

---

_“In recommender systems, sparse data isn't a weakness — it's an invitation to innovate.”_
