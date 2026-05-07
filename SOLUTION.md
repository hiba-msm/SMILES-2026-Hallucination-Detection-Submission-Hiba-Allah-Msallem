# SMILES-2026 Hallucination Detection Submission

**Applicant:** Hiba-Allah Msallem  
**Repository:** https://github.com/hiba-msm/SMILES-2026-Hallucination-Detection-Submission-Hiba-Allah-Msallem.git

## Reproducibility Instructions

To reproduce my results and generate the `predictions.csv`:

1. **Clone and Setup:**
   ```bash
   git clone https://github.com/hiba-msm/SMILES-2026-Hallucination-Detection-Submission-Hiba-Allah-Msallem.git
   cd SMILES-2026-Hallucination-Detection-Submission-Hiba-Allah-Msallem
   pip install -r requirements.txt
   ```

2. **Environment:**
   The solution was finalized on a **Google Colab T4 GPU**. While it runs on CPU, the GPU environment provides the necessary stability for the MLP ensemble to achieve the reported 72.28% accuracy.

3. **Execution:**
   Run `python solution.py`. This will use the modified `aggregation.py`, `probe.py`, and `splitting.py` to extract features and train the hybrid ensemble.

---

## Final Solution Description

My final approach focuses on a **Hybrid Ensemble** paired with **Topological Trajectory Analysis**. I modified three core components:

### 1. Feature Engineering (`aggregation.py`)
Instead of just taking the last layer's representation, I extracted features from a "trajectory" of 5 layers (0, 6, 12, 18, 24). To enrich this, I implemented several **topological/geometric invariants** across all 25 layers:
- **Magnitude Trajectories**: L2 norms of activations per layer.
- **Representation Drift**: Angular displacement (cosine similarity) between consecutive layers.
- **Confidence Trajectory**: Variance of activations per layer.
These capture how the model's internal "certainty" evolves as it processes the answer.

### 2. Hybrid Ensemble Probe (`probe.py`)
Small datasets (689 samples) are notorious for overfitting neural networks. To solve this, I built a **5-fold Hybrid Ensemble**. In each fold, I alternate between a **2-layer MLP** and a **Random Forest Classifier**. 
- The MLP captures complex non-linear patterns.
- The Random Forest provides stable, tree-based decision boundaries that regularize the ensemble.
This hybrid approach was the single biggest contributor to pushing the Test Accuracy to **72.28%**.

### 3. Accuracy-Driven Optimization (`probe.py` & `splitting.py`)
Since the primary metric is Accuracy, I replaced the default F1-based threshold tuning with a sweep that specifically optimizes for **Validation Accuracy**. I used a Stratified 5-Fold split to ensure the class imbalance (70/30) was preserved in every training phase.

---

## Experiments and Failed Attempts

During development, I tried several approaches that were eventually discarded:

- **Mean-Pooling (Failed):** I tried averaging hidden states across all tokens in the response. This actually dropped AUROC by nearly 7% because the shared prompt tokens acted as noise, diluting the hallucination signal which is concentrated in the final token positions.
- **Logistic Regression (Failed):** While stable, a linear classifier couldn't capture the topological "drift" patterns as well as the MLP/Forest hybrid. It struggled to beat the baseline by more than 0.5%.
- **Deeper MLPs (Overfit):** I tested a 4-layer MLP, but even with high dropout (0.5), it quickly memorized the training set and failed to generalize to the test split.
- **Aggressive PCA (Failed):** Reducing to 64 components discarded too much of the geometric signal. Settling on 192 components provided the best balance for the Random Forest members of the ensemble.
- **Thermal Management (Removed):** I initially limited CPU threads to manage temperatures on my local machine, but removed this for the final submission to ensure maximum performance in the GPU environment.
