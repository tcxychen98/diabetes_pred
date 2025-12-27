# Diabetes Risk Prediction: Advanced Tabular Modeling

This repository contains an end-to-end pipeline for predicting diabetes risk using clinical and lifestyle data. By combining **Gradient Boosted Decision Trees (GBDT)** with **Domain-Specific Medical Feature Engineering**, the model achieves high diagnostic accuracy on tabular health records.

## Key Results
- **Final Model:** 70/30 Weighted Ensemble (CatBoost & XGBoost)
- **Validation Strategy:** 5-Fold Stratified Cross-Validation
- **Mean CV AUC:** ~0.72

## Methodology

### 1. Medical Feature Engineering
Raw health data often obscures critical physiological patterns. We implemented several "Calculated Biomarkers":
*   **AIP Index (Atherogenic Index of Plasma):** Calculated as `log10(Triglycerides / HDL)`. Clinical studies suggest this is a more potent predictor of insulin resistance than individual lipid levels.
*   **Metabolic Syndrome Index:** A composite flag for individuals exhibiting three or more concurrent conditions (High BMI, High BP, and Dyslipidemia).
*   **Sedentary Ratio:** A lifestyle metric quantifying the imbalance between physical activity and screen time.

### 2. The Ensemble Strategy (CatBoost + XGBoost)
Tabular data is best handled by tree-based models. We used an ensemble for two reasons:
*   **CatBoost:** Chosen for its superior handling of categorical variables (like Ethnicity and Education) through its symmetric tree algorithm.
*   **XGBoost:** Used to refine the numeric predictions and capture granular decision boundaries that CatBoost might overlook.
*   **Blending:** A 70/30 weighted average reduces model variance and improves generalization on unseen test data.

### 3. Cross-Validation & Stability
Instead of a simple train/test split, we used **5-Fold Stratified CV**. This ensures every segment of the data is used for both training and validation, providing a statistically robust AUC score and preventing overfitting to "fluke" patterns in the data.

## 🛠️ Project Structure
- `src/features.py`: Encapsulates the medical logic and categorical encoding.
- `src/model.py`: Implements the CV loop and ensemble blending.
- `main.py`: The unified entry point for data ingestion to inference.

## Installation
1. Clone the repo:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
3. Run the script:
    ```bash
    python main.py

## Reference

<a id="1">[[1]](https://doi.org/10.1186/s13098-025-01907-1)</a> 
Liao, Y., Han, Y., Cao, C., Song, H., & Hu, H. (2025). Association between atherogenic index of plasma and risk of type 2 diabetes mellitus and the mediating effect of BMI: a comparative analysis in Chinese and Japanese populations. Diabetology & Metabolic Syndrome, 17(1), 349.

<a id="2">[[2]](https://doi.org/10.1016/j.hnm.2025.200341)</a>
Qasrawi, R., Thwib, S., Issa, G., Ghoush, R. A., & Amro, M. (2025). Type 2 diabetes risk prediction using glycemic control Metrics: A machine learning approach. Human Nutrition & Metabolism, 42, 200341.

<a id="3">[[3]](https://doi.org/10.1049/htl2.12039)</a>
Tasin, I., Nabil, T. U., Islam, S., & Khan, R. (2022). Diabetes prediction using machine learning and explainable AI techniques. Healthcare Technology Letters, 10(1–2), 1–10.