# ProjectChurn
# 📉 Customer Churn Prediction

This project is part of a data science assignment focused on **predicting customer churn** in the telecommunications industry using machine learning techniques. The full pipeline follows the **CRISP-DM methodology**, from business understanding to model evaluation and deployment recommendations.

## 🎯 Business Context

- **Objective**: Reduce customer churn and improve retention using predictive models.
- **Motivation**: Retaining customers is more cost-effective than acquiring new ones.
- **Impact**: A model that identifies clients likely to churn enables the company to take early action and reduce revenue loss.

## 📚 Methodology: CRISP-DM

### 1. Business Understanding

- Defined churn as the key problem affecting telecom revenue.
- Set goals around **improving retention** and building models that **accurately identify customers at risk**.
- Focused on **F1-score and Recall for Churn = Yes** as the most meaningful metrics.

### 2. Data Understanding

- 5000 samples, 18 variables (usage behavior, service plans, customer support).
- Exploratory data analysis (EDA): histograms, boxplots, density plots, correlation analysis.
- Identified class imbalance and key churn-related features.

### 3. Data Preparation

We tested **five different preprocessing pipelines**:

- `d0`: Label encoding for correlation analysis  
- `d1`: Drop NaNs and replace outliers with median  
- `d2`: Fill NaNs and outliers using median  
- `d3`: Fill NaNs, apply Min-Max scaling + log transform (📈 best results)  
- `d4`: Fill NaNs and apply data smoothing  

→ Data was split 80% training / 20% test with **5-fold cross-validation**.

### 4. Modeling

We compared 7 classifiers:

- `Random Forest`
- `XGBoost`
- `SVM`
- `KNN`
- `Decision Tree`
- `Naive Bayes`
- `MLP (Multi-layer Perceptron)`

All models were evaluated across all data versions using:

- **F1-score**
- **Recall**
- **Precision**
- **Accuracy**

### 5. Evaluation & Fine-tuning

Focusing on models that best predict the **Churn = Yes** class:

- `XGBoost` and `Random Forest` showed strong F1 and Precision.
- `SVM` stood out for **highest Recall**, making it ideal for prioritizing at-risk customers.

#### 📌 Final Fine-Tuned Results (best on dataset `d3`)

| Classifier     | Accuracy | Recall | Precision | F1 Score |
|----------------|----------|--------|-----------|----------|
| Random Forest  | 0.962    | 0.816  | 0.906     | 0.858    |
| XGBoost        | 0.962    | 0.766  | 0.956     | 0.850    |
| **SVM**        | 0.860    | **0.894**  | 0.502     | 0.643    |

> 🔍 **Best recall on Churn = Yes:** SVM  
> 🏆 **Best balance (F1/Precision/Recall):** Random Forest & XGBoost

## ✅ Conclusions

- **Best classifier for Churn = Yes**: SVM (TP = 87%)
- **Best overall models**: XGBoost and Random Forest
- **Key insight**: Data preprocessing (log transformation, scaling, outlier handling) greatly impacts model performance.
- **Metric importance**: F1-score and recall are critical in imbalanced churn scenarios. Accuracy alone is misleading.
- **Business impact**: Enables cost-effective, targeted retention campaigns based on predicted churn risk.

## 📁 Dataset

- Public dataset with 5000 entries, 18 features  
- Available: [https://data.world/earino/churn/discuss/churn/3in3tgcr](https://data.world/earino/churn/discuss/churn/3in3tgcr)
  
- The dataset contains 5000 entries and 18 variables related to customer usage behavior and service plans.
- Since the original source is no longer available online, we have included the dataset in this repository for reproducibility.

📌 File: `data/churn.csv`


## ▶️ How to Run

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Open and run
   churn_analysis.ipynb




