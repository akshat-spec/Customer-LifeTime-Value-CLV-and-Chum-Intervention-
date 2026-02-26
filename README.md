# Customer Lifetime Value (CLV) & Churn Intervention

## 🚀 The Prescriptive Retention Engine
**Optimizing CLV through Targeted Churn Intervention**

This project implementes an end-to-end pipeline for predicting customer churn and prescribing the most profitable intervention strategies. It follows a multi-stage modeling process designed for business impact.

### 📊 Project Overview
The core objective is to move beyond simple prediction (who will leave?) to **prescription** (what should we do about it?). By combining machine learning with financial logic, this engine recommends incentives (like discounts) only when they result in a net retention gain for the company.

### 🛠️ Key Features
- **Synthetic Data Generation**: Simulates realistic customer data with features like tenure, charges, contract type, and support calls.
- **Customer Segmentation (K-Means)**: Groups customers into VIPs, Steady Users, At-Risk High-Value, and Low-Value Transients.
- **Churn Prediction (XGBoost)**: Predicts the probability of churn for each customer with high accuracy (ROC-AUC 0.90+).
- **Prescriptive Analytics**: Calculates the Expected Value (EV) of different intervention strategies (No action, 5% discount, 10% discount) to maximize retained CLV.
- **Interactive Dashboard**: A Streamlit-based interface for business stakeholders to visualize risk, segments, and financial impact.

### 📁 File Structure
- `retention_engine.py`: The main pipeline (Data Generation -> Segmentation -> Prediction -> Prescription).
- `dashboard.py`: Streamlit dashboard for visualizing results.
- `retention_results.csv`: The output file containing all predictions and recommended interventions.

### 🚀 Getting Started

#### Prerequisites
- Python 3.8+
- Required libraries: `pandas`, `numpy`, `xgboost`, `scikit-learn`, `streamlit`, `plotly`

#### Installation
```bash
pip install pandas numpy xgboost scikit-learn streamlit plotly
```

#### Running the Project
1. **Execute the Pipeline**:
   ```bash
   python retention_engine.py
   ```
   This will train the models and generate `retention_results.csv`.

2. **Launch the Dashboard**:
   ```bash
   streamlit run dashboard.py
   ```

### 📈 Business Impact
The dashboard provides a clear view of:
- **Total Potential Value**: Maximum possible CLV.
- **Baseline vs. Optimized Retention**: Financial gain from using the AI-driven intervention strategy.
- **Priority Outreach**: Identifies the top at-risk high-value customers for immediate contact.

---

