import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# --- Phase 1: Synthetic Data Generation ---
def generate_synthetic_data(n=10000):
    np.random.seed(42)
    
    # Basic customer features
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.random.uniform(20, 120, n)
    total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, n)
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n)
    support_calls = np.random.poisson(2, n)
    
    # Margin and Sensitivity (Crucial for Prescriptive logic)
    margin_per_customer = monthly_charges * np.random.uniform(0.2, 0.4, n)
    historical_discount_sensitivity = np.random.uniform(0.0, 1.0, n)
    
    # Logic for Churn Flag (Higher charges, higher calls, month-to-month = higher churn)
    churn_prob = (
        (monthly_charges / 120) * 0.4 + 
        (support_calls / 10) * 0.3 + 
        (np.where(contract_type == 'Month-to-month', 0.3, 0)) - 
        (tenure / 72) * 0.2
    )
    churn_prob = np.clip(churn_prob, 0, 1)
    churn_flag = np.random.binomial(1, churn_prob)
    
    df = pd.DataFrame({
        'CustomerID': range(1000, 1000 + n),
        'Tenure': tenure,
        'Monthly_Charges': monthly_charges,
        'Total_Charges': total_charges,
        'Contract_Type': contract_type,
        'Support_Calls': support_calls,
        'Margin_Per_Customer': margin_per_customer,
        'Historical_Discount_Sensitivity': historical_discount_sensitivity,
        'Churn_Flag': churn_flag
    })
    
    # Encode Contract Type for modeling
    df_encoded = pd.get_dummies(df, columns=['Contract_Type'], drop_first=True)
    return df, df_encoded

# --- Phase 2: Multi-Stage Model ---

# Step A: Segmentation (K-Means)
def segment_customers(df_encoded):
    features = ['Tenure', 'Monthly_Charges', 'Margin_Per_Customer']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_encoded[features])
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_encoded['Segment'] = kmeans.fit_predict(scaled_features)
    
    # Map segments to business labels based on centroids (simplified mapping)
    # 0: VIPs (High Tenure, High Margin)
    # 1: Steady Users (Mid Tenure, Low-ish Charges)
    # 2: At-Risk High-Value (High Charges, Low Tenure)
    # 3: Low-Value Transients (Low Tenure, Low Charges)
    segment_map = {0: 'VIPs', 1: 'Steady Users', 2: 'At-Risk High-Value', 3: 'Low-Value Transients'}
    df_encoded['Segment_Label'] = df_encoded['Segment'].map(segment_map)
    
    return df_encoded

# Step B: Prediction (XGBoost)
def train_prediction_model(df_encoded):
    X = df_encoded.drop(['CustomerID', 'Churn_Flag', 'Segment', 'Segment_Label'], axis=1)
    y = df_encoded['Churn_Flag']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train, y_train)
    
    # Predict probabilities for the entire dataset
    df_encoded['P_Churn'] = model.predict_proba(X)[:, 1]
    
    print("XGBoost Model Trained.")
    print(f"ROC-AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
    
    return df_encoded, model

# Step C: Prescription (Net Retention Gain)
def calculate_prescriptions(df_encoded):
    # CLV Approximation: Margin * 12 (Simplified annual CLV)
    clv = df_encoded['Margin_Per_Customer'] * 12
    
    # Definitions
    cost_a = df_encoded['Monthly_Charges'] * 0.05  # 5% Discount
    cost_b = df_encoded['Monthly_Charges'] * 0.10  # 10% Discount
    
    # Probabilities of staying given incentive (heuristic based on sensitivity)
    # P_Stay = (1 - P_Churn) + (Sensitivity * P_Churn * Uplift)
    # We assume base stay prob is 1 - P_Churn. Incentive increases it.
    uplift_a = 0.2 * df_encoded['Historical_Discount_Sensitivity']
    uplift_b = 0.4 * df_encoded['Historical_Discount_Sensitivity']
    
    p_stay_base = 1 - df_encoded['P_Churn']
    p_stay_a = np.clip(p_stay_base + uplift_a, 0, 1)
    p_stay_b = np.clip(p_stay_base + uplift_b, 0, 1)
    
    # Expected Value Calculation
    ev_none = p_stay_base * clv
    ev_a = (p_stay_a * clv) - cost_a
    ev_b = (p_stay_b * clv) - cost_b
    
    # Decision Logic
    prescriptions = []
    for i in range(len(df_encoded)):
        options = {'None': ev_none[i], '5% Discount': ev_a[i], '10% Discount': ev_b[i]}
        best_option = max(options, key=options.get)
        
        # Only recommend if better than doing nothing
        if best_option == 'None' or options[best_option] <= ev_none[i]:
            prescriptions.append('No Intervention')
        else:
            prescriptions.append(best_option)
            
    df_encoded['Recommended_Incentive'] = prescriptions
    df_encoded['Expected_Value_Optimized'] = np.maximum(ev_none, np.maximum(ev_a, ev_b))
    df_encoded['Expected_Value_Baseline'] = ev_none
    
    return df_encoded

# --- Phase 3: MBA Strategic Output ---
def generate_strategic_output(df_final):
    baseline_churn_cost = (df_final['P_Churn'] * (df_final['Margin_Per_Customer'] * 12)).sum()
    optimized_churn_cost = ( (1 - df_final['Expected_Value_Optimized'] / (df_final['Margin_Per_Customer']*12)) * (df_final['Margin_Per_Customer']*12) ).sum()
    
    # A more direct retention gain summary
    total_clv = (df_final['Margin_Per_Customer'] * 12).sum()
    baseline_retention_val = df_final['Expected_Value_Baseline'].sum()
    optimized_retention_val = df_final['Expected_Value_Optimized'].sum()
    
    summary_table = pd.DataFrame({
        'Metric': ['Total Potential CLV', 'Baseline Expected Retention', 'Optimized Expected Retention', 'Net Profit Gain'],
        'Value': [total_clv, baseline_retention_val, optimized_retention_val, optimized_retention_val - baseline_retention_val]
    })
    
    print("\n--- MBA Strategic Summary ---")
    print(summary_table.to_string(index=False))
    
    print("\n--- Segment Performance ---")
    segment_summary = df_final.groupby('Segment_Label').agg({
        'P_Churn': 'mean',
        'Margin_Per_Customer': 'mean',
        'Recommended_Incentive': lambda x: (x != 'No Intervention').mean()
    }).rename(columns={'Recommended_Incentive': 'Intervention_Rate'})
    print(segment_summary)
    
    return summary_table

# --- Main Pipeline Execution ---
if __name__ == "__main__":
    print("Starting Prescriptive Retention Engine Pipeline...")
    
    # 1. Data Generation
    raw_df, encoded_df = generate_synthetic_data(10000)
    
    # 2. Segmentation
    encoded_df = segment_customers(encoded_df)
    
    # 3. Prediction
    encoded_df, churn_model = train_prediction_model(encoded_df)
    
    # 4. Prescription
    final_df = calculate_prescriptions(encoded_df)
    
    # 5. Output
    summary = generate_strategic_output(final_df)
    
    # Save results
    final_df.to_csv('retention_results.csv', index=False)
    print("\nPipeline Complete. Results saved to 'retention_results.csv'.")

    # --- Conceptual logic for Streamlit Dashboard ---
    """
    CONCEPTUAL STREAMLIT LOGIC:
    1. Sidebar: Sliders for Incentive A (default 5%) and Incentive B (default 10%).
    2. Logic: The `calculate_prescriptions` function re-runs on the fly using slider values.
    3. UI:
       - Metric Cards: "Total Retention Value", "Change from Baseline %".
       - Plot: Histogram of Churn Probability across segments.
       - Table: Interactive dataframe showing Top 10 'At-Risk High-Value' customers for priority outreach.
       - Bar Chart: Expected Value by Segment (Baseline vs. Optimized).
    """
