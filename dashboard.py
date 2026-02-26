import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Prescriptive Retention Engine", layout="wide")

# Custom CSS for Cyberpunk/Modern look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .explanation {
        background: rgba(0, 150, 255, 0.1);
        padding: 15px;
        border-left: 5px solid #0096ff;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Helper to load data
@st.cache_data
def load_data():
    file_path = 'retention_results.csv'
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

df = load_data()

if df is None:
    st.error("Data file 'retention_results.csv' not found. Please run the backend script first.")
else:
    # --- HEADER ---
    st.title("🚀 The Prescriptive Retention Engine")
    st.subheader("Optimizing Customer Lifetime Value (CLV) through Targeted Churn Intervention")
    
    with st.expander("ℹ️ What is this dashboard? (Easy Mode)"):
        st.write("""
        This dashboard shows how we can use **Artificial Intelligence** to keep our customers from leaving.
        - **Churn Prediction**: We predict who is likely to cancel their service.
        - **Segmentation**: We group customers into 4 types so we know who is a 'VIP' and who is 'At-Risk'.
        - **Prescription**: The AI tells us exactly what discount to give each person to keep them, but *only* if it makes the company more money.
        """)

    # --- TOP LEVEL METRICS ---
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    total_clv = (df['Margin_Per_Customer'] * 12).sum()
    baseline_ev = df['Expected_Value_Baseline'].sum()
    optimized_ev = df['Expected_Value_Optimized'].sum()
    profit_gain = optimized_ev - baseline_ev
    
    with col1:
        st.metric("Total Potential Value", f"${total_clv/1e6:.2f}M", help="The maximum money we could make if 100% of customers stayed.")
    with col2:
        st.metric("Baseline Retention", f"${baseline_ev/1e6:.2f}M", help="Expected earnings if we do nothing and let people leave naturally.")
    with col3:
        st.metric("Optimized Retention", f"${optimized_ev/1e6:.2f}M", delta=f"{profit_gain/1e3:.1f}k", help="Expected earnings after AI-driven discounts.")
    with col4:
        roi = (profit_gain / (optimized_ev - profit_gain)) * 100
        st.metric("Retention ROI", f"{roi:.1f}%", help="Return on investment for the discount strategy.")

    # --- SECTION 1: CUSTOMER SEGMENTS ---
    st.markdown("### 👥 Customer Insights")
    c1, c2 = st.columns([1, 1])
    
    with c1:
        st.write("**Who are our customers?**")
        segment_counts = df['Segment_Label'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        fig_pie = px.pie(segment_counts, values='Count', names='Segment', 
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
        fig_pie.update_layout(template="plotly_dark", margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.markdown("""
        <div class="explanation">
        <b>Simplified Meaning:</b> This pie chart shows our customer mix. 
        <b>VIPs</b> are our best customers. <b>Transient</b> customers are here today, gone tomorrow.
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.write("**Churn Risk by Segment**")
        risk_data = df.groupby('Segment_Label')['P_Churn'].mean().reset_index()
        fig_bar = px.bar(risk_data, x='Segment_Label', y='P_Churn', 
                         labels={'P_Churn': 'Avg Churn Risk (%)', 'Segment_Label': 'Segment'},
                         color='P_Churn', color_continuous_scale='RdYlGn_r')
        fig_bar.update_layout(template="plotly_dark")
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.markdown("""
        <div class="explanation">
        <b>Simplified Meaning:</b> This shows which groups are most likely to leave. 
        Higher bars (Red) mean we are losing those customers fast!
        </div>
        """, unsafe_allow_html=True)

    # --- SECTION 2: PROFITABILITY IMPACT ---
    st.markdown("### 💰 Financial Impact of AI")
    
    total_impact = pd.DataFrame({
        'Scenario': ['No AI (Baseline)', 'With AI (Optimized)'],
        'Expected Value': [baseline_ev, optimized_ev]
    })
    
    fig_impact = px.bar(total_impact, x='Scenario', y='Expected Value', 
                        color='Scenario', text_auto='.3s')
    fig_impact.update_layout(template="plotly_dark", showlegend=False)
    st.plotly_chart(fig_impact, use_container_width=True)
    
    st.markdown(f"""
    <div class="explanation">
    <b>The Bottom Line:</b> By using AI to target customers with the right incentive, 
    we increase our expected revenue from **${baseline_ev:,.0f}** to **${optimized_ev:,.0f}**. 
    That's a free gain of **${profit_gain:,.0f}**!
    </div>
    """, unsafe_allow_html=True)

    # --- SECTION 3: INTERVENTION STRATEGY ---
    st.markdown("### 🎯 AI Recommendations")
    
    strategy_counts = df['Recommended_Incentive'].value_counts().reset_index()
    strategy_counts.columns = ['Incentive Type', 'Customer Count']
    
    fig_strat = px.bar(strategy_counts, y='Incentive Type', x='Customer Count', 
                       orientation='h', color='Incentive Type',
                       color_discrete_map={'No Intervention': '#636EFA', '5% Discount': '#00CC96', '10% Discount': '#EF553B'})
    fig_strat.update_layout(template="plotly_dark")
    st.plotly_chart(fig_strat, use_container_width=True)

    st.markdown("""
    <div class="explanation">
    <b>Easy Explanation:</b> The AI doesn't give a discount to everyone (that would be expensive!). 
    It only gives a <b>5% or 10% discount</b> to people it thinks will stay longer because of it. 
    Everyone else gets "No Intervention" to save us costs.
    </div>
    """, unsafe_allow_html=True)

    # --- RAW DATA PREVIEW ---
    st.markdown("### 📋 Top At-Risk Customers to Contact")
    at_risk = df[df['P_Churn'] > 0.5].sort_values(by='Margin_Per_Customer', ascending=False)
    st.dataframe(at_risk[['CustomerID', 'Segment_Label', 'P_Churn', 'Recommended_Incentive', 'Margin_Per_Customer']].head(10))
    
    st.download_button(
        "Download Full AI Retention List (CSV)",
        df.to_csv(index=False),
        "ai_retention_list.csv",
        "text/csv"
    )

    st.markdown("---")
    st.caption("Developed by Lead AI Solutions Architect | MBA Capstone: Prescriptive Retention Engine")
