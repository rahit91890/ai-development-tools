#!/usr/bin/env python3
"""
Streamlit Model Dashboard

Interactive dashboard for visualizing ML model performance,
experiments, and metrics from MLflow and W&B.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI Model Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("🤖 AI Model Performance Dashboard")
st.markdown("""
    Real-time monitoring and visualization of ML model experiments,
    metrics, and performance tracking.
""")

# Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    
    # Experiment selector
    experiment = st.selectbox(
        "Select Experiment",
        ["Random Forest", "Gradient Boosting", "Neural Network"]
    )
    
    # Date range
    st.subheader("Date Range")
    start_date = st.date_input("Start Date")
    end_date = st.date_input("End Date")
    
    # Metric selector
    metrics = st.multiselect(
        "Select Metrics",
        ["Accuracy", "Precision", "Recall", "F1 Score"],
        default=["Accuracy", "F1 Score"]
    )
    
    st.markdown("---")
    st.info("Connect to MLflow or W&B for live data")

# Main dashboard area
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Metrics", "🔍 Analysis"])

with tab1:
    st.header("Model Performance Overview")
    
    # KPI metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Best Accuracy",
            value="94.5%",
            delta="+2.3%"
        )
    
    with col2:
        st.metric(
            label="Total Runs",
            value="156",
            delta="+12"
        )
    
    with col3:
        st.metric(
            label="Avg Training Time",
            value="45.2s",
            delta="-5.3s"
        )
    
    with col4:
        st.metric(
            label="Active Models",
            value="8",
            delta="0"
        )
    
    # Sample data for visualization
    st.subheader("Recent Experiment Results")
    
    # Generate sample data
    dates = pd.date_range(start='2025-01-01', periods=20, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Accuracy': np.random.uniform(0.85, 0.95, 20),
        'F1 Score': np.random.uniform(0.82, 0.93, 20),
        'Precision': np.random.uniform(0.83, 0.94, 20)
    })
    
    # Line chart
    fig = px.line(
        data, 
        x='Date', 
        y=['Accuracy', 'F1 Score', 'Precision'],
        title='Model Performance Over Time'
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Detailed Metrics Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Scatter plot
        scatter_data = pd.DataFrame({
            'Training Time': np.random.uniform(30, 60, 50),
            'Accuracy': np.random.uniform(0.85, 0.95, 50),
            'Model': np.random.choice(['RF', 'GB', 'NN'], 50)
        })
        
        fig_scatter = px.scatter(
            scatter_data,
            x='Training Time',
            y='Accuracy',
            color='Model',
            title='Training Time vs Accuracy'
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with col2:
        # Bar chart
        bar_data = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Neural Net'],
            'Accuracy': [0.92, 0.94, 0.91]
        })
        
        fig_bar = px.bar(
            bar_data,
            x='Model',
            y='Accuracy',
            title='Model Comparison'
        )
        st.plotly_chart(fig_bar, use_container_width=True)

with tab3:
    st.header("Deep Dive Analysis")
    
    st.subheader("Hyperparameter Impact")
    
    # Sample hyperparameter data
    param_data = pd.DataFrame({
        'n_estimators': np.random.randint(50, 200, 30),
        'max_depth': np.random.randint(3, 10, 30),
        'accuracy': np.random.uniform(0.85, 0.95, 30)
    })
    
    fig_3d = px.scatter_3d(
        param_data,
        x='n_estimators',
        y='max_depth',
        z='accuracy',
        color='accuracy',
        title='Hyperparameter Space Exploration'
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Feature importance
    st.subheader("Feature Importance")
    feature_data = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(10)],
        'Importance': np.random.uniform(0.01, 0.15, 10)
    }).sort_values('Importance', ascending=True)
    
    fig_features = px.bar(
        feature_data,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Feature Importances'
    )
    st.plotly_chart(fig_features, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    **Note:** This is a demo dashboard. Connect to your MLflow or W&B instance 
    for real-time experiment tracking and metrics.
    
    💡 To run: `streamlit run model_dashboard.py`
""")
