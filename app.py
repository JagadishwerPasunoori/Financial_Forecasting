import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import tensorflow as tf
from utils.data_loader import load_data

def main():
    st.title("Financial Forecasting App")
    
    # Load models
    ml_model = load('models/ml_model.pkl')
    dl_model = tf.keras.models.load_model('models/dl_model.h5')
    
    # Sidebar inputs
    with st.sidebar:
        st.header("Input Parameters")
        year = st.number_input('Year', min_value=2001, max_value=2030, value=2024)
        revenue = st.number_input('Total Revenue', min_value=0)
        net_income = st.number_input('Net Income', min_value=0)
    
    # Create input array
    input_data = pd.DataFrame([[year, revenue, net_income]], 
                            columns=['Year', 'Total Revenue', 'Net Income'])
    
    # Make predictions
    if st.button('Predict'):
        ml_pred = ml_model.predict(input_data)
        dl_pred = dl_model.predict(input_data)
        
        # Get original columns
        df = load_data('data/financial_data.csv')
        targets = [col for col in df.columns if col not in 
                 ['Year', 'Total Revenue', 'Net Income']]
        
        # Create result DataFrames
        ml_results = pd.DataFrame(ml_pred, columns=targets)
        dl_results = pd.DataFrame(dl_pred, columns=targets)
        
        # Display results
        st.subheader("Machine Learning Forecating")
        st.dataframe(ml_results.style.format("{:,.2f}"))
        
        st.subheader("Deep Learning Forecasting")
        st.dataframe(dl_results.style.format("{:,.2f}"))

if __name__ == '__main__':
    main()