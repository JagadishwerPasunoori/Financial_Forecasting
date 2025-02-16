import pandas as pd

def load_data(file_path):
    """Load and preprocess financial data"""
    df = pd.read_csv(file_path)
    transposed = df.set_index('Account').T.reset_index()
    transposed.columns.name = None
    transposed = transposed.rename(columns={'index': 'Year'})
    
    # Convert all numeric columns to float
    numeric_cols = transposed.columns[transposed.columns != 'Year']
    transposed[numeric_cols] = transposed[numeric_cols].apply(
        pd.to_numeric, errors='coerce'
    )
    
    # Convert Year to integer
    transposed['Year'] = transposed['Year'].astype(int)
    
    return transposed