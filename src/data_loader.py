# Copyright (c) 2024 [Your Name]
# SPDX-License-Identifier: Apache-2.0
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def get_dataloader(dataset_name="uci"):
    """
    Factory function to load and clean medical datasets.
    Supports: 'uci' (Heart Disease) and 'pima' (Diabetes)
    """
    if dataset_name == "uci":
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        df = pd.read_csv(url, names=cols, na_values='?')
        
        # Numeric conversion & Median fill
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(df.median(), inplace=True)
        
        # Binary target: 0 (No Disease) vs 1 (Disease)
        df['target'] = (df['num'] > 0).astype(int)
        X = df.drop(['num', 'target'], axis=1).values
        y = df['target'].values

    elif dataset_name == "pima":
        url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
        cols = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'target']
        df = pd.read_csv(url, names=cols)
        
        # Pima specific: Replace 0 with NaN for biological markers then fill
        cols_to_fix = ['plas', 'pres', 'skin', 'test', 'mass']
        df[cols_to_fix] = df[cols_to_fix].replace(0, np.nan)
        df.fillna(df.median(), inplace=True)
        
        X = df.drop('target', axis=1).values
        y = df['target'].values
    
    else:
        raise ValueError("Invalid dataset name. Choose 'uci' or 'pima'.")

    # Scaling is crucial for FHE Quantization stability
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
