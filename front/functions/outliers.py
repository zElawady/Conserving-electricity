import pandas as pd
import numpy as np
from scipy import stats

def IQR(df: pd.DataFrame):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        iqr_val = Q3 - Q1
        lower_bound = Q1 - 1.5 * iqr_val
        upper_bound = Q3 + 1.5 * iqr_val
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

def Zscore(df: pd.DataFrame):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        df_clean['zscore'] = stats.zscore(df_clean[col].fillna(df_clean[col].mean()))
        df_clean = df_clean[np.abs(df_clean['zscore']) <= 3]
        df_clean = df_clean.drop(columns=['zscore'])
    return df_clean

def Winsorization_Method(df: pd.DataFrame):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        lower_bound = df_clean[col].quantile(0.05)
        upper_bound = df_clean[col].quantile(0.95)
        df_clean[col] = np.where(df_clean[col] < lower_bound, lower_bound, df_clean[col])
        df_clean[col] = np.where(df_clean[col] > upper_bound, upper_bound, df_clean[col])
    return df_clean

def Clipping_Method(df: pd.DataFrame):
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=['number'])
    all_values = numeric_cols.values.flatten()
    all_values = all_values[~np.isnan(all_values)]
    global_mean = np.mean(all_values)
    global_std = np.std(all_values)
    lower_bound = global_mean - 3 * global_std
    upper_bound = global_mean + 3 * global_std
    for col in numeric_cols.columns:
        df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    return df_clean

# Aliasing for compatibility
clipping = Clipping_Method
winsorize = Winsorization_Method
