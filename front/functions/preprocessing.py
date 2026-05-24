import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer, PolynomialFeatures
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split

def one_hot_encoding(df, column_name):
    encoder = OneHotEncoder(sparse_output=False)
    encoded = encoder.fit_transform(df[[column_name]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column_name]))
    df = pd.concat([df.drop(columns=[column_name]), encoded_df], axis=1)
    return df

def label_encoding(df, column_name):
    le = LabelEncoder()
    df[column_name] = le.fit_transform(df[column_name].astype(str))
    return df

def standard_scaler(df, column_name):
    if df.empty: return df
    scaler = StandardScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    return df

def min_max_scaler(df, column_name):
    if df.empty: return df
    scaler = MinMaxScaler()
    df[column_name] = scaler.fit_transform(df[[column_name]])
    return df

def apply_simple_imputer(df, column_name, strategy='mean'):
    imputer = SimpleImputer(strategy=strategy)
    df[column_name] = imputer.fit_transform(df[[column_name]]).ravel()
    return df

def apply_knn_imputer(df, n_neighbors=3):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    cols = df.columns
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols)
    return df_imputed

def apply_iterative_imputer(df):
    imputer = IterativeImputer()
    cols = df.columns
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=cols)
    return df_imputed

def log_transformation(df, column_name):
    df["log_" + column_name] = np.log1p(df[column_name])
    return df

def boxcox_transformation(df, column_name):
    data = df[column_name] - df[column_name].min() + 1
    df["boxcox_" + column_name], _ = stats.boxcox(data)
    return df

def power_transformation(df, column_name):
    pt = PowerTransformer(method='yeo-johnson')
    df["power_" + column_name] = pt.fit_transform(df[[column_name]])
    return df

def apply_pca(df, n_components=2):
    if df.empty: return df
    if len(df) < n_components:
         n_components = len(df)
    if n_components < 1: return df
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    cols = [f'PC{i+1}' for i in range(min(n_components, X_pca.shape[1]))]
    return pd.DataFrame(data=X_pca, columns=cols)

def apply_rfe(df, target_col, n_features_to_select=5):
    if df.empty: return df
    X = df.drop(columns=[target_col])
    y = df[target_col]
    if X.empty: return df
    
    n_features_to_select = min(n_features_to_select, X.shape[1])
    
    model = RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42)
    rfe = RFE(model, n_features_to_select=n_features_to_select, step=2)
    X_selected = rfe.fit_transform(X, y)
    selected_cols = X.columns[rfe.support_]
    return pd.concat([pd.DataFrame(X_selected, columns=selected_cols), y.reset_index(drop=True)], axis=1)

def handle_imbalanced_data(df, target, method='SMOTE'):  
    if df[target].isnull().any():
        raise ValueError(f"Target column '{target}' contains missing values. Please handle them first.")
    
    
    X = df.drop(target, axis=1)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        raise ValueError("No numeric features found for balancing. Please encode categorical features first.")
    
    X_numeric = X[numeric_cols]
    y = df[target]
    
    if X_numeric.isnull().any().any():
        raise ValueError("Features contain missing values. Please handle them first.")

    if method == 'SMOTE':
        class_counts = y.value_counts()
        min_samples = class_counts.min()
        
        if min_samples < 2:
            sampler = RandomOverSampler(random_state=42)
        else:
            k = min(5, min_samples - 1)
            sampler = SMOTE(k_neighbors=k, random_state=42)
    elif method == 'Undersampling':
        sampler = RandomUnderSampler(random_state=42)
    else:
        return df
    
    X_res, y_res = sampler.fit_resample(X_numeric, y)
    
    # Reconstruct DF
    df_res = pd.concat([pd.DataFrame(X_res, columns=numeric_cols), pd.Series(y_res, name=target)], axis=1)
    return df_res
