import os
import uuid
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
from pandas import DataFrame

PLOTS_DIR = "plots"
if not os.path.exists(PLOTS_DIR):
    os.makedirs(PLOTS_DIR)

def split_features(df: DataFrame):
    numerical_data = [col for col in df if df[col].dtype in ["int64", "float64"]]
    categorical_data = [col for col in df if df[col].dtype == "object"]
    return numerical_data, categorical_data

def save_plot(filename_prefix="plot"):
    filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    plt.savefig(filepath, bbox_inches="tight")
    plt.close()
    return filepath

def line_plot(df: DataFrame, x_feature: str, y_feature: str, hue: str | None = None, size: str | None = None, style: str | None = None):
    numerical_data, categorical_data = split_features(df)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x=x_feature,y=y_feature, hue=hue, size=size, style=style)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    return save_plot("line")

def scatter_plot(df: DataFrame, x_feature: str, y_feature: str, hue: str | None = None, size: str | None = None, style: str | None = None):
    numerical_data, categorical_data = split_features(df)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_feature, y=y_feature, hue=hue, size=size, style=style)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    return save_plot("scatter")

def box_plot(df: DataFrame, x_feature: str, y_feature: str, hue: str | None = None):
    numerical_data, categorical_data = split_features(df)
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x=x_feature, y=y_feature, hue=hue)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    return save_plot("box")

def plot_distribution(df: DataFrame, column: str, title: str = ""):
    plt.figure(figsize=(10, 6))
    if df[column].dtype in ["int64", "float64"]:
        sns.histplot(df[column], kde=True)
    else:
        sns.countplot(x=df[column])
    plt.title(title)
    return save_plot("distribution")

def plot_confusion_matrix_heatmap(cm, title: str = "Confusion Matrix"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    return save_plot("confusion_matrix")

def plot_clusters(X, labels, title="Cluster Visualization"):
    from sklearn.decomposition import PCA
    import numpy as np
    
    plt.figure(figsize=(10, 6))
    
    
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_vis = pca.fit_transform(X)
        xlabel, ylabel = "PCA 1", "PCA 2"
    else:
        X_vis = X.values if hasattr(X, 'values') else X
        xlabel = X.columns[0] if hasattr(X, 'columns') and len(X.columns) > 0 else "Feature 1"
        ylabel = X.columns[1] if hasattr(X, 'columns') and len(X.columns) > 1 else "Feature 2"
    
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=labels, palette="viridis", s=100, alpha=0.7)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    return save_plot("clusters")
