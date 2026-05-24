from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import pandas as pd
import io
import os
import sys

# Add parent directory to path to import functions
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from functions import visualization, preprocessing, outliers, models

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for simplicity in a student project
state = {
    "df": None,
    "X_train": None,
    "X_test": None,
    "y_train": None,
    "y_test": None,
    "model": None,
}


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        elif file.filename.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        state["df"] = df

        num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        missing_report = df.isnull().sum().to_dict()

        # Guess Target (Assuming last column)
        last_col = df.columns[-1]
        is_classification = (
            df[last_col].dtype == "object" or df[last_col].nunique() < 20
        )
        task_guess = "Classification" if is_classification else "Regression"

        return {
            "filename": file.filename,
            "columns": df.columns.tolist(),
            "numerical_columns": num_cols,
            "categorical_columns": cat_cols,
            "missing_values": missing_report,
            "task_guess": task_guess,
            "guessed_target": last_col,
            "shape": df.shape,
            "head": df.head(10).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/detect_task")
def detect_task(target: str):
    if state["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    df = state["df"]
    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Column not found")
    is_classification = df[target].dtype == "object" or df[target].nunique() < 20
    return {"task_type": "Classification" if is_classification else "Regression"}


@app.get("/api/visualization/{plot_type}")
def get_visualization(plot_type: str, x: str, y: str = None, hue: str = None):
    if state["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    df = state["df"]
    try:
        if plot_type == "line":
            path = visualization.line_plot(df, x, y, hue)
        elif plot_type == "scatter":
            path = visualization.scatter_plot(df, x, y, hue)
        elif plot_type == "box":
            path = visualization.box_plot(df, x, y, hue)
        else:
            raise HTTPException(status_code=400, detail="Unknown plot type")

        # Ensure path uses forward slashes for URLs
        url_path = path.replace("\\", "/")
        return {"plot_url": f"/{url_path}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/preprocess")
async def preprocess_data(config: dict):
    if state["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")
    df = state["df"]
    action = config.get("action")
    col = config.get("column")

    # Store before state
    before_df = df.copy()
    before_shape = df.shape
    before_missing = int(df.isnull().sum().sum())

    # Generate before plot (if column is provided and exists, else pick first numeric)
    plot_col = (
        col
        if col and col in df.columns
        else df.select_dtypes(include=["int64", "float64"]).columns[0]
    )
    try:
        b_path = visualization.plot_distribution(df, plot_col, title=f"Before {action}")
        before_img_url = "/" + b_path.replace("\\", "/")
    except:
        before_img_url = None

    try:
        # Encoding
        if action == "one_hot":
            df = preprocessing.one_hot_encoding(df, col)
        elif action == "label_encode":
            df = preprocessing.label_encoding(df, col)

        # Scaling
        elif action == "standard_scaler":
            df = preprocessing.standard_scaler(df, col)
        elif action == "min_max_scaler":
            df = preprocessing.min_max_scaler(df, col)

        # Missing
        elif action == "simple_impute":
            df = preprocessing.apply_simple_imputer(
                df, col, config.get("strategy", "mean")
            )
        elif action == "knn_impute":
            df = preprocessing.apply_knn_imputer(
                df.select_dtypes(include=["int64", "float64"])
            )
        elif action == "iterative_impute":
            df = preprocessing.apply_iterative_imputer(
                df.select_dtypes(include=["int64", "float64"])
            )

        # Outliers
        elif action == "outlier_iqr":
            temp_df = outliers.IQR(df)
            if temp_df.empty:
                raise ValueError("Applying IQR would remove all samples. Operation cancelled.")
            df = temp_df
        elif action == "outlier_zscore":
            temp_df = outliers.Zscore(df)
            if temp_df.empty:
                raise ValueError("Applying Z-score would remove all samples. Operation cancelled.")
            df = temp_df
        elif action == "outlier_winsor":
            df = outliers.winsorize(df)
        elif action == "outlier_clip":
            df = outliers.clipping(df)

        # Transformation
        elif action == "log_transform":
            df = preprocessing.log_transformation(df, col)
        elif action == "boxcox_transform":
            df = preprocessing.boxcox_transformation(df, col)
        elif action == "power_transform":
            df = preprocessing.power_transformation(df, col)
        elif action == "poly_features":
            pass

        # Reduction & Selection
        elif action == "pca":
            df = preprocessing.apply_pca(
                df.select_dtypes(include=["int64", "float64"]),
                config.get("n_components", 2),
            )
        elif action == "rfe":
            if col in df.columns:
                df = preprocessing.apply_rfe(
                    df.select_dtypes(include=["int64", "float64"]).join(
                        df[col] if df[col].dtype == "object" else pd.DataFrame()
                    ),
                    col,
                    5,
                )

        # Balancing
        elif action == "smote":
            df = preprocessing.handle_imbalanced_data(df, col, "SMOTE")
        elif action == "undersampling":
            df = preprocessing.handle_imbalanced_data(df, col, "Undersampling")

        if df.empty:
            raise ValueError("Processing resulted in an empty dataset. Please check your parameters.")

        state["df"] = df

        # Store after state
        after_shape = df.shape
        after_missing = int(df.isnull().sum().sum())

        return {
            "message": f"Applied {action} successfully",
            "columns": df.columns.tolist(),
            "before_shape": before_shape,
            "after_shape": after_shape,
            "before_missing": before_missing,
            "after_missing": after_missing,
            "before_head": before_df.head(5).fillna("").to_dict(orient="records"),
            "after_head": df.head(5).fillna("").to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train_all")
def train_all_models(config: dict):
    if state["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    target = config.get("target")
    test_size = float(config.get("test_size", 0.2))
    df = state["df"]

    if target not in df.columns:
        raise HTTPException(status_code=400, detail="Target not found")

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer

        is_classification_auto = df[target].dtype == "object" or df[target].nunique() < 20
        task_type = config.get("task_type", "Classification" if is_classification_auto else "Regression")
        is_classification = (task_type == "Classification")

        # Support unsupervised learning (no target) for kmeans
        is_unsupervised = (task_type == "Clustering")

        # Automatic Feature Handling: Encode categorical features (strings like 'female')
        X_raw = df if is_unsupervised else df.drop(columns=[target])
        X = pd.get_dummies(X_raw, drop_first=True)
        y = pd.Series([0]*len(df), name="dummy") if is_unsupervised else df[target]

        # Automatic Target Handling: Label encode if classification or if target is string
        if not is_unsupervised and (is_classification or y.dtype == "object"):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            y = pd.Series(y, name=target)

        # AUTO CLEANING
        if is_unsupervised:
            X = X.dropna()
            y = y.iloc[X.index]
        else:
            combined = pd.concat([X, y], axis=1).dropna()
            X = combined.drop(columns=[target])
            y = combined[target]
        
        if X.isnull().values.any():
            X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)

        from sklearn.preprocessing import StandardScaler
        X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        state["X_train"], state["X_test"] = X_train, X_test
        state["y_train"], state["y_test"] = y_train, y_test

        algos = []
        if task_type == "Clustering":
            algos = ["kmeans"]
        elif is_classification:
            algos = ["DecisionTree", "LogisticRegression", "SVM", "RandomForest", "KNN", "NaiveBayes", "ANN"]
        else:
            algos = ["LinearRegression", "DecisionTree", "RandomForest", "SVM", "KNN"]

        for algo in algos:
            try:
                if algo == "kmeans":
                    model = models.kmeans_fit(X_train)
                elif algo == "DecisionTree":
                    model = models.dt_fit(X_train, y_train, task=task_type)
                elif algo == "LogisticRegression":
                    model = models.logistic_regression_fit(X_train, y_train)
                elif algo == "SVM":
                    model = models.svm_fit(X_train, y_train, task=task_type)
                elif algo == "RandomForest":
                    model = models.rf_fit(X_train, y_train, task=task_type)
                elif algo == "KNN":
                    model = models.knn_fit(X_train, y_train, task=task_type)
                elif algo == "NaiveBayes":
                    model = models.naive_bayes_fit(X_train, y_train)
                elif algo == "ANN":
                    model = models.ann_fit(X_train, y_train, task=task_type)
                elif algo == "LinearRegression":
                    model = models.linreg_fit(X_train, y_train)

                cm_url = None
                if algo == "kmeans":
                    metrics = {"inertia": float(model.inertia_)}
                    path = visualization.plot_clusters(X_test, model.predict(X_test))
                    cm_url = "/" + path.replace("\\", "/")
                elif is_classification:
                    metrics = models.evaluate_classification(model, X_test, y_test)
                    metrics["precision"] = metrics["classification_report"][
                        "macro avg"
                    ]["precision"]
                    metrics["recall"] = metrics["classification_report"]["macro avg"][
                        "recall"
                    ]
                    metrics["f1-score"] = metrics["classification_report"]["macro avg"][
                        "f1-score"
                    ]
                    path = visualization.plot_confusion_matrix_heatmap(
                        metrics["confusion_matrix"], title=f"CM: {algo}"
                    )
                    cm_url = "/" + path.replace("\\", "/")
                else:
                    metrics = models.evaluate_regression(model, X_test, y_test)

                results.append(
                    {"algorithm": algo, "metrics": metrics, "cm_url": cm_url}
                )
            except:
                pass

        return {"task_type": task_type, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


CLASSIFICATION_MODELS = {
    "DecisionTree",
    "LogisticRegression",
    "SVM",
    "RandomForest",
    "KNN",
    "NaiveBayes",
    "ANN",
}
REGRESSION_MODELS = {"LinearRegression", "DecisionTree", "RandomForest", "SVM", "KNN"}


@app.post("/api/train_single")
def train_single_model(config: dict):
    if state["df"] is None:
        raise HTTPException(status_code=400, detail="No data uploaded")

    target, algo = config.get("target"), config.get("algo")
    test_size = float(config.get("test_size", 0.2))
    df = state["df"]

    # Support unsupervised learning (no target) for kmeans
    is_unsupervised = (algo == "kmeans") and not target

    try:
        from sklearn.model_selection import train_test_split
        from sklearn.impute import SimpleImputer

        # Task detection (skip if unsupervised)
        is_classification = False
        task_type = "Clustering" if is_unsupervised else "Regression"
        
        if not is_unsupervised:
            is_classification_auto = df[target].dtype == "object" or df[target].nunique() < 20
            task_type = config.get("task_type", "Classification" if is_classification_auto else "Regression")
            is_classification = (task_type == "Classification")

        # Automatic Feature Handling: Encode categorical features (strings like 'female')
        X_raw = df if is_unsupervised else df.drop(columns=[target])
        X = pd.get_dummies(X_raw, drop_first=True)
        y = pd.Series([0]*len(df), name="dummy") if is_unsupervised else df[target]

        # Automatic Target Handling: Label encode if classification or if target is string
        if not is_unsupervised and (is_classification or y.dtype == "object"):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y = le.fit_transform(y.astype(str))
            y = pd.Series(y, name=target)

        # AUTO CLEANING
        if is_unsupervised:
            X = X.dropna()
            y = y.iloc[X.index]
        else:
            combined = pd.concat([X, y], axis=1).dropna()
            X = combined.drop(columns=[target])
            y = combined[target]
        
        if X.isnull().values.any():
            X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        if algo == "DecisionTree":
            model = models.dt_fit(X_train, y_train, task=task_type)
        elif algo == "LogisticRegression":
            model = models.logistic_regression_fit(X_train, y_train)
        elif algo == "SVM":
            model = models.svm_fit(X_train, y_train, task=task_type)
        elif algo == "RandomForest":
            model = models.rf_fit(X_train, y_train, task=task_type)
        elif algo == "KNN":
            model = models.knn_fit(X_train, y_train, task=task_type)
        elif algo == "NaiveBayes":
            model = models.naive_bayes_fit(X_train, y_train)
        elif algo == "ANN":
            model = models.ann_fit(X_train, y_train, task=task_type)
        elif algo == "LinearRegression":
            model = models.linreg_fit(X_train, y_train)
        elif algo == "kmeans":
            model = models.kmeans_fit(X_train)

        cm_url = None
        if algo == "kmeans":
            # Clustering evaluation (Inertia, etc.)
            metrics = {"inertia": float(model.inertia_)}
            # Cluster Visualization
            path = visualization.plot_clusters(X_test, model.predict(X_test))
            cm_url = "/" + path.replace("\\", "/")
        elif is_classification:
            metrics = models.evaluate_classification(model, X_test, y_test)
            metrics["precision"] = metrics["classification_report"]["macro avg"][
                "precision"
            ]
            metrics["recall"] = metrics["classification_report"]["macro avg"]["recall"]
            metrics["f1-score"] = metrics["classification_report"]["macro avg"][
                "f1-score"
            ]
            path = visualization.plot_confusion_matrix_heatmap(
                metrics["confusion_matrix"], title=f"CM: {algo}"
            )
            cm_url = "/" + path.replace("\\", "/")
        else:
            metrics = models.evaluate_regression(model, X_test, y_test)

        return {
            "task_type": task_type,
            "algorithm": algo,
            "metrics": metrics,
            "cm_url": cm_url,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


plots_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"
)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

front_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "front"
)
try:
    app.mount("/", StaticFiles(directory=front_dir, html=True), name="front")

except Exception as e:
    import traceback
    traceback.print_exc()
    raise HTTPException(status_code=500, detail=str(e))
plots_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "plots"
)
if not os.path.exists(plots_dir):
    os.makedirs(plots_dir)
app.mount("/plots", StaticFiles(directory=plots_dir), name="plots")

front_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "front"
)
app.mount("/", StaticFiles(directory=front_dir, html=True), name="front")
