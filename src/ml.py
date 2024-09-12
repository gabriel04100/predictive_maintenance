import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from typing import Tuple, Dict, Any
from src.transform import transform_maintenance_data
import os

def prepare_data(data_path: str, target: str, test_size: float = 0.2, random_state: int = 42,
                 split_filepath: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Transforms the data, then either splits it or loads a pre-saved train-test split.
    If split_filepath is provided, it will attempt to load the split from there.
    """
    df = pd.read_csv(data_path, index_col="UDI")
    df = df.drop(columns=["Product ID", "HDF","TWF","PWF","OSF","RNF"])
    # Apply transformations to the dataset
    df = transform_maintenance_data(df=df, speed_column="Rotational speed [rpm]",
                                    torque_column="Torque [Nm]", col1="Process temperature [K]",
                                    col2="Air temperature [K]", result_col="temp_diff [K]")
    df=pd.get_dummies(df)
    df = df.rename(columns={
    'Air temperature [K]': 'Air temperature',
    'Process temperature [K]': 'Process temperature',
    'Rotational speed [rpm]': 'Rotational speed',
    'Torque [Nm]': 'Torque',
    'Tool wear [min]': 'Tool wear',
    'mechanical_power [W]': 'Mechanical power',
    'temp_diff [K]': 'Temp diff'
    })
    print(df)
    if split_filepath and os.path.exists(split_filepath):
        # Load pre-saved train-test split
        X_train, X_test, y_train, y_test = joblib.load(split_filepath)
    else:
        # Perform train-test split
        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
        
        # Optionally save the split
        if split_filepath:
            joblib.dump((X_train, X_test, y_train, y_test), split_filepath)
    
    return X_train, X_test, y_train, y_test


# Function to fit the model and save train-test split
def fit_model(X_train: pd.DataFrame, y_train: pd.Series, split_filepath: str = None) -> Any:
    """
    Trains the model using HalvingGridSearchCV, and saves the train-test split if split_filepath is provided.
    """
    lgbm = LGBMClassifier(force_col_wise=True, verbose=-1, class_weight='balanced')

    param_grid = {
        'num_leaves': [31, 50, 70, 100],
        'max_depth': [10, 20, 30],
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'n_estimators': [100, 200],
        'min_child_samples': [10, 20],
        'subsample': [0.8, 0.9, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5, 1.0],
        'reg_lambda': [0.0, 0.1, 0.5, 1.0]
    }

    halving_cv = HalvingGridSearchCV(
        estimator=lgbm,
        param_grid=param_grid,
        factor=3,
        random_state=42,
        scoring='recall',  
        cv=StratifiedKFold(n_splits=5),
        verbose=0
    )

    halving_cv.fit(X_train, y_train)
    
    return halving_cv.best_estimator_  # Return the best model


# Function to evaluate the model
def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Evaluates the model and returns a dictionary of metrics.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics


# Function to save the model and train-test split
def save_model_and_split(model: Any, model_filepath: str, split_filepath: str = None) -> None:
    """
    Saves the model to the model_filepath and optionally saves the train-test split to split_filepath.
    """
    joblib.dump(model, model_filepath)
    if split_filepath:
        joblib.dump(split_filepath)


# Function to load the model and train-test split
def load_model_and_split(model_filepath: str, split_filepath: str = None) -> Tuple[Any, Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]]:
    """
    Loads the model and optionally loads the train-test split.
    """
    model = joblib.load(model_filepath)
    
    if split_filepath and os.path.exists(split_filepath):
        X_train, X_test, y_train, y_test = joblib.load(split_filepath)
        return model, (X_train, X_test, y_train, y_test)
    else:
        return model, None

