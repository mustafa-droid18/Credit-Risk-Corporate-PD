#!/usr/bin/env python3
import os
import json
import warnings
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.isotonic import IsotonicRegression
import joblib
from preprocessing import Preprocessing_Pipeline
warnings.filterwarnings('ignore', category=RuntimeWarning)

DATA_PATH = "Default_added.csv"
ARTIFACT_DIR = "artifacts"
TARGET_COLUMN = "default_12m"
AVAIL_COLUMN = "avail_date"
AVAIL_CUTOFF = "2012-05-01"
CALIBRATION_MONTHS = 12

'''Feature Engineering'''
FEATURE_COLUMNS: List[str] = [
    "leverage_w",
    "roa_w",                 
    "cfroa_w",
    "tangible_asset_ratio_w",
    "debt_maturity_ratio_w", 
    "roic_operating_pct_w",
    "altman_z",
    "cf_to_debt_std",
    "cash_to_assets_w",
    "fin_debt_ratio_w",
    "interest_coverage_w", 
    "ebitda_margin_w",
]


def bootstrap_auc_ci(y_true, y_pred_proba, n_bootstrap=1000, confidence_level=0.95, random_state=42):
    '''Calculate bootstrap confidence interval for AUC'''
    rng = np.random.RandomState(random_state)
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    n_samples = len(y_true)
    bootstrap_aucs = []
    
    for i in range(n_bootstrap):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_boot = y_true[indices]
        pred_boot = y_pred_proba[indices]
        
        if len(np.unique(y_boot)) < 2:
            continue
        
        try:
            auc_boot = roc_auc_score(y_boot, pred_boot)
            bootstrap_aucs.append(auc_boot)
        except Exception:
            continue
    '''Bootstrap AUC CI'''
    bootstrap_aucs = np.array(bootstrap_aucs)
    if len(bootstrap_aucs) < 100:
        print(f"Warning: Only {len(bootstrap_aucs)} successful bootstrap samples")

    '''Calculate CI'''
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, (alpha / 2) * 100)
    ci_upper = np.percentile(bootstrap_aucs, (1 - alpha / 2) * 100)
    return {
        'mean_auc': float(np.mean(bootstrap_aucs)),
        'std_auc': float(np.std(bootstrap_aucs)),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
        'ci_width': float(ci_upper - ci_lower),
        'confidence_level': confidence_level,
        'n_bootstrap': len(bootstrap_aucs)
    }
'''PD Model Calibrator using Isotonic Regression'''
class PDModelCalibrator:
    def __init__(self):
        self.calibrator = None
        self.is_fitted = False
        self.calibration_metrics = {}

    def fit(self, y_true, y_pred_proba):
        y_true = np.array(y_true)
        y_pred_proba = np.array(y_pred_proba)
        
        auc_before = roc_auc_score(y_true, y_pred_proba)
        brier_before = brier_score_loss(y_true, y_pred_proba)
        
        self.calibrator = IsotonicRegression(out_of_bounds='clip')
        self.calibrator.fit(y_pred_proba, y_true)
        self.is_fitted = True
        
        y_pred_calibrated = self.calibrator.transform(y_pred_proba)
        auc_after = roc_auc_score(y_true, y_pred_calibrated)
        brier_after = brier_score_loss(y_true, y_pred_calibrated)
        
        self.calibration_metrics = {
            'auc_before': float(auc_before),
            'auc_after': float(auc_after),
            'auc_change': float(auc_after - auc_before),
            'brier_before': float(brier_before),
            'brier_after': float(brier_after),
            'brier_improvement': float(brier_before - brier_after)
        }
        
        return self.calibration_metrics
    
    def transform(self, y_pred_proba):
        '''Apply calibration to predicted probabilities'''
        if not self.is_fitted:
            return y_pred_proba
        y_pred_proba = np.array(y_pred_proba)
        return self.calibrator.transform(y_pred_proba)
    
    def should_use_calibration(self, tolerance=0.001):
        if not self.is_fitted:
            return False
        brier_improves = self.calibration_metrics['brier_improvement'] > tolerance
        auc_not_hurt = self.calibration_metrics['auc_change'] >= -tolerance
        return brier_improves and auc_not_hurt


def load_data(path: str) -> pd.DataFrame:
    '''Load and validate data from CSV'''
    print("Loading data...")
    df = pd.read_csv(path, low_memory=False)
    print(f"Total rows: {len(df):,}")
    
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    print(f"Target dtype: {df[TARGET_COLUMN].dtype}")
    print(f"Target unique values: {sorted(df[TARGET_COLUMN].dropna().unique())}")
    print(f"Missing targets: {df[TARGET_COLUMN].isna().sum():,}")
    n_before = len(df)
    df = df.dropna(subset=[TARGET_COLUMN]).copy()
    n_after = len(df)
    print(f"Dropped {n_before - n_after:,} rows with missing target")
    
    unique_vals = df[TARGET_COLUMN].unique()
    if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        raise ValueError(f"Target must be binary (0/1), found: {unique_vals}")
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)
    print(f"Target validated: binary, {n_after:,} rows, default rate = {df[TARGET_COLUMN].mean():.4f}")
    
    df[AVAIL_COLUMN] = pd.to_datetime(df[AVAIL_COLUMN])
    return df


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Split data into train and test based on AVAIL_CUTOFF'''
    cutoff_ts = pd.to_datetime(AVAIL_CUTOFF)
    train_df = df[df[AVAIL_COLUMN] < cutoff_ts].copy()
    test_df = df[df[AVAIL_COLUMN] >= cutoff_ts].copy()
    print(f"\nTrain/Test Split:")
    print(f"Train: avail_date < {AVAIL_CUTOFF} -> {len(train_df):,} rows")
    print(f"Test:  avail_date >= {AVAIL_CUTOFF} -> {len(test_df):,} rows")
    
    return train_df, test_df



def split_train_calibration(df: pd.DataFrame, cal_months: int = CALIBRATION_MONTHS) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''Split training data into model training and calibration holdout sets'''
    df = df.sort_values(by=AVAIL_COLUMN)
    max_date = df[AVAIL_COLUMN].max()
    cal_start = max_date - pd.DateOffset(months=cal_months)
    train_model = df[df[AVAIL_COLUMN] <= cal_start].copy()
    train_cal = df[df[AVAIL_COLUMN] > cal_start].copy()
    print(f"\nCalibration Holdout Split:")
    print(f"Model training: avail_date < {cal_start.date()} -> {len(train_model):,} rows")
    print(f"Calibration:    avail_date >= {cal_start.date()} -> {len(train_cal):,} rows")
    
    return train_model, train_cal


def fit_xgboost(X: pd.DataFrame, y: pd.Series, scale_pos_weight: float, params: Optional[Dict] = None) -> xgb.XGBClassifier:
    '''Fit XGBoost model with specified parameters'''
    if params is None:
        print("Applying STRONGER REGULARIZED parameters...")
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'n_estimators': 500,         # Keep high for early stopping
            'learning_rate': 0.05,        # Keep low
            'max_depth': 4,               # <- STRONGER (Decreased from 5)
            'subsample': 0.8,             # <- Kept
            'colsample_bytree': 0.8,      # <- STRONGER (Increased from 0.7)
            'min_child_weight': 20,       # <- STRONGER (Increased from 10)
            'gamma': 0.2,                 # <- STRONGER (Increased from 0.1)
            'random_state': 42,
            'early_stopping_rounds': 75,   # Keep
        }
    
    # Add the dynamic scale_pos_weight, calculated from the training set
    params['scale_pos_weight'] = scale_pos_weight
    model = xgb.XGBClassifier(**params)
    from sklearn.model_selection import train_test_split
    X_train_fit, X_eval, y_train_fit, y_eval = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    print(f"Training with early stopping (eval_size={len(X_eval)})...")
    model.fit(X_train_fit, y_train_fit, 
              eval_set=[(X_eval, y_eval)],
              verbose=False)
    print(f"Best iteration: {model.best_iteration}")
    return model

def predict_xgboost(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    '''Predict probabilities using the XGBoost model'''
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=FEATURE_COLUMNS)
    return model.predict_proba(X)[:, 1]


def validate_processed_data(df: pd.DataFrame, dataset_name: str = "data"):
    '''Validate processed data for target and features'''
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column missing from {dataset_name}")
    target = df[TARGET_COLUMN]
    
    if target.isna().any():# Check for NaNs in target
        n_missing = target.isna().sum()
        print(f"Warning: {n_missing} NaN values in {dataset_name} target")
        return False
    
    unique_vals = target.unique()# Check target values
    if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
        print(f"Error: {dataset_name} target has invalid values: {unique_vals}")
        return False
    
    missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing_features:
        print(f"Error: Missing features in {dataset_name}: {missing_features}")
        return False
    for feat in FEATURE_COLUMNS:
        if df[feat].isna().any():
            n_missing = df[feat].isna().sum()
            print(f"Warning: {n_missing} NaN values in {dataset_name} feature '{feat}'")
            return False
    
    return True



def walk_forward_evaluation(train_df: pd.DataFrame, bootstrap_ci: bool = True) -> pd.DataFrame:
    '''Perform walk-forward evaluation on training data'''
    df = train_df.copy()
    avail_min = df[AVAIL_COLUMN].min()
    cutoff_ts = pd.to_datetime(AVAIL_CUTOFF)
    min_year = avail_min.year
    cutoff_year = cutoff_ts.year
    results = []
    all_fold_predictions = []
    print("\nWALK-FORWARD EVALUATION")
    print(f"Using {len(FEATURE_COLUMNS)} features")
    print(f"Date range: {avail_min.date()} to {cutoff_ts.date()}")

    # Iterate over each year for walk-forward splits
    for y in range(min_year, cutoff_year):
        train_end = pd.Timestamp(year=y, month=5, day=1)
        val_start = train_end
        next_may = pd.Timestamp(year=y + 1, month=5, day=1)
        val_end = min(next_may, cutoff_ts)
        if train_end <= avail_min:
            continue
        '''Split data into train and validation sets'''
        train_mask = df[AVAIL_COLUMN] < train_end
        val_mask = (df[AVAIL_COLUMN] >= val_start) & (df[AVAIL_COLUMN] < val_end)
        df_train = df[train_mask].copy()
        df_val = df[val_mask].copy()
        if len(df_train) < 100 or len(df_val) < 10:
            print(f"\nSkipping year {y}: insufficient data")
            continue
        print(f"\nFold {y}:")
        print(f"Train: avail < {train_end.date()} (n={len(df_train):,})")
        print(f"Val:   {val_start.date()} <= avail < {val_end.date()} (n={len(df_val):,})")

        try:
            # We still run the *full* preprocessing
            df_train_proc, details = Preprocessing_Pipeline.train_preprocessing(df_train)
            df_val_proc = Preprocessing_Pipeline.transform_preprocessing(df_val, details)
            
            # We validate data, but only on the columns we selected in FEATURE_COLUMNS
            if not validate_processed_data(df_train_proc, "train"):
                print(f"Skipping fold: train data validation failed")
                continue
            
            if not validate_processed_data(df_val_proc, "validation"):
                print(f"Skipping fold: validation data validation failed")
                continue
            # We select *only* our new feature list
            X_train = df_train_proc[FEATURE_COLUMNS]
            y_train = df_train_proc[TARGET_COLUMN].astype(int)
            X_val = df_val_proc[FEATURE_COLUMNS]
            y_val = df_val_proc[TARGET_COLUMN].astype(int)
            print(f"Data validated")
            print(f"Train: {len(X_train):,} samples, default rate = {y_train.mean():.4f}")
            print(f"Val:   {len(X_val):,} samples, default rate = {y_val.mean():.4f}")
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
            print(f"Applying scale_pos_weight: {scale_pos_weight:.2f}")

            model = fit_xgboost(X_train, y_train, scale_pos_weight=scale_pos_weight)
            pd_hat = predict_xgboost(model, X_val)
            
            try:
                auc = roc_auc_score(y_val, pd_hat)
            except ValueError as e:
                print(f"AUC calculation failed: {e}")
                auc = np.nan
            
            brier = brier_score_loss(y_val, pd_hat)
            
            print(f"AUC:   {auc:.4f}")
            print(f"Brier: {brier:.4f}")
            
            fold_ci = None
            if bootstrap_ci and not np.isnan(auc):
                '''Calculate bootstrap CI for this fold'''
                try:
                    fold_ci = bootstrap_auc_ci(y_val.values, pd_hat, n_bootstrap=1000, confidence_level=0.95, random_state=42 + y)
                    print(f"95% CI: [{fold_ci['ci_lower']:.4f}, {fold_ci['ci_upper']:.4f}]")
                except Exception as e:
                    print(f"Bootstrap CI failed: {e}")
                    fold_ci = None
            
            results.append({
                "year": y,
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "n_train": int(len(df_train_proc)),
                "n_val": int(len(df_val_proc)),
                "default_rate_train": float(y_train.mean()),
                "default_rate_val": float(y_val.mean()),
                "auc": float(auc),
                "brier": float(brier),
                "auc_ci_lower": float(fold_ci['ci_lower']) if fold_ci else None,
                "auc_ci_upper": float(fold_ci['ci_upper']) if fold_ci else None,
                "auc_ci_width": float(fold_ci['ci_width']) if fold_ci else None
            })
            
            if bootstrap_ci and not np.isnan(auc):
                '''Store predictions for overall CI calculation'''
                all_fold_predictions.append({'y_true': y_val.values, 'y_pred': pd_hat})
            
        except Exception as e:
            print(f"Error processing fold: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    results_df = pd.DataFrame(results)
    
    if len(results_df) > 0:
        '''Display walk-forward evaluation results'''
        print("\nWALK-FORWARD SUMMARY")
        print(results_df[['year', 'n_train', 'n_val', 'auc', 'brier']])
        print(f"\nAggregate Statistics:")
        print(f"Mean AUC:   {results_df['auc'].mean():.4f} +/- {results_df['auc'].std():.4f}")
        print(f"Median AUC: {results_df['auc'].median():.4f}")
        print(f"Mean Brier: {results_df['brier'].mean():.4f} +/- {results_df['brier'].std():.4f}")
        
        if bootstrap_ci and 'auc_ci_width' in results_df.columns:
            '''Display bootstrap CI statistics'''
            print(f"\nBootstrap CI Statistics:")
            print(f"Mean CI Width: {results_df['auc_ci_width'].mean():.4f}")
            print(f"Median CI Width: {results_df['auc_ci_width'].median():.4f}")
        
        if bootstrap_ci and all_fold_predictions:
            print("\nOVERALL BOOTSTRAP CI (Pooling All Validation Folds)")
            
            y_true_all = np.concatenate([pred['y_true'] for pred in all_fold_predictions])
            y_pred_all = np.concatenate([pred['y_pred'] for pred in all_fold_predictions])
            
            overall_auc = roc_auc_score(y_true_all, y_pred_all)
            print(f"Total validation samples: {len(y_true_all):,}")
            print(f"Overall AUC: {overall_auc:.4f}")
            
            try:
                overall_ci = bootstrap_auc_ci(y_true_all, y_pred_all, n_bootstrap=2000, confidence_level=0.95, random_state=42)
                print(f"Bootstrap Mean AUC: {overall_ci['mean_auc']:.4f} +/- {overall_ci['std_auc']:.4f}")
                print(f"95% CI: [{overall_ci['ci_lower']:.4f}, {overall_ci['ci_upper']:.4f}]")
                print(f"CI Width: {overall_ci['ci_width']:.4f}")
                
                results_df.attrs['overall_bootstrap_ci'] = overall_ci
                results_df.attrs['overall_auc'] = float(overall_auc)
            except Exception as e:
                print(f"Overall bootstrap CI failed: {e}")
    else:
        print("\nWarning: No folds were successfully processed")
    
    return results_df



def train_final_model(train_df: pd.DataFrame, bootstrap_ci: bool = True) -> Tuple[object, Dict, List[str], Dict]:
    '''Train final model on all training data'''

    print("\nTRAINING FINAL MODEL")
    print(f"Training on all data with avail_date < {AVAIL_CUTOFF}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    
    train_proc, details = Preprocessing_Pipeline.train_preprocessing(train_df)
    '''Validate processed training data'''
    
    if not validate_processed_data(train_proc, "training"):
        raise ValueError("Training data validation failed after preprocessing")
    
    X = train_proc[FEATURE_COLUMNS]
    y = train_proc[TARGET_COLUMN].astype(int)
    
    print(f"\nData prepared:")
    print(f"Samples: {len(X):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print(f"Default rate: {y.mean():.4f}")
    
    # Calculate scale_pos_weight
    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Applying scale_pos_weight: {scale_pos_weight:.2f}")

    model = fit_xgboost(X, y, scale_pos_weight=scale_pos_weight)

    # Evaluate on training data
    pd_hat_train = predict_xgboost(model, X)
    auc_train = roc_auc_score(y, pd_hat_train)
    brier_train = brier_score_loss(y, pd_hat_train)
    
    print(f"\nModel trained successfully")
    print(f"Training AUC:   {auc_train:.4f}")
    print(f"Training Brier: {brier_train:.4f}")
    
    train_ci = None
    if bootstrap_ci and not np.isnan(auc_train):
        '''Calculate bootstrap CI for training data'''
        try:
            train_ci = bootstrap_auc_ci(y_true=y.values, y_pred_proba=pd_hat_train, n_bootstrap=2000, confidence_level=0.95, random_state=42)
            print(f"95% CI: [{train_ci['ci_lower']:.4f}, {train_ci['ci_upper']:.4f}]")
        except Exception as e:
            print(f"Training bootstrap CI failed: {e}")
            train_ci = None
    
    train_metrics: Dict = {
        "n_samples": int(len(y)),
        "default_rate": float(y.mean()),
        "auc": float(auc_train),
        "brier": float(brier_train),
    }
    if train_ci is not None:
        train_metrics["bootstrap_ci"] = train_ci
    
    print("\nFEATURE IMPORTANCE")
    feature_importances = model.feature_importances_
    for feat, importance in zip(FEATURE_COLUMNS, feature_importances):
        print(f"{feat:30s}: {importance:8.4f}")
    
    return model, details, FEATURE_COLUMNS, train_metrics



def train_and_evaluate_calibration(cal_df: pd.DataFrame, model: object, preproc_details: Dict, 
                                   bootstrap_ci: bool = True, tolerance: float = 0.001) -> Tuple[Optional[PDModelCalibrator], Dict]:
    '''Train and evaluate calibration on holdout set'''

    print("\nCALIBRATION ON HOLDOUT SET")
    cal_proc = Preprocessing_Pipeline.transform_preprocessing(cal_df, preproc_details)    
    if not validate_processed_data(cal_proc, "calibration"):
        '''Validate processed calibration data'''
        
        raise ValueError("Calibration data validation failed after preprocessing")
    
    X_cal = cal_proc[FEATURE_COLUMNS]
    y_cal = cal_proc[TARGET_COLUMN].astype(int)
    n_samples = len(X_cal)
    default_rate = float(y_cal.mean())
    
    print(f"\nCalibration data prepared:")
    print(f"Samples: {n_samples:,}")
    print(f"Default rate: {default_rate:.4f}")
    
    pd_hat_uncal = predict_xgboost(model, X_cal)
    try:
        auc_uncal = roc_auc_score(y_cal, pd_hat_uncal)
    except ValueError as e:
        print(f"AUC calculation failed on calibration set: {e}")
        auc_uncal = np.nan
    brier_uncal = brier_score_loss(y_cal, pd_hat_uncal)
    
    print(f"\nUNCALIBRATED METRICS (calibration holdout):")
    print(f"AUC:   {auc_uncal:.4f}")
    print(f"Brier: {brier_uncal:.4f}")
    
    uncal_ci = None # Calculate bootstrap CI for uncalibrated probs
    if bootstrap_ci and not np.isnan(auc_uncal):
        try:
            uncal_ci = bootstrap_auc_ci(y_true=y_cal.values, y_pred_proba=pd_hat_uncal, n_bootstrap=1000, 
                                       confidence_level=0.95, random_state=202)
            print(f"95% CI: [{uncal_ci['ci_lower']:.4f}, {uncal_ci['ci_upper']:.4f}]")
        except Exception as e:
            print(f"Uncalibrated bootstrap CI failed: {e}")
            uncal_ci = None
    
    uncalibrated_metrics: Dict = {
        "auc": float(auc_uncal),
        "brier": float(brier_uncal),
        "default_rate": default_rate,
    }
    if uncal_ci is not None:
        uncalibrated_metrics["bootstrap_ci"] = uncal_ci

    # Train calibrator
    calibrator = PDModelCalibrator()
    calib_summary = calibrator.fit(y_cal, pd_hat_uncal)
    
    print("\nCALIBRATION METRIC CHANGES:")
    print(f"AUC before:  {calib_summary['auc_before']:.6f}")
    print(f"AUC after:   {calib_summary['auc_after']:.6f}")
    print(f"AUC change:  {calib_summary['auc_change']:+.6f}")
    print(f"Brier before:{calib_summary['brier_before']:.6f}")
    print(f"Brier after: {calib_summary['brier_after']:.6f}")
    print(f"Brier improv:{calib_summary['brier_improvement']:+.6f}")
    
    pd_hat_cal = calibrator.transform(pd_hat_uncal)
    # Evaluate calibrated probabilities
    try:
        auc_cal = roc_auc_score(y_cal, pd_hat_cal)
    except ValueError as e:
        print(f"AUC calculation failed for calibrated probs: {e}")
        auc_cal = np.nan
    brier_cal = brier_score_loss(y_cal, pd_hat_cal)
    
    print(f"\nCALIBRATED METRICS (calibration holdout):")
    print(f"AUC:   {auc_cal:.4f}")
    print(f"Brier: {brier_cal:.4f}")
    
    cal_ci = None
    if bootstrap_ci and not np.isnan(auc_cal):
        try:
            cal_ci = bootstrap_auc_ci(y_true=y_cal.values, y_pred_proba=pd_hat_cal, n_bootstrap=1000, 
                                     confidence_level=0.95, random_state=303)
            print(f"95% CI: [{cal_ci['ci_lower']:.4f}, {cal_ci['ci_upper']:.4f}]")
        except Exception as e:
            print(f"Calibrated bootstrap CI failed: {e}")
            cal_ci = None
    # Prepare calibrated metrics
    calibrated_metrics: Dict = {
        "auc": float(auc_cal),
        "brier": float(brier_cal),
        "default_rate": default_rate,
    }
    if cal_ci is not None:
        calibrated_metrics["bootstrap_ci"] = cal_ci
    
    use_calibration = calibrator.should_use_calibration(tolerance=tolerance)
    print("\nCALIBRATION DECISION:")
    print(f"Brier improvement > {tolerance}? {calib_summary['brier_improvement'] > tolerance}")
    print(f"AUC degradation <= {tolerance}? {calib_summary['auc_change'] >= -tolerance}")
    print(f"Use calibration: {'YES' if use_calibration else 'NO'}")
    #  metrics summary
    cal_metrics: Dict = {
        "n_samples": int(n_samples),
        "default_rate": default_rate,
        "use_calibration": bool(use_calibration),
        "uncalibrated": uncalibrated_metrics,
        "calibration_summary": calib_summary
    }
    
    if use_calibration:
        cal_metrics["calibrated"] = calibrated_metrics
        cal_metrics["improvements"] = {
            "auc_change": float(calib_summary["auc_change"]),
            "brier_improvement": float(calib_summary["brier_improvement"])
        }
        calibrator_to_return = calibrator
    else:
        cal_metrics["calibrated"] = calibrated_metrics
        cal_metrics["improvements"] = {
            "auc_change": float(calib_summary["auc_change"]),
            "brier_improvement": float(calib_summary["brier_improvement"])
        }
        calibrator_to_return = None
    
    return calibrator_to_return, cal_metrics



def save_artifacts(model: object, preproc_details: Dict, feature_columns: List[str],
                  train_model_df: pd.DataFrame, train_cal_df: pd.DataFrame,
                  calibrator: Optional[PDModelCalibrator], train_metrics: Dict,
                  cal_metrics: Dict, wf_results: pd.DataFrame):
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    '''Save model, calibrator, and metadata'''
    
    model_path = os.path.join(ARTIFACT_DIR, "model.joblib")
    meta_path = os.path.join(ARTIFACT_DIR, "meta.json")
    
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")
    
    if calibrator is not None:
        cal_path = os.path.join(ARTIFACT_DIR, "calibrator.joblib")
        joblib.dump(calibrator, cal_path)
        print(f"Saved calibrator to: {cal_path}")
    else:
        print(f"No calibrator saved (not beneficial)")
    
    avail_years_model = sorted(int(y) for y in train_model_df[AVAIL_COLUMN].dt.year.unique())
    avail_years_cal = sorted(int(y) for y in train_cal_df[AVAIL_COLUMN].dt.year.unique())
    
    meta = {
        "model_version": "3.3_financials_strong_regex", # --- MODIFIED ---
        "feature_columns": feature_columns,
        "target_column": TARGET_COLUMN,
        "avail_column": AVAIL_COLUMN,
        "avail_cutoff": AVAIL_CUTOFF,
        "calibration_months": CALIBRATION_MONTHS,

        "training": {
            "n_obs": int(len(train_model_df)),
            "avail_years": avail_years_model,
            "metrics": train_metrics
        },
        
        "calibration": {
            "n_obs": int(len(train_cal_df)),
            "avail_years": avail_years_cal,
            "metrics": cal_metrics
        },
        
        "walk_forward": {
            "mean_auc": float(wf_results['auc'].mean()) if len(wf_results) > 0 else None,
            "std_auc": float(wf_results['auc'].std()) if len(wf_results) > 0 else None,
            "n_folds": int(len(wf_results)),
            "overall_ci": wf_results.attrs.get('overall_bootstrap_ci') if hasattr(wf_results, 'attrs') else None
        },
        
        "preprocessing_details": preproc_details,
        "model_type": "xgboost.XGBClassifier",
        "calibration_used": calibrator is not None,
        "data_leakage_prevented": True,
        "calibration_method": "temporal_holdout"
    }
    
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved metadata to: {meta_path}")
    
    bootstrap_path = os.path.join(ARTIFACT_DIR, "bootstrap_results.json")
    bootstrap_data = {
        "training_set": train_metrics.get('bootstrap_ci'),
        "calibration_set": {
            "uncalibrated": cal_metrics.get('uncalibrated', {}).get('bootstrap_ci'),
            "calibrated": cal_metrics.get('calibrated', {}).get('bootstrap_ci')
        },
        "walk_forward": wf_results.attrs.get('overall_bootstrap_ci') if hasattr(wf_results, 'attrs') else None
    }
    
    with open(bootstrap_path, "w") as f:
        json.dump(bootstrap_data, f, indent=2)
    print(f"Saved bootstrap results to: {bootstrap_path}")


# Main execution
def main():
    print("BANCA MASSICCIA PD MODEL - TRAINING PIPELINE (XGBoost)")
    print("Walk-forward validation with bootstrap CI")
    print("Temporal calibration holdout")
    print("Calibration with validation")
    
    df = load_data(DATA_PATH)
    '''Initial data loading and preprocessing'''
    train_df, test_df = split_train_test(df)
    
    print("\nSTEP 1: WALK-FORWARD VALIDATION")
    wf_results = walk_forward_evaluation(train_df, bootstrap_ci=True)
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    wf_path = os.path.join(ARTIFACT_DIR, "walk_forward_results.csv")
    wf_results.to_csv(wf_path, index=False)
    print(f"\nWalk-forward results saved to: {wf_path}")
    
    print("\nSTEP 2: SPLIT FOR CALIBRATION")
    train_model_df, train_cal_df = split_train_calibration(train_df, CALIBRATION_MONTHS)
    
    print("\nSTEP 3: TRAIN FINAL MODEL")
    model, preproc_details, feature_columns, train_metrics = train_final_model(train_model_df, bootstrap_ci=True)
    
    print("\nSTEP 4: TRAIN AND VALIDATE CALIBRATION")
    calibrator, cal_metrics = train_and_evaluate_calibration(train_cal_df, model, preproc_details, bootstrap_ci=True)
    
    print("\nSTEP 5: SAVE ARTIFACTS")
    save_artifacts(model, preproc_details, feature_columns, train_model_df, train_cal_df, 
                  calibrator, train_metrics, cal_metrics, wf_results)
    
    print("\nTRAINING COMPLETE - SUMMARY")
    
    print(f"\nWALK-FORWARD VALIDATION:")
    if len(wf_results) > 0:
        print(f"Folds processed: {len(wf_results)}")
        print(f"Mean AUC: {wf_results['auc'].mean():.4f} +/- {wf_results['auc'].std():.4f}")
        print(f"Median AUC: {wf_results['auc'].median():.4f}")
        if 'overall_auc' in wf_results.attrs:
            print(f"Overall AUC: {wf_results.attrs['overall_auc']:.4f}")
            if 'overall_bootstrap_ci' in wf_results.attrs:
                ci = wf_results.attrs['overall_bootstrap_ci']
                print(f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    print(f"\nFINAL MODEL (Training Set):")
    print(f"Samples: {train_metrics['n_samples']:,}")
    print(f"AUC: {train_metrics['auc']:.4f}")
    print(f"Brier: {train_metrics['brier']:.4f}")
    if train_metrics.get('bootstrap_ci'):
        ci = train_metrics['bootstrap_ci']
        print(f"95% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
    
    print(f"\nCALIBRATION:")
    print(f"Using calibration: {'YES' if cal_metrics['use_calibration'] else 'NO'}")
    print(f"Calibration samples: {cal_metrics['n_samples']:,}")
    if cal_metrics['use_calibration']:
        print(f"Brier improvement: {cal_metrics['improvements']['brier_improvement']:.6f}")
        print(f"AUC change: {cal_metrics['improvements']['auc_change']:+.6f}")
        print(f"Final calibrated AUC: {cal_metrics['calibrated']['auc']:.4f}")
    else:
        print(f"Uncalibrated AUC on holdout: {cal_metrics['uncalibrated']['auc']:.4f}")
    
    print(f"\nARTIFACTS SAVED TO: {ARTIFACT_DIR}/")
    print(f"model.joblib")
    if cal_metrics['use_calibration']:
        print(f"calibrator.joblib")
    print(f"meta.json")
    print(f"bootstrap_results.json")
    print(f"walk_forward_results.csv")


if __name__ == "__main__":
    main()