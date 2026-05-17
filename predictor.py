#!/usr/bin/env python3
"""
Helper module for scoring PDs from the trained XGBoost model.
Handles preprocessing, prediction, and calibration.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import warnings
import sys  # <-- FIX 1: Added missing import
from typing import Dict, List, Tuple, Optional

# Import the preprocessing pipeline used during training
try:
    # --- FIX 2: Corrected file name ---
    from preprocessing import Preprocessing_Pipeline
except ImportError:
    # --- FIX 2: Corrected file name in error message ---
    print("Error: Could not find 'preprocessing.py'.")
    print("Make sure that file is in the same directory as this predictor.")
    sys.exit(1)

# Imports required to load the PDModelCalibrator class
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

ARTIFACT_DIR = "artifacts"
MODEL_FILE = "model.joblib"
META_FILE = "meta.json"
CALIBRATOR_FILE = "calibrator.joblib"


# ===================================================================
# --- CLASS DEFINITION ---
# This class definition MUST be present for joblib to load
# the calibrator.joblib object successfully.
# ===================================================================
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
# ===================================================================
# --- END OF CLASS ---
# ===================================================================


def load_artifacts():
    """
    Loads the model, metadata, and calibrator from the artifacts directory.
    """
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILE)
    meta_path = os.path.join(ARTIFACT_DIR, META_FILE)
    cal_path = os.path.join(ARTIFACT_DIR, CALIBRATOR_FILE)

    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"Missing required artifacts in '{ARTIFACT_DIR}/'. "
            "Please run the estimator script first."
        )

    model = joblib.load(model_path)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    calibrator = None
    if os.path.exists(cal_path):
        print("Predictor: Loading calibration file...")
        calibrator = joblib.load(cal_path)
    else:
        print("Predictor: No calibration file found. Returning raw probabilities.")

    return model, meta, calibrator


def predict_pd(df_raw: pd.DataFrame) -> np.ndarray:
    """
    Generates Probability of Default (PD) predictions for new data.
    
    Args:
        df_raw: A raw DataFrame containing the input data.
        
    Returns:
        A numpy array (n x 1 vector) of PD probabilities.
    """
    model, meta, calibrator = load_artifacts()
    preproc_details = meta["preprocessing_details"]
    feature_columns = meta["feature_columns"]

    # Transform only (no fitting)
    df_proc = Preprocessing_Pipeline.transform_preprocessing(df_raw, preproc_details)

    # Build design matrix (XGBoost does not need sm.add_constant)
    X = df_proc[feature_columns]

    # Predict PDs using predict_proba (provides probabilities)
    pd_hat = model.predict_proba(X)[:, 1]

    # Apply calibration if the calibrator was loaded
    if calibrator is not None:
        pd_hat = calibrator.transform(pd_hat)

    return np.asarray(pd_hat)

# No __main__ block, as this script is intended to be imported by harness.py