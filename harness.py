#!/usr/bin/env python3
"""
XGBoost PD Model Harness
Command-line interface for generating PD predictions
Conforms to project output specifications (nx1 vector).
"""

import argparse
import pandas as pd
import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import the predictor functions from our new predictor.py
try:
    from predictor import predict_pd, PDModelCalibrator
    
except ImportError:
    print("Error: Could not find 'predictor.py'.")
    print("Make sure that file is in the same directory as this harness.")
    sys.exit(1)

# --- IMPORTS Required for the calibrator class ---
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from typing import Dict, List, Tuple, Optional
# --- END IMPORTS ---


def main():
    """Main function for generating PD predictions"""
    parser = argparse.ArgumentParser(
        description='XGBoost PD Model Harness'
    )
    parser.add_argument("--input_csv", required=True,
                        help='Path to input CSV file')
    parser.add_argument("--output_csv", required=True,
                        help='Path to output CSV file for predictions')
    args = parser.parse_args()

    print("XGBOOST PD MODEL HARNESS")
    print(f"Input:  {args.input_csv}")
    print(f"Output: {args.output_csv}")
    
    try:
        # Validate input
        if not os.path.exists(args.input_csv):
            raise FileNotFoundError(f"Input file not found: {args.input_csv}")
        
        # Load data
        print(f"\nLoading input data...")
        df = pd.read_csv(args.input_csv, low_memory=False)
        print(f"Records loaded: {len(df):,}")
        
        # Check artifacts
        script_dir = os.path.dirname(os.path.abspath(__file__))
        artifacts_dir = os.path.join(script_dir, 'artifacts')
        model_path = os.path.join(artifacts_dir, 'model.joblib')
        meta_path = os.path.join(artifacts_dir, 'meta.json')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")
        
        print(f"Model artifacts found in: {artifacts_dir}")
        
        # Generate predictions
        print("\nGenerating predictions...")
        # --- MODIFICATION ---
        # Call predict_pd directly to get the nx1 vector
        pd_hat = predict_pd(df)
        
        # Validate predictions
        if pd_hat is None or len(pd_hat) == 0:
            raise ValueError("predict_pd returned empty predictions")
        
        if len(pd_hat) != len(df):
            raise ValueError(f"Length mismatch: {len(pd_hat)} predictions for {len(df)} records")
        
        # Check for invalid values
        if np.any(np.isnan(pd_hat)):
            n_nan = np.sum(np.isnan(pd_hat))
            raise ValueError(f"Found {n_nan} NaN values in predictions")
        
        if np.any((pd_hat < 0) | (pd_hat > 1)):
            n_invalid = np.sum((pd_hat < 0) | (pd_hat > 1))
            raise ValueError(f"Found {n_invalid} predictions outside [0, 1] range")
        
        # --- MODIFICATION ---
        # Save predictions as a single n x 1 vector (text file)
        # This matches the project requirements 
        print(f"\nSaving predictions to: {args.output_csv}")
        np.savetxt(args.output_csv, pd_hat, fmt='%.10f', delimiter=',')
        
        # Summary statistics
        print("\nPREDICTION SUMMARY")
        print(f"Total predictions: {len(pd_hat):,}")
        print(f"PD Range:  [{pd_hat.min():.6f}, {pd_hat.max():.6f}]")
        print(f"Mean PD:   {pd_hat.mean():.6f} ({pd_hat.mean()*100:.3f}%)")
        print(f"Median PD: {np.median(pd_hat):.6f}")
        print(f"Std PD:    {pd_hat.std():.6f}")
        
        print(f"\nSuccess: Predictions saved to {args.output_csv}")
        
        return 0
        
    except FileNotFoundError as e:
        print(f"\nERROR: {str(e)}")
        print("Please ensure you have run the estimator script to train the model first.")
        sys.exit(1)
        
    except ValueError as e:
        print(f"\nERROR: {str(e)}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    sys.exit(main())