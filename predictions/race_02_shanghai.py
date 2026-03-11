"""
2026 Chinese Grand Prix - Podium Prediction for 15/03/2026
Run this script after Saturday qualifying to predict Sunday's podium
"""

import fastf1
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import xgboost as xgb 

# ============================================================
# RACE CONFIGURATION
# ============================================================
YEAR = 2026
RACE_NUMBER = 2
RACE_NAME = "Chinese Grand Prix"
CIRCUIT = "Shanghai"

print(f"\n{'='*60}")
print(f"🏎️  {RACE_NAME} - Podium Prediction")
print(f"{'='*60}\n")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading trained model...")
model = joblib.load('models/xgb_f1_2026_final.pkl')

with open('models/feature_list.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]

print(f"✅ Model loaded with {len(feature_cols)} features\n")

# ============================================================
# GET QUALIFYING RESULTS
# ============================================================
print(f"Fetching {YEAR} qualifying results...")
fastf1.Cache.enable_cache('data/fastf1_cache')

