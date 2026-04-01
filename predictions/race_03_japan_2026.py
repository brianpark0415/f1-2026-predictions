"""
2026 Japanese Grand Prix - Podium Prediction
Run this to see what the model would have predicted for Japan
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import xgboost as xgb 

# ============================================================
# RACE CONFIGURATION
# ============================================================
YEAR = 2026
RACE_NUMBER = 3
RACE_NAME = "Japanese Grand Prix"
CIRCUIT = "Suzuka"

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
# REAL QUALIFYING RESULTS - Suzuka 2026
# ============================================================
print("Using REAL Suzuka qualifying results...\n")

quali_data = {
    'Driver': ['ANT', 'RUS', 'PIA', 'LEC', 'NOR', 'HAM', 'GAS', 'HAD', 'BOR',
               'LIN', 'VER', 'OCO', 'HUL', 'LAW', 'COL', 'SAI', 'ALB', 'BEA',
               'PER', 'BOT', 'ALO', 'STR'],
    'Team': ['Mercedes', 'Mercedes', 'McLaren', 'Ferrari', 'McLaren', 'Ferrari',
             'Alpine', 'Red Bull Racing', 'Audi', 'Racing Bulls', 'Red Bull Racing',
             'Haas', 'Audi', 'Racing Bulls', 'Alpine', 'Williams', 'Williams',
             'Haas', 'Cadillac', 'Cadillac', 'Aston Martin', 'Aston Martin'],
    'QualiPosition': list(range(1, 23)),
    'GridPosition': list(range(1, 23)),
    'QualiGapToPole': [0.000, 0.298, 0.354, 0.627, 0.631, 0.789, 0.913, 1.200,
                       1.496, 1.541, 1.741, 1.897, 2.080, 2.103, 2.153, 2.149,
                       2.296, 2.312, 3.428, 3.552, 3.868, 4.142]
}

quali = pd.DataFrame(quali_data)
print("✅ Real qualifying data loaded\n")
print(quali[['Driver', 'Team', 'QualiPosition']].head(10))

# ============================================================
# LOAD 2026 SEASON STATS (Including Races 1-2)
# ============================================================
print("\nLoading 2026 season statistics (including Races 1-2)...")

# Load updated features with 2026 data
historical = pd.read_csv('data/features_2022_2025.csv')

# Get 2026 stats
races_2026 = historical[historical['Year'] == 2026]

if len(races_2026) > 0:
    print(f"✅ Found {len(races_2026)} driver-race results from 2026")
    
    # For each driver, get their most recent stats from 2026
    driver_stats_2026 = races_2026.groupby('Abbreviation').agg({
        'AvgFinishLast5': 'last',
        'AvgQualiLast5': 'last',
        'PointsLast3': 'last',
        'DNFRateSeason': 'last',
        'PodiumRateSeason': 'last',
        'DriverCircuitAvgFinish': 'last'
    }).reset_index()
    
    driver_stats_2026.columns = ['Driver', 'AvgFinishLast5', 'AvgQualiLast5', 
                                  'PointsLast3', 'DNFRateSeason', 'PodiumRateSeason',
                                  'DriverCircuitAvgFinish']
    
    driver_stats = driver_stats_2026
    
else:
    # Fallback to 2025 if no 2026 data
    print("⚠️  No 2026 data found, using 2025 baseline")
    end_of_2025 = historical[historical['Year'] == 2025].copy()
    driver_stats = end_of_2025.groupby('Abbreviation').agg({
        'AvgFinishLast5': 'last',
        'AvgQualiLast5': 'last',
        'PointsLast3': 'last',
        'DNFRateSeason': 'last',
        'PodiumRateSeason': 'last',
    }).reset_index()
    driver_stats.columns = ['Driver', 'AvgFinishLast5', 'AvgQualiLast5', 
                            'PointsLast3', 'DNFRateSeason', 'PodiumRateSeason']
    driver_stats['DriverCircuitAvgFinish'] = driver_stats['AvgFinishLast5']

# Constructor stats (use latest 2026 data if available, else 2025)
if len(races_2026) > 0:
    constructor_stats = races_2026.groupby('TeamName').agg({
        'ConstructorAvgFinishLast3': 'last',
        'ConstructorPodiumRateSeason': 'last',
        'ConstructorChampPos': 'last',
    }).reset_index()
    constructor_stats.columns = ['Team', 'ConstructorAvgFinishLast3', 
                                  'ConstructorPodiumRateSeason', 'ConstructorChampPos']
else:
    end_of_2025 = historical[historical['Year'] == 2025].copy()
    constructor_stats = end_of_2025.groupby('TeamName').agg({
        'ConstructorAvgFinishLast3': 'last',
        'ConstructorPodiumRateSeason': 'last',
        'ConstructorChampPos': 'last',
    }).reset_index()
    constructor_stats.columns = ['Team', 'ConstructorAvgFinishLast3', 
                                  'ConstructorPodiumRateSeason', 'ConstructorChampPos']

print(f"✅ Loaded stats from {'2026' if len(races_2026) > 0 else '2025'}\n")

# ============================================================
# MERGE FEATURES
# ============================================================
print("DEBUG: Checking data before merge...")
print(f"Quali drivers: {quali['Driver'].tolist()}")
print(f"Historical drivers: {driver_stats['Driver'].tolist()[:10]}")

features = quali.merge(driver_stats, on='Driver', how='left')
print(f"After driver merge: {len(features)} rows")

features = features.merge(constructor_stats, on='Team', how='left')
print(f"After constructor merge: {len(features)} rows")
print(f"\nFeatures shape: {features.shape}")
print(features[['Driver', 'Team']].head())

# Handle new drivers/teams - fill with median values
for col in feature_cols:
    if col not in features.columns:
        print(f"⚠️  Missing feature: {col}, filling with median")
        features[col] = 11  # Midfield default
    else:
        features[col] = features[col].fillna(features[col].median())

# ============================================================
# 2026 REGULATION UNCERTAINTY ADJUSTMENT
# ============================================================
# For early races, constructor hierarchy is uncertain due to reg changes
# Dampen constructor advantage toward midfield (5.5 = neutral)

REGULATION_UNCERTAINTY = 0.5  # 50% trust by Race 3 (less dampening)

features['ConstructorChampPos'] = (
    features['ConstructorChampPos'] * REGULATION_UNCERTAINTY + 
    5.5 * (1 - REGULATION_UNCERTAINTY)
)

print("\nApplied 2026 regulation uncertainty adjustment")
print(f"(Constructor advantages dampened {int((1-REGULATION_UNCERTAINTY)*100)}% for early season)\n")

# ============================================================
# PREDICT
# ============================================================
X = features[feature_cols]

# Convert to DMatrix with explicit feature names
dmatrix = xgb.DMatrix(X.values, feature_names=feature_cols)
features['PredictedPosition'] = model.get_booster().predict(dmatrix)

# Sort by predicted position
predictions = features[['Driver', 'Team', 'GridPosition', 'PredictedPosition']].copy()
predictions = predictions.sort_values('PredictedPosition').reset_index(drop=True)
predictions['PredictedFinish'] = predictions.index + 1

# ============================================================
# OUTPUT PREDICTIONS
# ============================================================
print(f"{'='*60}")
print(f"🏁  PREDICTED PODIUM - {RACE_NAME}")
print(f"{'='*60}\n")

podium = predictions.head(3)
for i, row in podium.iterrows():
    medals = ["🥇", "🥈", "🥉"]
    print(f"{medals[i]}  {row['Driver']:3s} - {row['Team']:20s} (Grid: P{int(row['GridPosition'])})")

print(f"\n{'='*60}")
print("TOP 10 PREDICTIONS")
print(f"{'='*60}\n")

print(predictions[['PredictedFinish', 'Driver', 'Team', 'GridPosition']].head(10).to_string(index=False))

print(f"\n{'='*60}")
print("FULL PREDICTED ORDER")
print(f"{'='*60}\n")

print(predictions[['PredictedFinish', 'Driver', 'Team', 'GridPosition']].to_string(index=False))

# ============================================================
# SAVE PREDICTION
# ============================================================
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
predictions['Race'] = RACE_NAME
predictions['Timestamp'] = timestamp

output_file = f'predictions/race_{RACE_NUMBER:02d}_{RACE_NAME.replace(" ", "_").lower()}_prediction.csv'
predictions.to_csv(output_file, index=False)

print(f"\n✅ Predictions saved to: {output_file}")
print(f"{'='*60}\n")