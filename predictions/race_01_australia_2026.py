"""
2026 Australian Grand Prix - Podium Prediction
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
RACE_NUMBER = 1
RACE_NAME = "Australian Grand Prix"
CIRCUIT = "Melbourne"

print(f"\n{'='*60}")
print(f"🏎️  {RACE_NAME} - Podium Prediction")
print(f"{'='*60}\n")

# ============================================================
# LOAD MODEL
# ============================================================
print("Loading trained model...")
model = joblib.load('../models/xgb_f1_podium_predictor.pkl')

with open('../models/feature_list.txt', 'r') as f:
    feature_cols = [line.strip() for line in f]

print(f"✅ Model loaded with {len(feature_cols)} features\n")

# ============================================================
# GET QUALIFYING RESULTS
# ============================================================
print(f"Fetching {YEAR} qualifying results...")
fastf1.Cache.enable_cache('../data/fastf1_cache')

# ============================================================
# GET QUALIFYING RESULTS
# ============================================================
print(f"Fetching {YEAR} qualifying results...")
fastf1.Cache.enable_cache('../data/fastf1_cache')

# For 2026, qualifying hasn't happened yet, so we ALWAYS use sample data
print("⚠️  2026 qualifying data not available yet")
print("Using sample data based on 2025 form...\n")

# Create sample quali data
quali = pd.DataFrame({
    'Driver': ['VER', 'NOR', 'LEC', 'PIA', 'SAI', 'HAM', 'RUS', 'ALO', 
               'GAS', 'HUL', 'TSU', 'OCO', 'ALB', 'PER', 'BOT', 'MAG',
               'HAD', 'BOR', 'ANT', 'DOO'],
    'Team': ['Red Bull Racing', 'McLaren', 'Ferrari', 'McLaren', 'Williams',
             'Ferrari', 'Mercedes', 'Aston Martin', 'Alpine', 'Audi',
             'Racing Bulls', 'Alpine', 'Williams', 'Cadillac', 'Cadillac',
             'Haas', 'Racing Bulls', 'Audi', 'Mercedes', 'Haas'],
    'QualiPosition': list(range(1, 21)),
    'GridPosition': list(range(1, 21)),
    'BestQualiTime': [80.0 + i*0.3 for i in range(20)],
})
quali['QualiGapToPole'] = quali['BestQualiTime'] - quali['BestQualiTime'].min()

print(f"✅ Sample qualifying data created")
print(f"   {len(quali)} drivers")
print(quali[['Driver', 'Team', 'QualiPosition']].head(10))
print()
# ============================================================
# LOAD 2025 END-OF-SEASON STATS (Baseline for 2026 Race 1)
# ============================================================
print("Loading 2025 season statistics...")

# Load historical data to calculate baseline features
historical = pd.read_csv('../data/features_2022_2025.csv')
end_of_2025 = historical[historical['Year'] == 2025].copy()

# Calculate each driver's 2025 end-of-season stats
driver_stats = end_of_2025.groupby('Abbreviation').agg({
    'Position': 'mean',
    'QualiPosition': 'mean',
    'Points': 'sum',
    'DNFRateSeason': 'last',
    'PodiumRateSeason': 'last',
}).reset_index()

driver_stats.columns = ['Driver', 'AvgFinish2025', 'AvgQuali2025', 
                        'TotalPoints2025', 'DNFRate2025', 'PodiumRate2025']

# Constructor stats from 2025
constructor_stats = end_of_2025.groupby('TeamName').agg({
    'Position': 'mean',
    'ConstructorChampPos': 'last',
    'ConstructorPodiumRateSeason': 'last',
}).reset_index()

constructor_stats.columns = ['Team', 'ConstructorAvgFinish2025', 
                              'ConstructorChampPos2025', 'ConstructorPodiumRate2025']

print(f"✅ Loaded stats for {len(driver_stats)} drivers and {len(constructor_stats)} teams\n")

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
# Build feature matrix matching what model expects
# For race 1, we use 2025 end-of-season stats as baseline

features['AvgFinishLast5'] = features['AvgFinish2025']
features['AvgQualiLast5'] = features['AvgQuali2025']
features['PointsLast3'] = features['TotalPoints2025'] / 8  # Rough proxy
features['ConstructorAvgFinishLast3'] = features['ConstructorAvgFinish2025']
features['ConstructorPodiumRateSeason'] = features['ConstructorPodiumRate2025']
features['DriverCircuitAvgFinish'] = features['AvgFinish2025']  # No circuit history yet
features['DNFRateSeason'] = features['DNFRate2025']
features['PodiumRateSeason'] = features['PodiumRate2025']
features['ConstructorChampPos'] = features['ConstructorChampPos2025']

# Handle new drivers/teams (Cadillac, rookies) - fill with median values
for col in feature_cols:
    if col not in features.columns:
        print(f"⚠️  Missing feature: {col}, filling with median")
        features[col] = 11  # Midfield default
    else:
        features[col] = features[col].fillna(features[col].median())

# ============================================================
# 2026 REGULATION UNCERTAINTY ADJUSTMENT
# ============================================================
# For race 1, constructor hierarchy is uncertain due to reg changes
# Dampen constructor advantage toward midfield (5.5 = neutral)

REGULATION_UNCERTAINTY = 0.4  # 40% trust in 2025 constructor form

features['ConstructorChampPos'] = (
    features['ConstructorChampPos'] * REGULATION_UNCERTAINTY + 
    5.5 * (1 - REGULATION_UNCERTAINTY)
)

print("Applied 2026 regulation uncertainty adjustment")
print("(Constructor advantages dampened for early season)\n")

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

output_file = f'../predictions/race_{RACE_NUMBER:02d}_{RACE_NAME.replace(" ", "_").lower()}_prediction.csv'
predictions.to_csv(output_file, index=False)

print(f"\n✅ Predictions saved to: {output_file}")
print(f"{'='*60}\n")