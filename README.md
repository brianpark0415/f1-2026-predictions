# F1 2026 Podium Predictor 🏎️ 

Machine learning model that predicts Formula 1 race podium finishers for the 2026 season races using XGBoost trained on historical race data (2022-2025).

## Project Overview

This project was built to predict podium results for the **2026 F1 season**, which features the biggest regulation changes in F1 history. The model explicitly handles the uncertainty of a regulation reset by:
- Weighting recent seasons more heavily in training (2025 = 10x, 2024 = 5x, 2023 = 2x, 2022 = 1x)
- Dampening constructor advantages early in the season
- Adapting predictions as 2026 race data accumulates

---

## Model Performance

Trained on 1,838 driver-race records from 2022-2025:

| Metric | 2024 Validation | 2025 Test |
|--------|----------------|-----------|
| **Average correct podium drivers** | 1.71 / 3 | 1.79 / 3 |
| **Perfect podium predictions** | 12.5% | 12.5% |
| **Mean Absolute Error** | 2.98 positions | 3.29 positions |
| **Baseline (top-3 qualifiers)** | ~1.5 / 3 | ~1.5 / 3 |

**Key Finding:** Model achieves ~60% accuracy on podium prediction, beating naive baselines by leveraging driver form, constructor strength, and circuit-specific history.

---

## Technical Details

### Features Engineered (12 total)
- **Qualifying Performance:** Grid position, quali gap to pole
- **Driver Form:** Rolling avg finish/quali over last 5 races, recent points
- **Constructor Strength:** Team championship position, recent form, podium rate
- **Circuit History:** Driver's avg finish at a specific circuit
- **Reliability:** DNF rate, podium rate

### Model Architecture
- **Algorithm:** XGBoost Regressor (predicts finish position, then ranks)
- **Training:** Sample-weighted (2023 = 3x weight, 2022 = 1x) to prioritise recent data
- **Validation:** Time-based splits (never train on future to predict past)
- **Regulation Handling:** Constructor features dampened 40% for early 2026 races

### Tech Stack
- `fastf1` - Official F1 data API
- `xgboost` - Gradient boosting model
- `pandas` / `numpy` - Data processing
- `scikit-learn` - ML utilities
- `matplotlib` / `seaborn` - Visualisation

---

## Project Structure
```
f1_2026_predictions/
├── data/
│   ├── fastf1_cache/              # Cached F1 data
│   ├── race_results_2022_2025.csv # Raw race results
│   ├── qualifying_2022_2025.csv   # Qualifying times
│   └── features_2022_2025.csv     # Feature matrix
├── notebooks/
│   ├── 01_data_collection.ipynb   # FastF1 data gathering
│   ├── 02_feature_engineering.ipynb # Feature creation & analysis
│   └── 03_modeling.ipynb          # Model training & evaluation
├── models/
│   ├── xgb_f1_podium_predictor.pkl # Trained model
│   └── feature_list.txt           # Model feature names
├── predictions/
│   └── race_01_australia_2026.py  # Race prediction script
└── README.md
```

---

## How to Use

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/f1_2026_predictions.git
cd f1_2026_predictions

# Install dependencies
pip install fastf1 pandas numpy scikit-learn xgboost matplotlib seaborn jupyter joblib
```

### Run Prediction for 2026 Race
```bash
cd predictions
python race_01_australia_2026.py
```

**Output:**
```
🏁  PREDICTED PODIUM - Australian Grand Prix
🥇  RUS - Mercedes             (Grid: P7)
🥈  VER - Red Bull Racing      (Grid: P1)
🥉  NOR - McLaren              (Grid: P2)
```

### Retrain Model (after new 2026 races)
1. Update `data/features_2022_2025.csv` with new race results
2. Run `notebooks/03_modeling.ipynb`
3. Model automatically saved to `models/`

---

## 🔍 Key Insights

### Feature Importance
Top predictive features discovered:
1. **Qualifying Position** (0.65 correlation) - Starting position is critical
2. **Recent Form** (0.60) - Last 5 races matter more than season-long stats
3. **Constructor Strength** (0.53) - Car quality determines ceiling
4. **Circuit History** (0.47) - Track-specific knowledge persists

### 2026 Regulation Impact
The model handles the unprecedented 2026 regulation reset by:
- Reducing trust in constructor hierarchy early in the season (40% dampening)
- Prioritising driver skill metrics over car-dependent features
- Updating confidence as real 2026 data accumulates race-by-race

---

## 📈 Future Improvements

- [ ] Incorporate weather data
- [ ] Add pit stop strategy modelling
- [ ] Real-time updates during race weekends
- [ ] Web dashboard for interactive predictions
- [ ] Ensemble multiple models (XGBoost + LightGBM + Neural Net)
- [ ] Sentiment analysis from team radio / press conferences

---

## 📄 License

MIT License - feel free to use for learning or personal projects

---

## 🙏 Acknowledgments

- **FastF1** - Amazing open-source F1 data library

---

**Built by Brian Park**  
February 2026
