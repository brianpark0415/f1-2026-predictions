import pandas as pd

# ============================================================
# CONFIGURATION - Edit this section after each race
# ============================================================

RACE_NUM = 3  # Japan
RACE_NAME = "Japanese Grand Prix"
CIRCUIT = "Suzuka"

# Qualifying (from Saturday)
QUALI_GRID = ['ANT', 'RUS', 'PIA', 'LEC', 'NOR', 'HAM', 'GAS', 'HAD', 'BOR',
              'LIN', 'VER', 'OCO', 'HUL', 'LAW', 'COL', 'SAI', 'ALB', 'BEA',
              'PER', 'BOT', 'ALO', 'STR']

QUALI_TIMES = [88.778, 89.076, 89.132, 89.405, 89.409, 89.567, 89.691, 89.978,
               90.274, 90.319, 90.519, 90.675, 90.858, 90.881, 90.931, 90.927,
               91.074, 91.090, 92.206, 92.330, 92.646, 92.920]

# Race results (from Sunday)
FINISHING_ORDER = ['ANT', 'PIA', 'LEC', 'RUS', 'NOR', 'HAM', 'GAS', 'VER', 'LAW',
                   'OCO', 'HUL', 'HAD', 'BOR', 'LIN', 'SAI', 'COL', 'PER', 'ALO',
                   'BOT', 'ALB', 'STR', 'BEA']

DNF_DRIVERS = ['STR', 'BEA']  # Stroll and Bearman DNF'd

# ============================================================
# AUTO-GENERATE DATA (Don't edit below this line)
# ============================================================

DRIVER_TEAMS = {
    'ANT': 'Mercedes', 'RUS': 'Mercedes',
    'HAM': 'Ferrari', 'LEC': 'Ferrari',
    'NOR': 'McLaren', 'PIA': 'McLaren',
    'VER': 'Red Bull Racing', 'HAD': 'Red Bull Racing',
    'LAW': 'Racing Bulls', 'LIN': 'Racing Bulls',
    'HUL': 'Audi', 'BOR': 'Audi',
    'BEA': 'Haas', 'OCO': 'Haas',
    'GAS': 'Alpine', 'COL': 'Alpine',
    'SAI': 'Williams', 'ALB': 'Williams',
    'ALO': 'Aston Martin', 'STR': 'Aston Martin',
    'PER': 'Cadillac', 'BOT': 'Cadillac'
}

DRIVER_NUMBERS = {
    'ANT': 12, 'RUS': 63, 'HAM': 44, 'LEC': 16, 'NOR': 4, 'PIA': 81,
    'VER': 1, 'HAD': 25, 'LAW': 30, 'LIN': 35, 'HUL': 87, 'BOR': 27,
    'BEA': 50, 'OCO': 31, 'GAS': 10, 'COL': 43, 'SAI': 2, 'ALB': 23,
    'ALO': 14, 'STR': 18, 'PER': 55, 'BOT': 77
}

POINTS_MAP = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1] + [0] * 12

print(f"\nAdding Race {RACE_NUM}: {RACE_NAME}...\n")

# Build race results
race_results = {
    'Year': [2026] * len(FINISHING_ORDER),
    'RaceNumber': [RACE_NUM] * len(FINISHING_ORDER),
    'EventName': [RACE_NAME] * len(FINISHING_ORDER),
    'CircuitName': [CIRCUIT] * len(FINISHING_ORDER),
    'DriverNumber': [DRIVER_NUMBERS[d] for d in FINISHING_ORDER],
    'Abbreviation': FINISHING_ORDER,
    'TeamName': [DRIVER_TEAMS[d] for d in FINISHING_ORDER],
    'GridPosition': [QUALI_GRID.index(d) + 1 for d in FINISHING_ORDER],
    'Position': [(None if d in DNF_DRIVERS else i + 1.0) for i, d in enumerate(FINISHING_ORDER)],
    'Status': [('Retired' if d in DNF_DRIVERS else 'Finished') for d in FINISHING_ORDER],
    'Points': [(0.0 if d in DNF_DRIVERS else POINTS_MAP[i]) for i, d in enumerate(FINISHING_ORDER)]
}

# Add to dataset
df_race = pd.DataFrame(race_results)
existing_race = pd.read_csv('data/race_results_2022_2025.csv')
updated_race = pd.concat([existing_race, df_race], ignore_index=True)
updated_race.to_csv('data/race_results_2022_2025.csv', index=False)
print(f"✅ Added race results! Total races: {len(updated_race)}")

# Build qualifying results
quali_results = {
    'Year': [2026] * len(QUALI_GRID),
    'RaceNumber': [RACE_NUM] * len(QUALI_GRID),
    'Abbreviation': QUALI_GRID,
    'QualiPosition': list(range(1, len(QUALI_GRID) + 1)),
    'BestQualiTime': QUALI_TIMES
}

df_quali = pd.DataFrame(quali_results)
existing_quali = pd.read_csv('data/qualifying_2022_2025.csv')
updated_quali = pd.concat([existing_quali, df_quali], ignore_index=True)
updated_quali.to_csv('data/qualifying_2022_2025.csv', index=False)
print(f"✅ Added qualifying! Total sessions: {len(updated_quali)}")

print(f"\n✅ Done! Re-run feature engineering notebook.\n")