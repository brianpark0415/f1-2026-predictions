import pandas as pd

print("Adding 2026 Australian Grand Prix results...\n")

# ============================================================
# RACE RESULTS - Use correct column names!
# ============================================================

race_1_results = {
    'Year': [2026] * 22,
    'RaceNumber': [1] * 22,
    'EventName': ['Australian Grand Prix'] * 22,
    'CircuitName': ['Albert Park'] * 22,
    'DriverNumber': [63, 12, 16, 4, 81, 1, 44, 25, 30, 35, 27, 87, 50, 31, 10, 43, 14, 55, 77, 18, 2, 23],  # Fill in actual numbers
    'Abbreviation': ['RUS', 'ANT', 'LEC', 'NOR', 'PIA', 'VER', 'HAM', 'HAD', 'LAW', 
                     'LIN', 'BOR', 'HUL', 'BEA', 'OCO', 'GAS', 'COL', 'ALO', 'PER', 
                     'BOT', 'STR', 'SAI', 'ALB'],
    'TeamName': ['Mercedes', 'Mercedes', 'Ferrari', 'McLaren', 'McLaren', 
                 'Red Bull Racing', 'Ferrari', 'Red Bull Racing', 'Racing Bulls',
                 'Racing Bulls', 'Audi', 'Audi', 'Haas', 'Haas', 'Alpine', 
                 'Alpine', 'Aston Martin', 'Cadillac', 'Cadillac', 'Aston Martin', 
                 'Williams', 'Williams'],
    'GridPosition': [1, 2, 4, 6, 5, 20, 7, 3, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 22, 21, 15],
    'Position': [1.0, 2.0, 3.0, 4.0, None, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0],
    'Status': ['Finished', 'Finished', 'Finished', 'Finished', 'Retired', 
               'Finished', 'Finished', 'Finished', 'Finished', 'Finished',
               'Finished', 'Finished', 'Finished', 'Finished', 'Finished',
               'Finished', 'Finished', 'Finished', 'Finished', 'Finished',
               'Finished', 'Finished'],
    'Points': [25.0, 18.0, 15.0, 12.0, 0.0, 8.0, 6.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
}

df_race1 = pd.DataFrame(race_1_results)
existing_race = pd.read_csv('data/race_results_2022_2025.csv')
updated_race = pd.concat([existing_race, df_race1], ignore_index=True)
updated_race.to_csv('data/race_results_2022_2025.csv', index=False)

print("✅ Added race results!")
print(f"   Total races: {len(updated_race)}")

# ============================================================
# QUALIFYING RESULTS
# ============================================================

quali_1 = {
    'Year': [2026] * 22,
    'RaceNumber': [1] * 22,
    'Abbreviation': ['RUS', 'ANT', 'HAD', 'LEC', 'PIA', 'NOR', 'HAM', 'LAW', 'LIN', 
                     'BOR', 'HUL', 'BEA', 'OCO', 'GAS', 'ALB', 'COL', 'ALO', 'PER', 
                     'BOT', 'VER', 'SAI', 'STR'],
    'QualiPosition': list(range(1, 23)),
    'BestQualiTime': [88.500, 88.656, 88.789, 88.912, 89.023, 89.098, 89.201, 
                      89.345, 89.434, 89.623, 89.701, 89.856, 89.945, 90.034, 
                      90.178, 90.323, 90.445, 90.634, 90.789, 
                      None, None, None]
}

df_quali1 = pd.DataFrame(quali_1)
existing_quali = pd.read_csv('data/qualifying_2022_2025.csv')
updated_quali = pd.concat([existing_quali, df_quali1], ignore_index=True)
updated_quali.to_csv('data/qualifying_2022_2025.csv', index=False)

print("✅ Added qualifying results!")
print(f"   Total qualifying sessions: {len(updated_quali)}")
print("\n✅ Done! Now re-run feature engineering notebook.\n")