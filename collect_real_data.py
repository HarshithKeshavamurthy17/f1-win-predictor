#!/usr/bin/env python3
"""
Collect real F1 data from Ergast API (2000-2024)
Using the user's proven method
"""
import requests
import pandas as pd
import time
from pathlib import Path

# Function to fetch race results for a given year and round
def fetch_race_results(year, round_num):
    url = f"http://ergast.com/api/f1/{year}/{round_num}/results.json"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        races = data['MRData']['RaceTable']['Races']
        if not races:
            return None
        race = races[0]
        results = race['Results']
        race_results = []
        for result in results:
            result_info = {
                'season': year,
                'race_name': race['raceName'],
                'round': int(race['round']),
                'driver_id': result['Driver']['driverId'],
                'driver_name': f"{result['Driver']['givenName']} {result['Driver']['familyName']}",
                'driver_code': result['Driver'].get('code', ''),
                'constructor': result['Constructor']['name'],
                'constructor_id': result['Constructor']['constructorId'],
                'position': int(result['position']) if result.get('position') else 999,
                'finish_pos': int(result['position']) if result.get('position') else 999,
                'grid': int(result['grid']),
                'laps': int(result['laps']),
                'status': result['status'],
                'points': float(result['points']),
                'wins_flag': 1 if result.get('position') == '1' else 0
            }
            race_results.append(result_info)
        return pd.DataFrame(race_results)
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return None

# Initialize an empty DataFrame to store all race results for multiple seasons
all_race_results_df = pd.DataFrame()

# Define the range of years you want to fetch data for (2000-2024)
years = list(range(2000, 2025))  # From 2000 to 2024

print("=" * 60)
print("🏎️  COLLECTING REAL F1 DATA FROM ERGAST API")
print("=" * 60)
print(f"📅 Years: {years[0]}-{years[-1]}")
print("=" * 60)
print()

# Fetch race results for each season and each race round
for year in years:
    # Fetch the races for the current year
    races_url = f"http://ergast.com/api/f1/{year}.json"
    try:
        races_response = requests.get(races_url, timeout=10)
        races_data = races_response.json()
        races = races_data['MRData']['RaceTable']['Races']
        
        print(f"📥 {year}: {len(races)} races")
        
        # Fetch race results for each round in the current year
        for race in races:
            round_num = race['round']
            print(f"  → Round {round_num}: {race['raceName']}", end=" ")
            race_result = fetch_race_results(year, round_num)
            if race_result is not None:
                all_race_results_df = pd.concat([all_race_results_df, race_result], ignore_index=True)
                print(f"✅ ({len(race_result)} drivers)")
            else:
                print("❌")
            
            # Be nice to the API
            time.sleep(0.3)
    
    except Exception as e:
        print(f"❌ {year}: Failed to fetch season - {e}")
        continue

print()
print("=" * 60)
print("✅ DATA COLLECTION COMPLETE!")
print("=" * 60)

if len(all_race_results_df) > 0:
    # Display summary
    print(f"📊 Total Results: {len(all_race_results_df):,}")
    print(f"🏁 Seasons: {all_race_results_df['season'].min()}-{all_race_results_df['season'].max()}")
    print(f"🏆 Total Races: {len(all_race_results_df.groupby(['season', 'round']))}")
    print(f"👤 Unique Drivers: {all_race_results_df['driver_id'].nunique()}")
    print(f"🏎️ Unique Teams: {all_race_results_df['constructor'].nunique()}")
    
    # Save to raw data directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save as parquet (better for ML)
    all_race_results_df.to_parquet(output_dir / "results.parquet", index=False)
    print(f"💾 Saved to: {output_dir / 'results.parquet'}")
    
    # Also save as CSV for backup
    all_race_results_df.to_csv(output_dir / 'f1_race_results_2000_2024.csv', index=False)
    print(f"💾 Saved to: {output_dir / 'f1_race_results_2000_2024.csv'}")
    
    # Save per-year files
    for year in all_race_results_df['season'].unique():
        year_df = all_race_results_df[all_race_results_df['season'] == year]
        year_df.to_parquet(output_dir / f"results_{int(year)}.parquet", index=False)
    print(f"💾 Saved per-year files: results_YYYY.parquet")
    
    print()
    print("📋 Sample Data (First 10 rows):")
    print(all_race_results_df.head(10).to_string())
    
    print()
    print("🏆 Top Winners:")
    winners = all_race_results_df[all_race_results_df['wins_flag'] == 1]['driver_id'].value_counts().head(10)
    print(winners.to_string())
    
else:
    print("❌ No data collected!")

