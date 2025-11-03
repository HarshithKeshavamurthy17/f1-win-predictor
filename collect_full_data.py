#!/usr/bin/env python3
"""
Robust F1 Data Collector
Collects complete historical data from 1995-2024 using Ergast API
"""
import requests
import pandas as pd
from pathlib import Path
import time
import json

def collect_season_results(year):
    """Collect all race results for a given season"""
    url = f"http://ergast.com/api/f1/{year}/results.json"
    params = {"limit": 1000}
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        races = data["MRData"]["RaceTable"]["Races"]
        
        rows = []
        for race in races:
            round_num = int(race["round"])
            race_name = race["raceName"]
            circuit = race["Circuit"]["circuitName"]
            date = race["date"]
            
            for result in race["Results"]:
                driver = result["Driver"]
                constructor = result["Constructor"]
                
                rows.append({
                    "season": year,
                    "round": round_num,
                    "race_name": race_name,
                    "circuit": circuit,
                    "date": date,
                    "driver_id": driver["driverId"],
                    "driver_code": driver.get("code", ""),
                    "driver_name": f"{driver['givenName']} {driver['familyName']}",
                    "constructor": constructor["name"],
                    "constructor_id": constructor["constructorId"],
                    "grid": int(result["grid"]),
                    "position": result.get("position", ""),
                    "position_text": result["positionText"],
                    "points": float(result["points"]),
                    "laps": int(result["laps"]),
                    "status": result["status"],
                    "finish_pos": int(result["position"]) if result.get("position") else 999,
                    "wins_flag": 1 if result.get("position") == "1" else 0,
                })
        
        df = pd.DataFrame(rows)
        print(f"✅ {year}: {len(df)} results from {len(races)} races")
        return df
        
    except Exception as e:
        print(f"❌ {year}: {str(e)}")
        return None

def main():
    print("=" * 60)
    print("🏎️  F1 DATA COLLECTION (1995-2024)")
    print("=" * 60)
    print()
    
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_seasons = []
    
    # Collect data for each season
    for year in range(1995, 2025):
        print(f"📥 Collecting {year}...", end=" ")
        
        df = collect_season_results(year)
        
        if df is not None and len(df) > 0:
            # Save individual season file
            df.to_parquet(output_dir / f"results_{year}.parquet", index=False)
            all_seasons.append(df)
        
        # Be nice to the API
        time.sleep(0.5)
    
    # Combine all seasons
    if all_seasons:
        combined = pd.concat(all_seasons, ignore_index=True)
        combined.to_parquet(output_dir / "results.parquet", index=False)
        
        print()
        print("=" * 60)
        print("✅ DATA COLLECTION COMPLETE!")
        print("=" * 60)
        print(f"📊 Total Results: {len(combined):,}")
        print(f"🏁 Seasons: {combined['season'].min()}-{combined['season'].max()}")
        print(f"👤 Unique Drivers: {combined['driver_id'].nunique()}")
        print(f"🏆 Unique Teams: {combined['constructor'].nunique()}")
        print(f"🎯 Total Races: {len(combined.groupby(['season', 'round']))}")
        print(f"💾 Saved to: {output_dir / 'results.parquet'}")
        print("=" * 60)
        
        # Show sample
        print("\n📋 Sample Data:")
        print(combined.head(10).to_string())
        
    else:
        print("❌ No data collected!")

if __name__ == "__main__":
    main()

