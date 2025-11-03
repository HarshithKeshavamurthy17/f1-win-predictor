"""
F1 Feature Engineering - Improved Version
==========================================

This file creates all the features our machine learning model needs to predict race winners.
Think of features as the "clues" the model uses to make predictions.

Based on real F1 racing factors:
- Starting position (grid) is HUGE - pole position wins ~40% of races
- Recent form matters - drivers on a hot streak keep winning
- Team performance - dominant teams keep dominating
- Driver experience - veterans know how to manage races
- Reliability - some drivers/teams have more mechanical failures
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from .utils import ensure_dir


def add_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add ALL the features that help predict F1 race winners
    
    This is where the magic happens! We transform raw race data into 
    meaningful patterns the model can learn from.
    
    Args:
        df: Raw race data with basics like driver, team, grid position, points
        
    Returns:
        Enhanced dataframe with 20+ predictive features
    """
    
    # Make a copy so we don't mess up the original data
    df = df.copy()
    
    # Sort by driver and time (season, race) so we can look at history
    # This is crucial - we need to see what happened BEFORE each race
    df = df.sort_values(['driver_id', 'season', 'round']).reset_index(drop=True)
    
    print("🏎️  Building F1 prediction features...")
    print("=" * 60)
    
    # ============================================================
    # FEATURE 1: Driver Experience
    # ============================================================
    # Research shows: More experienced drivers manage races better
    # They know when to push, when to save tires, when to pit
    
    print("📊 Feature 1: Calculating driver experience...")
    df['driver_start'] = 1  # Each race is one start
    
    # Count how many races this driver has done BEFORE this one
    # Example: Hamilton's 300th race means he has 299 races of experience
    df['exp_starts'] = (
        df.groupby('driver_id')['driver_start']
        .cumsum()  # Running total of races
        .shift(1)  # Shift down so we don't include current race
        .fillna(0)  # First race has 0 experience
    )
    
    # ============================================================
    # FEATURE 2: Team Changes
    # ============================================================
    # Research shows: Switching teams hurts performance initially
    # Drivers need time to adapt to new car, new team, new procedures
    
    print("🔄 Feature 2: Detecting team changes...")
    df['team_change'] = (
        df.groupby('driver_id')['constructor']
        .transform(lambda x: (x != x.shift(1)).astype(int))  # 1 if team changed, 0 if same
    )
    
    # ============================================================
    # FEATURE 3: Team Tenure
    # ============================================================
    # Longer with same team = better car understanding and setup
    
    print("⏱️  Feature 3: Calculating time with current team...")
    df['tenure'] = (
        df.groupby(['driver_id', 'constructor'])['driver_start']
        .cumsum()  # How many races with THIS team
    )
    
    # ============================================================
    # FEATURE 4: Recent Form (Last 5 Races)
    # ============================================================
    # This is CRITICAL! Hot streaks in F1 are real
    # Winning breeds confidence, momentum, and more wins
    
    print("🔥 Feature 4: Measuring recent form (last 5 races)...")
    
    # Get points from previous race (we can't use current race points!)
    df['points_lag'] = (
        df.groupby('driver_id')['points']
        .shift(1)  # Previous race points
        .fillna(0)
    )
    
    # Average points over last 5 races = form indicator
    df['form_5'] = (
        df.groupby('driver_id')['points_lag']
        .rolling(window=5, min_periods=1)  # Last 5 races (or fewer at season start)
        .mean()
        .reset_index(level=0, drop=True)
    )
    
    # ============================================================
    # FEATURE 5: Championship Standing
    # ============================================================
    # Points accumulated so far this season
    # Championship leaders often win more (confidence, team support)
    
    print("🏆 Feature 5: Championship points up to this race...")
    df['driver_points_at_stage_of_season'] = (
        df.groupby(['driver_id', 'season'])['points']
        .cumsum()  # Running total this season
        .shift(1)  # Don't include current race
        .fillna(0)
    )
    
    # ============================================================
    # FEATURE 6: Team Championship Standing
    # ============================================================
    # Dominant teams have better cars, more resources, better strategy
    
    print("🏗️  Feature 6: Team championship points...")
    
    # Sum up team points for each race
    team_race_points = (
        df.groupby(['season', 'round', 'constructor'])['points']
        .sum()
        .reset_index(name='team_race_total')
    )
    
    # Running total of team points throughout season
    team_race_points['constructorId_points_at_stage_of_season'] = (
        team_race_points.groupby(['season', 'constructor'])['team_race_total']
        .cumsum()
        .shift(1)
        .fillna(0)
    )
    
    # Merge team points back to main dataframe
    df = df.merge(
        team_race_points[['season', 'round', 'constructor', 'constructorId_points_at_stage_of_season']],
        on=['season', 'round', 'constructor'],
        how='left'
    )
    df['constructorId_points_at_stage_of_season'] = df['constructorId_points_at_stage_of_season'].fillna(0)
    
    # ============================================================
    # FEATURE 7: Race Number in Season
    # ============================================================
    # Early season vs late season dynamics are different
    # Early: everyone's optimistic, late: championship pressure
    
    print("📅 Feature 7: Race position in season (1st, 2nd, 3rd race...)...")
    df['race_number_in_season'] = (
        df.groupby(['season', 'round'])['round']
        .transform('first')  # Round number = race number
    )
    
    # ============================================================
    # FEATURE 8: Grid Position Delta
    # ============================================================
    # How far behind pole position?
    # Being 0.5 seconds behind in qualifying = big disadvantage
    
    print("🏁 Feature 8: Gap from pole position...")
    df['grid_delta_from_pole'] = (
        df.groupby(['season', 'round'])['grid']
        .transform(lambda x: x - x.min())  # Everyone's gap from P1
    )
    
    # ============================================================
    # FEATURE 9: Team Recent Wins
    # ============================================================
    # Team momentum - winning teams keep winning (car development cycle)
    
    print("✨ Feature 9: Team's recent wins (last 10 races)...")
    df['constructor_wins_last_10'] = (
        df.sort_values(['constructor', 'season', 'round'])
        .groupby('constructor')['wins_flag']
        .rolling(window=10, min_periods=1)
        .sum()
        .shift(1)  # Don't include current race
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    
    # ============================================================
    # NEW FEATURE 10: Qualifying Performance Trend
    # ============================================================
    # Getting better/worse qualifying positions = form indicator
    
    print("📈 Feature 10: Qualifying position trend...")
    df['grid_improving'] = (
        df.groupby('driver_id')['grid']
        .diff()  # Positive = worse position, negative = improving
        .fillna(0)
        * -1  # Flip so positive = improving
    )
    
    # ============================================================
    # NEW FEATURE 11: Driver vs Teammate Performance
    # ============================================================
    # Beating your teammate = you're getting most out of the car
    
    print("👥 Feature 11: Performance vs teammate...")
    
    # Find teammate (other driver with same constructor in same race)
    df['teammate_points'] = (
        df.groupby(['season', 'round', 'constructor'])['points']
        .transform(lambda x: x.shift(-1) if len(x) > 1 else 0)
    )
    
    # Points advantage over teammate
    df['points_vs_teammate'] = (df['points'] - df['teammate_points']).fillna(0)
    
    # ============================================================
    # NEW FEATURE 12: Recent Podiums
    # ============================================================
    # Podium finishes (top 3) in last 5 races = consistency indicator
    
    print("🥇 Feature 12: Recent podium count...")
    df['podium_flag'] = (df['points'] >= 15).astype(int)  # Top 3 usually get 15+ points
    df['podiums_last_5'] = (
        df.groupby('driver_id')['podium_flag']
        .rolling(window=5, min_periods=1)
        .sum()
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    
    # ============================================================
    # NEW FEATURE 13: Reliability Score
    # ============================================================
    # Finishing races consistently = reliable driver/car combo
    
    print("🔧 Feature 13: Reliability score...")
    df['finished_race'] = (df['dnf'] == 0).astype(int)
    df['reliability_last_10'] = (
        df.groupby('driver_id')['finished_race']
        .rolling(window=10, min_periods=1)
        .mean()
        .shift(1)
        .reset_index(level=0, drop=True)
        .fillna(1.0)  # Assume reliable at start
    )
    
    # ============================================================
    # NEW FEATURE 14: Championship Battle Pressure
    # ============================================================
    # Being in title fight adds pressure/motivation
    
    print("💪 Feature 14: Championship pressure indicator...")
    
    # Calculate gap to championship leader
    season_leaders = (
        df.groupby(['season', 'round', 'driver_id'])['driver_points_at_stage_of_season']
        .first()
        .reset_index()
    )
    season_leaders['season_leader_points'] = (
        season_leaders.groupby(['season', 'round'])['driver_points_at_stage_of_season']
        .transform('max')
    )
    
    df = df.merge(
        season_leaders[['season', 'round', 'driver_id', 'season_leader_points']],
        on=['season', 'round', 'driver_id'],
        how='left'
    )
    
    df['points_from_leader'] = (
        df['season_leader_points'] - df['driver_points_at_stage_of_season']
    ).fillna(0)
    
    # In title fight if within 50 points of leader
    df['in_title_fight'] = (df['points_from_leader'] <= 50).astype(int)
    
    # ============================================================
    # Encode categorical variables (team and driver names to numbers)
    # ============================================================
    # Machine learning models need numbers, not text
    
    print("🔢 Encoding team and driver names to numbers...")
    df['constructor_cat'] = df['constructor'].astype('category').cat.codes
    df['driver_cat'] = df['driver_id'].astype('category').cat.codes
    
    # ============================================================
    # Target variable (what we're trying to predict)
    # ============================================================
    df['y_win'] = df['wins_flag'].astype(int)
    
    print("=" * 60)
    print("✅ All features created successfully!")
    print(f"📊 Total features: {df.shape[1]}")
    
    return df


def main(inp: str, out: str):
    """
    Main function - loads data, creates features, saves enhanced data
    
    This is what runs when you execute: python -m src.features_improved
    """
    
    # Setup paths
    in_path = Path(inp) / "f1_training_base.parquet"
    out_dir = ensure_dir(out)
    
    print("\n🏎️  F1 FEATURE ENGINEERING PIPELINE")
    print("=" * 60)
    print(f"📥 Loading data from: {in_path}")
    
    # Load the preprocessed data
    df = pd.read_parquet(in_path)
    print(f"📊 Loaded {len(df):,} race results")
    
    # Create all the features!
    feat = add_all_features(df)
    
    # Select only the columns we need for training
    # These are all the features we created + some basic info
    keep = [
        # Basic race info
        "season", "round", "race_name", 
        
        # Driver/team identifiers
        "driver_id", "driver_cat", "constructor_cat",
        
        # Qualifying and race basics
        "grid", "points", "dnf",
        
        # Experience features
        "exp_starts", "tenure", "team_change",
        
        # Form and momentum features
        "form_5", "podiums_last_5", "reliability_last_10",
        
        # Championship position features
        "driver_points_at_stage_of_season",
        "constructorId_points_at_stage_of_season",
        "points_from_leader", "in_title_fight",
        
        # Race context features
        "race_number_in_season", "grid_delta_from_pole",
        
        # Team performance features
        "constructor_wins_last_10",
        
        # Performance indicators
        "grid_improving", "points_vs_teammate",
        
        # Target variable (what we predict)
        "y_win"
    ]
    
    # Save the feature-rich dataset
    output_path = out_dir / "f1_training.parquet"
    feat[keep].to_parquet(output_path, index=False)
    
    print(f"💾 Saved {len(keep)} features to: {output_path}")
    print("✅ Feature engineering complete!\n")


if __name__ == "__main__":
    # Command line argument parser
    # Usage: python -m src.features_improved --in data/curated --out data/curated
    
    ap = argparse.ArgumentParser(description="Create F1 prediction features")
    ap.add_argument("--in", dest="inp", required=True, help="Input directory with preprocessed data")
    ap.add_argument("--out", required=True, help="Output directory for feature-rich data")
    a = ap.parse_args()
    
    main(a.inp, a.out)

