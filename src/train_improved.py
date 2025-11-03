"""
F1 Race Winner Prediction - Model Training
==========================================

This trains a machine learning model to predict F1 race winners.
Think of it like teaching a computer to recognize patterns in racing data.

The model learns from 30 years of F1 history to understand:
- Pole position is a huge advantage
- Recent form matters a lot
- Dominant teams keep winning
- Experience helps in race management
- Reliability is crucial
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from .utils import ensure_dir


# All the features our model will learn from
# These are the "clues" that help predict race winners
FEATURES = [
    # Starting position - THIS IS HUGE in F1!
    "grid", "grid_delta_from_pole",
    
    # Team and driver identity
    "constructor_cat", "driver_cat",
    
    # Recent performance (hot streaks are real!)
    "form_5", "podiums_last_5",
    
    # Experience matters in F1
    "exp_starts", "tenure", "team_change",
    
    # Reliability (DNFs lose races)
    "dnf", "reliability_last_10",
    
    # Previous race points
    "points",
    
    # Championship position (leaders have momentum)
    "driver_points_at_stage_of_season",
    "constructorId_points_at_stage_of_season",
    "points_from_leader", "in_title_fight",
    
    # Race context
    "race_number_in_season",
    
    # Team performance
    "constructor_wins_last_10",
    
    # Performance trends
    "grid_improving", "points_vs_teammate"
]

# What we're trying to predict
TARGET = "y_win"  # 1 = won the race, 0 = didn't win


def build_model(kind: str):
    """
    Create the machine learning model
    
    Different model types have different strengths:
    - Random Forest: Great for capturing non-linear patterns
    - Gradient Boosting: Excellent accuracy, handles imbalanced data well
    - Logistic Regression: Simple baseline, easy to interpret
    
    Args:
        kind: Type of model ("rf", "gbm", or "logreg")
        
    Returns:
        Untrained model ready to learn from data
    """
    
    if kind == "logreg":
        # Logistic Regression - Simple but effective baseline
        # class_weight='balanced' helps with imbalanced data (only 1 winner per race!)
        return LogisticRegression(
            max_iter=300,  # More iterations for convergence
            class_weight='balanced',  # Handle class imbalance
            random_state=42
        )
    
    if kind == "gbm":
        # Gradient Boosting Machine - Usually highest accuracy
        # Builds trees sequentially, each fixing errors of previous ones
        return GradientBoostingClassifier(
            n_estimators=500,  # Number of trees (more = better but slower)
            learning_rate=0.05,  # How much each tree contributes
            max_depth=5,  # Tree depth (deeper = more complex patterns)
            min_samples_split=10,  # Prevent overfitting
            min_samples_leaf=5,  # Prevent overfitting
            subsample=0.8,  # Use 80% of data per tree (helps generalization)
            random_state=42
        )
    
    # Default: Random Forest (balanced speed and accuracy)
    # Builds many trees independently and averages their predictions
    return RandomForestClassifier(
        n_estimators=800,  # Lots of trees for stability
        max_depth=25,  # Deep trees to capture complex patterns
        min_samples_split=5,  # Prevent overfitting
        min_samples_leaf=2,  # Allow fine-grained decisions
        max_features='sqrt',  # Number of features per tree
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1  # Use all CPU cores for speed
    )


def cross_validate_model(model, X, y, groups):
    """
    Test the model properly using cross-validation
    
    We can't just train on all data and test on same data - that's cheating!
    Instead, we use "GroupKFold" which:
    - Splits data by SEASON (not random)
    - Trains on some seasons, tests on others
    - This simulates real prediction: using past to predict future
    
    Why AUC score? It measures how well the model ranks drivers.
    Perfect score = 1.0 means model always ranks the winner highest
    
    Args:
        model: The ML model to test
        X: Features (all the clues)
        y: Target (who won)
        groups: Season numbers (to split by season)
        
    Returns:
        Average AUC score across all test folds
    """
    
    print("\n🧪 Running cross-validation...")
    print("   (Testing model on seasons it hasn't seen before)")
    
    # Split by season - train on some seasons, test on others
    gkf = GroupKFold(n_splits=5)  # 5-fold = test on 20% of seasons each time
    scores = []
    
    fold_num = 1
    for train_idx, test_idx in gkf.split(X, y, groups=groups):
        # Get training and test data for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train the model on training seasons
        model.fit(X_train, y_train)
        
        # Predict probabilities on test seasons
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of winning
        
        # Calculate AUC score (how well did we rank the winners?)
        score = roc_auc_score(y_test, y_pred_proba)
        scores.append(score)
        
        print(f"   Fold {fold_num}/5: AUC = {score:.4f}")
        fold_num += 1
    
    avg_score = np.mean(scores)
    print(f"\n   ✅ Average AUC across all folds: {avg_score:.4f}")
    print(f"   📊 Score range: {min(scores):.4f} to {max(scores):.4f}")
    
    return avg_score


def train_and_predict(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    """
    Train the model and generate predictions for all races
    
    This is the main training pipeline:
    1. Prepare the data
    2. Cross-validate to check performance
    3. Train final model on ALL data
    4. Generate predictions
    5. Pick winner for each race (highest probability)
    
    Args:
        df: Full dataset with all features
        model_name: Type of model to use
        
    Returns:
        Original data with added prediction columns
    """
    
    print("\n" + "=" * 60)
    print(f"🏎️  TRAINING F1 RACE WINNER PREDICTOR ({model_name.upper()})")
    print("=" * 60)
    
    # Prepare features and target
    # Fill any missing values with 0 (rare but can happen)
    X = df[FEATURES].copy().fillna(0)
    y = df[TARGET].astype(int).copy()
    groups = df["season"]  # For season-based cross-validation
    
    print(f"\n📊 Training Data:")
    print(f"   • Total races: {len(df):,}")
    print(f"   • Winners: {y.sum():,}")
    print(f"   • Win rate: {y.mean():.2%}")
    print(f"   • Seasons: {df['season'].min()}-{df['season'].max()}")
    print(f"   • Features used: {len(FEATURES)}")
    
    # Build the model
    model = build_model(model_name)
    print(f"\n🤖 Model: {type(model).__name__}")
    
    # Cross-validate to see how well it works
    try:
        cv_score = cross_validate_model(
            build_model(model_name),  # Fresh model for CV
            X, y, groups
        )
    except Exception as e:
        print(f"\n⚠️  Cross-validation skipped: {e}")
        cv_score = None
    
    # Train final model on ALL data
    print(f"\n🎓 Training final model on all {len(df):,} races...")
    model.fit(X, y)
    print("   ✅ Training complete!")
    
    # Generate predictions for all races
    print("\n🔮 Generating predictions...")
    out = df.copy()
    
    # Get win probability for each driver in each race
    out["p_win"] = model.predict_proba(X)[:, 1]
    
    # Rank drivers within each race by win probability
    # Rank 1 = highest probability (our predicted winner)
    out["pred_rank"] = (
        out.groupby(["season", "round"])["p_win"]
        .rank(method="first", ascending=False)  # Highest probability = rank 1
    )
    
    # Mark our predicted winner (rank 1) for each race
    out["predicted_winner"] = (out["pred_rank"] == 1).astype(int)
    
    # Calculate accuracy on training data
    correct_predictions = ((out["predicted_winner"] == 1) & (out["y_win"] == 1)).sum()
    total_races = (out["predicted_winner"] == 1).sum()
    accuracy = correct_predictions / total_races if total_races > 0 else 0
    
    print(f"\n📈 Training Results:")
    print(f"   • Correctly predicted: {correct_predictions}/{total_races} races")
    print(f"   • Accuracy: {accuracy:.2%}")
    if cv_score:
        print(f"   • Cross-validation AUC: {cv_score:.4f}")
    
    # Show feature importance (which features matter most?)
    if hasattr(model, 'feature_importances_'):
        print(f"\n🔍 Top 10 Most Important Features:")
        importances = pd.DataFrame({
            'feature': FEATURES,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for idx, row in importances.head(10).iterrows():
            print(f"   {idx+1}. {row['feature']}: {row['importance']:.4f}")
    
    print("\n" + "=" * 60)
    print("✅ MODEL TRAINING COMPLETE!")
    print("=" * 60 + "\n")
    
    return out


def main(inp: str, out: str, model_name: str):
    """
    Main function - loads data, trains model, saves predictions
    
    This is what runs when you execute: python -m src.train_improved
    """
    
    # Setup
    out_dir = ensure_dir(out)
    
    # Load the feature-rich training data
    print(f"📥 Loading training data from: {inp}")
    df = pd.read_parquet(inp)
    
    # Train model and get predictions
    preds = train_and_predict(df, model_name)
    
    # Save predictions
    output_file = Path(out_dir, "preds.parquet")
    preds.to_parquet(output_file, index=False)
    print(f"💾 Predictions saved to: {output_file}")
    print(f"📊 Saved {len(preds):,} race predictions")


if __name__ == "__main__":
    # Command line argument parser
    # Usage: python -m src.train_improved --in data/curated/f1_training.parquet --out data/curated/models --model rf
    
    ap = argparse.ArgumentParser(description="Train F1 race winner prediction model")
    ap.add_argument("--in", dest="inp", required=True, help="Path to training data")
    ap.add_argument("--out", required=True, help="Directory to save model predictions")
    ap.add_argument("--model", default="rf", choices=["rf", "gbm", "logreg"],
                   help="Model type: rf=Random Forest (default), gbm=Gradient Boosting, logreg=Logistic Regression")
    a = ap.parse_args()
    
    main(a.inp, a.out, a.model)

