import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from .utils import ensure_dir

FEATURES = ["grid","constructor_cat","driver_cat","form_5","exp_starts","tenure","dnf","points",
            "driver_points_at_stage_of_season","constructorId_points_at_stage_of_season",
            "race_number_in_season","grid_delta_from_pole","constructor_wins_last_10"]
TARGET = "y_win"

def build_model(kind: str):
    if kind == "logreg":
        return LogisticRegression(max_iter=200, class_weight="balanced", n_jobs=None)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=400, max_depth=None, random_state=42, n_jobs=-1)
    # Default: GBM
    return GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3)

def group_cv_auc(model, X, y, groups):
    gkf = GroupKFold(n_splits=5)
    scores = []
    for tr, va in gkf.split(X, y, groups=groups):
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict_proba(X.iloc[va])[:,1]
        scores.append(roc_auc_score(y.iloc[va], p))
    return float(np.mean(scores))

def train_and_predict(df: pd.DataFrame, model_name: str) -> pd.DataFrame:
    X = df[FEATURES].copy()
    # Fill NaN values
    X = X.fillna(0)
    y = df[TARGET].astype(int).copy()
    groups = df["season"]
    model = build_model(model_name)
    try:
        cv = group_cv_auc(build_model(model_name), X, y, groups)
        print(f"[cv] {model_name} mean AUC: {cv:.3f}")
    except Exception as e:
        print(f"[cv] skipped: {e}")

    model.fit(X, y)
    out = df.copy()
    out["p_win"] = model.predict_proba(X)[:,1]
    out["pred_rank"] = out.groupby(["season","round"])["p_win"].rank(method="first", ascending=False)
    out["predicted_winner"] = (out["pred_rank"] == 1).astype(int)
    return out

def main(inp: str, out: str, model_name: str):
    out_dir = ensure_dir(out)
    df = pd.read_parquet(inp)
    preds = train_and_predict(df, model_name)
    preds.to_parquet(Path(out_dir, "preds.parquet"), index=False)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="gbm", choices=["logreg","rf","gbm"])
    a = ap.parse_args()
    main(a.inp, a.out, a.model)

