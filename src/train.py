import argparse

from pathlib import Path

import pandas as pd

import numpy as np

from sklearn.metrics import roc_auc_score

from sklearn.model_selection import GroupKFold

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from xgboost import XGBClassifier

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

    if kind == "gbm":

        return GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3)

    

    if kind == "tuned_xgb":

        # Best params from notebook grid search (v2): max_depth, learning_rate, subsample tuned

        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,

                             min_child_weight=2, subsample=0.9, colsample_bytree=0.9,

                             eval_metric="logloss", random_state=42, n_jobs=-1)

    

    # Default XGB with notebook-aligned params

    return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,

                         subsample=0.9, colsample_bytree=0.9,

                         eval_metric="logloss", random_state=42, n_jobs=-1)



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

    ap.add_argument("--model", default="xgb", choices=["xgb","logreg","rf","gbm","tuned_xgb"])

    a = ap.parse_args()

    main(a.inp, a.out, a.model)

