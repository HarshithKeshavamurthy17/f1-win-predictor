import argparse

from pathlib import Path

import shap

import pandas as pd

from xgboost import XGBClassifier



FEATURES = ["grid","constructor_cat","driver_cat","form_5","exp_starts","tenure","dnf","points",

            "driver_points_at_stage_of_season","constructorId_points_at_stage_of_season",

            "race_number_in_season","grid_delta_from_pole","constructor_wins_last_10"]



def main(pred_path: str, out_dir: str):

    preds = pd.read_parquet(pred_path)

    X = preds[FEATURES]

    y = preds["y_win"].astype(int)

    model = XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05,

                          subsample=0.9, colsample_bytree=0.9, reg_lambda=1.5,

                          eval_metric="logloss", n_jobs=-1)

    model.fit(X, y)

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    imp = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": abs(shap_values).mean(axis=0)})

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    imp.sort_values("mean_abs_shap", ascending=False).to_csv(Path(out_dir, "shap_importance.csv"), index=False)

    print("[interpret] Wrote shap_importance.csv")



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred", dest="pred_path", required=True)

    ap.add_argument("--out", required=True)

    a = ap.parse_args()

    main(a.pred_path, a.out)

