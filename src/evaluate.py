import argparse

from pathlib import Path

import pandas as pd

import numpy as np



def top1_accuracy(df: pd.DataFrame) -> float:

    # Handle both wins_flag and y_win column names

    win_col = "wins_flag" if "wins_flag" in df.columns else "y_win"

    picks = df.loc[df["predicted_winner"] == 1, [win_col]]

    return float(picks[win_col].mean())



def season_summary(df: pd.DataFrame) -> pd.DataFrame:

    # Handle both wins_flag and y_win column names

    win_col = "wins_flag" if "wins_flag" in df.columns else "y_win"

    picks = df.loc[df["predicted_winner"] == 1, ["season", win_col]]

    return (picks.groupby("season")[win_col].mean()

                 .reset_index(name="top1_accuracy"))



def simple_bet_backtest(df: pd.DataFrame, stake: float = 1.0) -> pd.DataFrame:

    # Handle both wins_flag and y_win column names

    win_col = "wins_flag" if "wins_flag" in df.columns else "y_win"

    picks = df.loc[df["predicted_winner"] == 1].copy()

    picks["payout"] = np.where(picks[win_col]==1, stake, -stake)

    picks["cum_pnl"] = picks["payout"].cumsum()

    return picks[["season","round","race_name","payout","cum_pnl"]]



def main(pred_path: str, races_path: str):

    preds = pd.read_parquet(pred_path)

    acc = top1_accuracy(preds)

    by_season = season_summary(preds)

    print(f"[eval] Top-1 race pick accuracy (overall): {acc:.3f}")

    print(by_season.to_string(index=False))

    pnl = simple_bet_backtest(preds, stake=1.0)

    out = Path(pred_path).parent / "betting_trace.parquet"

    pnl.to_parquet(out, index=False)

    print(f"[eval] Betting trace saved -> {out}")



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred", required=True)

    ap.add_argument("--races", required=True)

    a = ap.parse_args()

    main(a.pred, a.races)

