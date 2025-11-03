import argparse

from pathlib import Path

import pandas as pd

import numpy as np



def main(pred_path: str, odds_path: str, out_dir: str, stake: float = 1.0):

    preds = pd.read_parquet(pred_path)

    

    # Try multiple odds file paths

    odds_file = None

    if odds_path and Path(odds_path).exists():

        odds_file = odds_path

    else:

        # Try fallback paths

        for fallback in ["data/curated/odds.csv", "data/curated/f1_odds.csv"]:

            if Path(fallback).exists():

                odds_file = fallback

                print(f"[bet] Using odds file: {odds_file}")

                break

    

    if odds_file:

        odds = pd.read_csv(odds_file)

        df = preds.merge(odds, on=["season","round","driver_id"], how="left")

    else:

        print("[bet] Warning: No odds file found, using default odds of 2.0")

        df = preds.copy()

        df["decimal_odds"] = 2.0

    

    picks = df.loc[df["predicted_winner"] == 1].copy()

    

    # Profit = (odds - 1)*stake on win, else -stake

    # Fill missing odds with default 2.0

    picks["decimal_odds"] = picks["decimal_odds"].fillna(2.0)

    

    picks["payout"] = np.where(picks["wins_flag"]==1,

                               (picks["decimal_odds"] - 1.0) * stake,

                               -stake)

    

    picks["cum_pnl"] = picks["payout"].cumsum()

    

    # Calculate ROI = cumulative PnL / total amount staked (number of bets * stake)

    picks["number_of_bets"] = range(1, len(picks) + 1)

    picks["ROI"] = picks["cum_pnl"] / (picks["number_of_bets"] * stake)

    

    out = Path(out_dir)

    out.mkdir(parents=True, exist_ok=True)

    

    # Save both parquet and CSV

    picks.to_parquet(out / "betting_with_odds.parquet", index=False)

    picks.to_csv(out / "betting_with_odds.csv", index=False)

    

    print("[bet] Saved -> betting_with_odds.parquet")

    print("[bet] Saved -> betting_with_odds.csv")

    

    # Print summary statistics

    final_pnl = picks["cum_pnl"].iloc[-1] if len(picks) > 0 else 0

    final_roi = picks["ROI"].iloc[-1] if len(picks) > 0 else 0

    num_bets = len(picks)

    wins = (picks["wins_flag"] == 1).sum()

    

    print(f"[bet] Summary: {num_bets} bets, {wins} wins, Final P&L: {final_pnl:.2f}, ROI: {final_roi:.2%}")



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--pred", required=True)

    ap.add_argument("--odds", required=True)

    ap.add_argument("--out", required=True)

    a = ap.parse_args()

    main(a.pred, a.odds, a.out)

