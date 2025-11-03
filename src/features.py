import argparse

from pathlib import Path

import pandas as pd

from .utils import ensure_dir



def add_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["driver_start"] = 1

    df["exp_starts"] = (

        df.sort_values(["driver_id","season","round"])

          .groupby("driver_id")["driver_start"].cumsum() - 1

    )

    df_sorted = df.sort_values(["driver_id","season","round"])

    df["team_change"] = (

        df_sorted.groupby("driver_id")["constructor"]

          .transform(lambda s: (s != s.shift(1)).astype(int))

    )

    df["tenure"] = (

        df.sort_values(["driver_id","season","round"])

          .groupby(["driver_id","constructor"])["driver_start"].cumsum()

    )

    df["points_lag"] = (

        df.sort_values(["driver_id","season","round"])

          .groupby("driver_id")["points"].shift(1)

    )

    df["form_5"] = (

        df.sort_values(["driver_id","season","round"])

          .groupby("driver_id")["points_lag"]

          .rolling(5, min_periods=1).sum().reset_index(level=0, drop=True)

    )

    

    # Cumulative season points up to the race (excluding current race points)

    df["driver_points_at_stage_of_season"] = (

        df.sort_values(["driver_id","season","round"])

          .groupby(["driver_id","season"])["points"]

          .cumsum() - df["points"]

    )

    

    # Constructor cumulative points at stage of season

    # First sum points per constructor per race, then cumsum within season

    constructor_race_points = (

        df.sort_values(["season","round","constructor"])

          .groupby(["season","round","constructor"])["points"].sum()

          .groupby(level=[0, 2]).cumsum()

          .reset_index(name="constructorId_points_at_stage_of_season")

    )

    df = df.merge(constructor_race_points, on=["season","round","constructor"], how="left")

    

    # Race number in current season

    df["race_number_in_season"] = (

        df.sort_values(["season","round"])

          .groupby("season")["round"].rank(method="dense").astype(int)

    )

    

    # Qualifying/grid position delta from best in race

    df["grid_delta_from_pole"] = (

        df.groupby(["season","round"])["grid"].transform(lambda x: x - x.min())

    )

    

    # Constructor dominance indicator: how many wins constructor has in last N races

    df["constructor_wins_last_10"] = (

        df.sort_values(["constructor","season","round"])

          .groupby("constructor")["wins_flag"]

          .rolling(10, min_periods=1).sum()

          .reset_index(level=0, drop=True)

          .shift(1)  # lag by 1 to avoid leakage

    )

    

    df["constructor_cat"] = df["constructor"].astype("category").cat.codes

    df["driver_cat"] = df["driver_id"].astype("category").cat.codes

    df["y_win"] = df["wins_flag"].astype(int)

    return df



def main(inp: str, out: str):

    in_path = Path(inp) / "f1_training_base.parquet"

    out_dir = ensure_dir(out)

    df = pd.read_parquet(in_path)

    feat = add_features(df)

    keep = ["season","round","race_name","driver_id","driver_cat","constructor_cat",

            "grid","points","form_5","exp_starts","tenure","dnf","y_win",

            "driver_points_at_stage_of_season","constructorId_points_at_stage_of_season",

            "race_number_in_season","grid_delta_from_pole","constructor_wins_last_10"]

    feat[keep].to_parquet(out_dir / "f1_training.parquet", index=False)



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="inp", required=True)

    ap.add_argument("--out", required=True)

    a = ap.parse_args()

    main(a.inp, a.out)

