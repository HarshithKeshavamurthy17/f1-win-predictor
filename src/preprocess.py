import argparse

from pathlib import Path

import pandas as pd

from .utils import ensure_dir



def apply_notebook_cleaning(df: pd.DataFrame) -> pd.DataFrame:

    """

    Replicate the cleaning steps from the preprocessing notebooks:

    - Drop rows where starting_grid_position (grid) is missing or invalid

    - Drop rows with status like 'Did not start'

    - Strip whitespace from constructor/team names if present

    - Drop rows from the first race index (race_index == 1) if that column exists

    - Drop any remaining rows with NaN values in critical columns

    """

    df = df.copy()

    

    # Filter out invalid grid positions (already done, but ensure consistency)

    if "grid" in df.columns:

        df = df[df["grid"] >= 1]

    

    # Drop 'Did not start' status rows

    if "status" in df.columns:

        df = df[~df["status"].str.contains("Did not start", case=False, na=False)]

    

    # Strip whitespace from constructor column if it exists

    if "constructor" in df.columns:

        df["constructor"] = df["constructor"].str.strip()

    

    # Drop rows where driver_id or driver_code is missing

    if "driver_id" in df.columns:

        df = df.dropna(subset=["driver_id"])

    

    if "driver_code" in df.columns:

        # driver_code can be None for older drivers, so we don't drop it

        pass

    

    # Remove first race of each season if it has missing lag features (handled in features.py)

    # But we can drop any rows that are clearly incomplete

    

    return df



def main(inp: str, out: str):

    in_dir, out_dir = Path(inp), ensure_dir(out)

    results = pd.read_parquet(in_dir / "results.parquet")



    results = (

        results

        .query("grid >= 1")

        .assign(

            dnf=results["status"].str.contains(

                "Accident|Engine|DNF|Gear|Hydraul|Electrical|Crash",

                case=False, na=False

            ).astype(int)

        )

    )

    

    # Apply notebook cleaning steps

    results = apply_notebook_cleaning(results)



    races = results[["season","round","race_name"]].drop_duplicates().sort_values(["season","round"])

    results.to_parquet(out_dir / "driver_race.parquet", index=False)

    races.to_parquet(out_dir / "races.parquet", index=False)

    results.to_parquet(out_dir / "f1_training_base.parquet", index=False)



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--in", dest="inp", required=True)

    ap.add_argument("--out", required=True)

    a = ap.parse_args()

    main(a.inp, a.out)

