import argparse, time, random

from pathlib import Path

from typing import Dict, List

import requests, pandas as pd

from .utils import ensure_dir



ERGAST = "https://ergast.com/api/f1"

OPENF1 = "https://api.openf1.org"



def safe_request(url: str, params: Dict | None = None, retries: int = 3, pause: float = 0.1) -> Dict:

    for attempt in range(retries):

        try:

            r = requests.get(url, params=params or {}, timeout=30)

            r.raise_for_status()

            return r.json()

        except Exception as e:

            wait = pause * (2 ** attempt) + random.random()

            print(f"[retry] {url} ({attempt+1}/{retries}) after {wait:.2f}s → {e}")

            time.sleep(wait)

    raise RuntimeError(f"Failed after {retries} retries: {url}")



def collect_ergast_results(year: int) -> pd.DataFrame:

    url = f"{ERGAST}/{year}/results.json"

    payload = safe_request(url, params={"limit": 1000})

    races = payload["MRData"]["RaceTable"]["Races"]

    rows = []

    for race in races:

        rnd = int(race["round"])

        name = race["raceName"]

        for res in race["Results"]:

            rows.append({

                "season": year,

                "round": rnd,

                "race_name": name,

                "driver_id": res["Driver"]["driverId"],

                "driver_code": res["Driver"].get("code"),

                "constructor": res["Constructor"]["name"],

                "grid": int(res["grid"]),

                "finish_pos": int(res["position"]),

                "status": res["status"],

                "points": float(res["points"]),

                "wins_flag": 1 if int(res["position"]) == 1 else 0,

            })

    df = pd.DataFrame(rows)

    print(f"[year {year}] {len(df)} results")

    return df



def main(start: int, end: int, out: str):

    out_dir = ensure_dir(out)

    all_dfs: List[pd.DataFrame] = []

    for yr in range(start, end + 1):

        try:

            df_year = collect_ergast_results(yr)

            df_year.to_parquet(Path(out_dir, f"results_{yr}.parquet"), index=False)

            all_dfs.append(df_year)

        except Exception as e:

            print(f"[warn] skipping {yr}: {e}")



    if all_dfs:

        results = pd.concat(all_dfs, ignore_index=True)

        results.to_parquet(Path(out_dir, "results.parquet"), index=False)

        print(f"[done] combined {len(results)} rows from {len(all_dfs)} seasons")



if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--start", type=int, required=True)

    ap.add_argument("--end", type=int, required=True)

    ap.add_argument("--out", type=str, default="data/raw")

    a = ap.parse_args()

    main(a.start, a.end, a.out)

