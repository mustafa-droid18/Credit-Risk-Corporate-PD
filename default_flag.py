#!/usr/bin/env python3

import pandas as pd
import os


def main():
    # Input/output paths
    input_path = "train-2025.csv"
    output_path = "Default_added.csv"

    # --- Ensure output directory exists ---
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # Read CSV
    df = pd.read_csv(input_path, low_memory=False)

    # Parse dates
    df["stmt_date"] = pd.to_datetime(df["stmt_date"])
    df["def_date"] = pd.to_datetime(df["def_date"])

    # Params
    LAG_MONTHS = 4
    HORIZON_MONTHS = 12
    START_INCL, END_INCL = True, True

    # 1) Availability date
    df["avail_date"] = df["stmt_date"] + pd.DateOffset(months=LAG_MONTHS)

    # 2) Prediction window
    df["pred_start"] = df["avail_date"]
    df["pred_end"] = df["avail_date"] + pd.DateOffset(months=HORIZON_MONTHS)

    # 3) Find next default
    def get_next_default(group: pd.DataFrame) -> pd.Series:
        result = pd.Series(pd.NaT, index=group.index)
        for idx, row in group.iterrows():
            future_defaults = group.loc[
                (group["def_date"].notna()) &
                (group["def_date"] > row["stmt_date"]),
                "def_date"
            ]
            if len(future_defaults) > 0:
                result[idx] = future_defaults.min()
        return result

    df["next_def"] = df.groupby("id", group_keys=False).apply(get_next_default)

    # 4) Build 12-month label
    y = pd.Series(0, index=df.index, dtype="Int64")
    has_next_def = df["next_def"].notna()

    # 4a) Default occurred before info was available → NA
    default_before_info = has_next_def & (df["next_def"] < df["avail_date"])
    y[default_before_info] = pd.NA

    # 4b) Default within prediction window → 1
    if START_INCL and END_INCL:
        in_win = (
            has_next_def &
            (df["next_def"] >= df["pred_start"]) &
            (df["next_def"] <= df["pred_end"])
        )
    elif START_INCL and not END_INCL:
        in_win = (
            has_next_def &
            (df["next_def"] >= df["pred_start"]) &
            (df["next_def"] < df["pred_end"])
        )
    elif not START_INCL and END_INCL:
        in_win = (
            has_next_def &
            (df["next_def"] > df["pred_start"]) &
            (df["next_def"] <= df["pred_end"])
        )
    else:
        in_win = (
            has_next_def &
            (df["next_def"] > df["pred_start"]) &
            (df["next_def"] < df["pred_end"])
        )

    y[in_win] = 1
    df["default_12m"] = y

    print(df["default_12m"].value_counts(dropna=False))

    # Write output
    df.to_csv(output_path, index=False)
    print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()