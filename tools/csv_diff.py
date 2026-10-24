#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="find and save the difference between two csv files with the same structure."
    )
    parser.add_argument(
        "file1",
        type=Path,
        help="path to the first (base or old) csv file",
    )
    parser.add_argument(
        "file2",
        type=Path,
        help="path to the second (new) csv file",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="path to the output csv file for the result",
    )
    parser.add_argument(
        "--mode",
        choices=["b_minus_a", "symmetric", "a_minus_b"],
        default="b_minus_a",
        help=(
            "difference mode: "
            "'b_minus_a' means rows from file2 that are not in file1 (default); "
            "'symmetric' means rows that are only in one of the two files; "
            "'a_minus_b' means rows from file1 that were removed in file2."
        ),
    )
    return parser.parse_args()


def export_csv_difference(
    path1: Path, path2: Path, out_path: Path, mode: str = "b_minus_a"
) -> None:
    for path in (path1, path2):
        if not path.is_file():
            raise FileNotFoundError(f"file not found at the given path: {path.resolve()}")

    df1 = pd.read_csv(path1)
    df2 = pd.read_csv(path2)

    # check that column names and their order match
    if list(df1.columns) != list(df2.columns):
        raise ValueError(
            "column structure of the files is not identical\n"
            f"  {path1.name}: {list(df1.columns)}\n"
            f"  {path2.name}: {list(df2.columns)}"
        )

    cols = list(df1.columns)

    if mode == "b_minus_a":
        # rows of the second file that are missing in the first
        merged = df2.merge(df1.drop_duplicates(), on=cols, how="left", indicator=True)
        diff_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    elif mode == "a_minus_b":
        # rows of the first file that are missing in the second
        merged = df1.merge(df2.drop_duplicates(), on=cols, how="left", indicator=True)
        diff_df = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

    elif mode == "symmetric":
        # rows that appear in exactly one of the files
        diff_df = pd.concat([df1, df2]).drop_duplicates(keep=False)

    # create parent folders if they do not exist yet
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # save with the same schema and no index
    diff_df.to_csv(out_path, index=False)
    print(f"saved {len(diff_df)} rows to: {out_path.resolve()}")


def main() -> None:
    args = parse_args()
    try:
        export_csv_difference(args.file1, args.file2, args.output, args.mode)
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()