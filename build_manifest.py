from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SOURCE_SUFFIXES = {
    "CYGNSS": "CYGNSS.tif",
    "Tianmu-1": "Tianmu-1.tif",
    "FY-3E": "FY3E.tif",
    "FY-3F": "FY3F.tif",
    "FY-3G": "FY3G.tif",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the daily raster manifest.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--start", required=True, help="YYYYMMDD")
    parser.add_argument("--end", required=True, help="YYYYMMDD")
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def locate(root: Path, filename: str) -> str:
    matches = list(root.rglob(filename))
    if len(matches) > 1:
        raise RuntimeError(f"Multiple rasters match {filename}: {matches[:3]}")
    return str(matches[0].resolve()) if matches else ""


def main() -> None:
    args = parse_args()
    root = Path(args.input_dir)
    records = []
    for date in pd.date_range(args.start, args.end).strftime("%Y%m%d"):
        record = {"date": date}
        for source, suffix in SOURCE_SUFFIXES.items():
            record[source] = locate(root, f"{date}_{suffix}")
        records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"Saved {len(records)} dates to {output}")


if __name__ == "__main__":
    main()
