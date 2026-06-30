import polars as pl
from pathlib import Path
import fnmatch
import re

def find_empty_climate_matches(base_out_dir: Path):
    """
    Scan all zone feeder_summaries and report which have empty or missing climate matches.
    Saves a CSV file listing all feeders with missing matches.
    """
    if not base_out_dir.exists():
        print(f"❌ Output directory not found: {base_out_dir}")
        return

    # ✅ Correct regex — matches folders like P1R, P23U, etc.
    zones = [z for z in base_out_dir.iterdir() if z.is_dir() and re.match(r"P\d+[RU]$", z.name)]
    if not zones:
        print(f"⚠️ No zone directories found under {base_out_dir}")
        return

    results = []  # collect rows as dicts for efficiency
    print(f"🔍 Scanning {len(zones)} zones in {base_out_dir}...\n")

    for zone in zones:
        feeder_summary_dir = zone / "feeder_summaries"
        if not feeder_summary_dir.exists():
            print(f"⏭️ Zone {zone.name}: no feeder_summaries folder found.")
            continue

        feeder_files = [
            f for f in feeder_summary_dir.iterdir()
            if f.is_file() and fnmatch.fnmatch(f.name, "feeder_summary_*.csv")
        ]
        if not feeder_files:
            print(f"⏭️ Zone {zone.name}: no feeder summary CSVs found.")
            continue

        for feeder_file in feeder_files:
            try:
                df = pl.read_csv(feeder_file)
            except Exception as e:
                print(f"⚠️ Error reading {feeder_file.name}: {e}")
                continue

            if "nrel_climate_match" not in df.columns:
                print(f"⚠️ {feeder_file.name}: missing column 'nrel_climate_match'")
                continue

            # Check for null or empty string entries
            mask = pl.col("nrel_climate_match").is_null() | (pl.col("nrel_climate_match").cast(pl.Utf8) == "")
            empty_count = df.filter(mask).height

            if empty_count > 0:
                print(f"🚨 {zone.name}/{feeder_file.name}: {empty_count} empty climate matches")
                results.append({
                    "zone": zone.name,
                    "feeder_file": feeder_file.name,
                    "empty_count": empty_count
                })
            else:
                print(f"✅ {zone.name}/{feeder_file.name}: all climate matches filled")

    # Save results if any were found
    if results:
        df_out = pl.DataFrame(results)
        out_path = base_out_dir / "missing_matches.csv"
        df_out.write_csv(out_path)
        print(f"\n📄 Saved summary to: {out_path}")
    else:
        print("\n✅ No missing matches found anywhere!")

    print("🎉 Scan complete.")


if __name__ == "__main__":
    current_path = Path(__file__).resolve()
    parent_dir = current_path.parents[1]
    out_dir = parent_dir / "out"

    find_empty_climate_matches(out_dir)
