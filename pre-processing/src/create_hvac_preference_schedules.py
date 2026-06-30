import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d

# ─── Config ──────────────────────────────────────────────────────────────────
HEAT_ENTER = 0.80
HEAT_EXIT  = 0.70
COOL_ENTER = 0.20
COOL_EXIT  = 0.30

COLS = [
    "timestamp",
    "out.indoor_radiant_temperature.conditioned_space..c",
    "out.outdoor_air_drybulb_temp..c",
]

current_file   = Path(__file__).resolve()
project_folder = current_file.parents[1]
input_folder   = project_folder / "out" / "consumption_files"
output_folder  = project_folder / "out" / "schedules_test"
output_folder.mkdir(parents=True, exist_ok=True)

NUM_PLOTS = 10


# ─── Core logic ──────────────────────────────────────────────────────────────
def compute_schedule(parquet_path):
    """Load a single parquet and return (timestamps, smoothed, schedule, y)."""
    df = pd.read_parquet(parquet_path, engine="fastparquet", columns=COLS)

    timestamps = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
    indoor     = df["out.indoor_radiant_temperature.conditioned_space..c"].values
    outdoor    = df["out.outdoor_air_drybulb_temp..c"].values

    y = (indoor > outdoor).astype(int)

    n             = len(y)
    steps_per_day = n / 365
    window        = int(steps_per_day * 28)

    smoothed = uniform_filter1d(y.astype(float), size=window)

    # Hysteresis state machine
    schedule = np.zeros(n, dtype=int)
    state    = 0
    if smoothed[0] >= HEAT_ENTER:
        state = 1
    elif smoothed[0] <= COOL_ENTER:
        state = -1

    for i in range(n):
        if state == 1:
            if smoothed[i] < HEAT_EXIT:
                state = 0
        elif state == -1:
            if smoothed[i] > COOL_EXIT:
                state = 0
        else:
            if smoothed[i] >= HEAT_ENTER:
                state = 1
            elif smoothed[i] <= COOL_ENTER:
                state = -1
        schedule[i] = state

    return timestamps, smoothed, schedule, y


# ─── Plotting ────────────────────────────────────────────────────────────────
def save_plot(building_id, timestamps, smoothed, schedule, y):
    """Generate and save the two-panel diagnostic plot."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- top: smoothed signal + thresholds ---
    axes[0].plot(timestamps, smoothed, color="steelblue", linewidth=1.5, label="Smoothed label")
    axes[0].axhline(HEAT_ENTER, color="salmon",  linestyle="--", linewidth=0.9, label=f"Heat enter ({HEAT_ENTER})")
    axes[0].axhline(HEAT_EXIT,  color="salmon",  linestyle=":",  linewidth=0.9, label=f"Heat exit  ({HEAT_EXIT})")
    axes[0].axhline(COOL_ENTER, color="skyblue", linestyle="--", linewidth=0.9, label=f"Cool enter ({COOL_ENTER})")
    axes[0].axhline(COOL_EXIT,  color="skyblue", linestyle=":",  linewidth=0.9, label=f"Cool exit  ({COOL_EXIT})")
    axes[0].set_ylabel("Smoothed P(heating)", fontsize=11)
    axes[0].set_title(f"Building {building_id} — Transition Detection", fontsize=13)
    axes[0].legend(loc="upper center", fontsize=8)
    axes[0].set_ylim(-0.05, 1.05)

    # --- bottom: raw labels + final schedule ---
    y_remapped = np.where(y == 1, 1, -1)
    heat_mask  = y_remapped ==  1
    cool_mask  = y_remapped == -1

    axes[1].scatter(timestamps[heat_mask], y_remapped[heat_mask],  color="salmon",  s=4, alpha=0.3, label="Raw — Heating (1)",  zorder=2)
    axes[1].scatter(timestamps[cool_mask], y_remapped[cool_mask],  color="skyblue", s=4, alpha=0.3, label="Raw — Cooling (-1)", zorder=2)
    axes[1].step(timestamps, schedule, color="crimson", linewidth=2.5, label="Schedule output", where="mid", zorder=3)

    axes[1].set_xlabel("Date", fontsize=12)
    axes[1].set_ylabel("Schedule label", fontsize=11)
    axes[1].set_title(f"Building {building_id} — HVAC Schedule Output", fontsize=13)
    axes[1].set_ylim(-1.3, 1.3)
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_yticklabels(["Cooling (-1)", "Shoulder (0)", "Heating (1)"])
    axes[1].legend(loc="lower center", fontsize=8)

    axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    fig.autofmt_xdate(rotation=45)

    plt.tight_layout()
    plt.savefig(output_folder / f"schedule_{building_id}.png", dpi=150)
    plt.close(fig)


# ─── Main loop ───────────────────────────────────────────────────────────────
parquet_files = sorted(input_folder.glob("*.parquet"))
print(f"Found {len(parquet_files)} parquet files in {input_folder}")

# Pick 10 random buildings to plot (by index into the sorted file list)
rng          = np.random.default_rng(seed=42)
plot_indices = set(rng.choice(len(parquet_files), size=min(NUM_PLOTS, len(parquet_files)), replace=False))

for idx, parquet_path in enumerate(parquet_files):
    building_id = parquet_path.stem.split("-")[0]
    print(f"[{idx+1}/{len(parquet_files)}] Processing building {building_id} ({parquet_path.name}) ...", end=" ")

    try:
        timestamps, smoothed, schedule, y = compute_schedule(parquet_path)
    except Exception as e:
        print(f"SKIPPED — {e}")
        continue

    # --- save parquet ---
    out_df = pd.DataFrame({
        "timestamp":   timestamps,
        "building_id": building_id,
        "schedule":    schedule,
    })
    out_df.to_parquet(output_folder / f"schedule_{building_id}.parquet", index=False)

    # --- plot only the randomly selected buildings ---
    if idx in plot_indices:
        save_plot(building_id, timestamps, smoothed, schedule, y)
        print(f"saved + plotted  (heating: {(schedule==1).sum()}, shoulder: {(schedule==0).sum()}, cooling: {(schedule==-1).sum()})")
    else:
        print(f"saved            (heating: {(schedule==1).sum()}, shoulder: {(schedule==0).sum()}, cooling: {(schedule==-1).sum()})")

print(f"\nDone. Parquets and plots saved to {output_folder}")