import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage import uniform_filter1d

# ─── 1. Load data ────────────────────────────────────────────────────────────
current_file    = Path(__file__).resolve()
current_folder  = current_file.parent
project_folder  = current_file.parents[1]
parquet_path    = project_folder / "out" / "consumption_files" / "77-0.parquet"
building_id     = parquet_path.stem.split("-")[0]   # "83-0" → "83"
output_folder   = project_folder / "out" / "setpoint_timeseries"
output_folder.mkdir(parents=True, exist_ok=True)

cols = [
    "timestamp",
    "out.indoor_radiant_temperature.conditioned_space..c",
    "out.outdoor_air_drybulb_temp..c",
]

df = pd.read_parquet(parquet_path, engine="fastparquet", columns=cols)

timestamps = pd.to_datetime(df["timestamp"], format="%m/%d/%Y %H:%M")
indoor     = df["out.indoor_radiant_temperature.conditioned_space..c"].values
outdoor    = df["out.outdoor_air_drybulb_temp..c"].values

# ─── 2. Build labels ─────────────────────────────────────────────────────────
y = (indoor > outdoor).astype(int)

print(f"[debug] indoor range:  {indoor.min():.2f} – {indoor.max():.2f}")
print(f"[debug] outdoor range: {outdoor.min():.2f} – {outdoor.max():.2f}")
print(f"[debug] label counts → 0: {(y==0).sum()}, 1: {(y==1).sum()}")

# ─── 3. Infer resolution and set window ──────────────────────────────────────
n             = len(y)
steps_per_day = n / 365
window         = int(steps_per_day * 14)   # 2-week moving average

print(f"[debug] n={n}, steps_per_day={steps_per_day:.1f}, window={window}")

# ─── 4. Find transitions via smoothed label signal ──────────────────────────
smoothed = uniform_filter1d(y.astype(float), size=window)

# Three-class schedule with hysteresis to prevent flickering near boundaries.
# Each state has an entry threshold (must cross to enter) and an exit threshold
# (must cross to leave). The gap between them is what kills the oscillation.
#
#   To enter heating:  smoothed must rise above HEAT_ENTER (0.85)
#   To leave heating:  smoothed must fall below HEAT_EXIT  (0.70)
#   To enter cooling:  smoothed must fall below COOL_ENTER (0.15)
#   To leave cooling:  smoothed must rise above COOL_EXIT  (0.30)
#   Everything in between is shoulder.
HEAT_ENTER = 0.85
HEAT_EXIT  = 0.70
COOL_ENTER = 0.15
COOL_EXIT  = 0.30

schedule = np.zeros(n, dtype=int)   # start as shoulder
state    = 0                        # current state: 1, 0, or -1

# Seed the initial state from the first smoothed value so we don't
# always start as shoulder if the data begins mid-winter.
if smoothed[0] >= HEAT_ENTER:
    state = 1
elif smoothed[0] <= COOL_ENTER:
    state = -1

for i in range(n):
    if state == 1:          # currently heating
        if smoothed[i] < HEAT_EXIT:
            state = 0       # drop to shoulder
    elif state == -1:       # currently cooling
        if smoothed[i] > COOL_EXIT:
            state = 0       # rise to shoulder
    else:                   # currently shoulder
        if smoothed[i] >= HEAT_ENTER:
            state = 1       # commit to heating
        elif smoothed[i] <= COOL_ENTER:
            state = -1      # commit to cooling
    schedule[i] = state

print(f"[debug] schedule counts → heating(1): {(schedule==1).sum()}, "
      f"shoulder(0): {(schedule==0).sum()}, cooling(-1): {(schedule==-1).sum()}")

# ─── 5. Plot ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# --- top: smoothed signal + thresholds ---
axes[0].plot(timestamps, smoothed, color="steelblue", linewidth=1.5, label="Smoothed label")
axes[0].axhline(HEAT_ENTER, color="salmon",  linestyle="--", linewidth=0.9, label=f"Heat enter ({HEAT_ENTER})")
axes[0].axhline(HEAT_EXIT,  color="salmon",  linestyle=":",  linewidth=0.9, label=f"Heat exit  ({HEAT_EXIT})")
axes[0].axhline(COOL_ENTER, color="skyblue", linestyle="--", linewidth=0.9, label=f"Cool enter ({COOL_ENTER})")
axes[0].axhline(COOL_EXIT,  color="skyblue", linestyle=":",  linewidth=0.9, label=f"Cool exit  ({COOL_EXIT})")
axes[0].set_ylabel("Smoothed P(heating)", fontsize=11)
axes[0].set_title("Transition Detection via Smoothed Labels", fontsize=13)
axes[0].legend(loc="upper center", fontsize=8)
axes[0].set_ylim(-0.05, 1.05)

# --- bottom: raw labels (remapped) + final schedule ---
y_remapped = np.where(y == 1, 1, -1)

heat_mask = y_remapped ==  1
cool_mask = y_remapped == -1

axes[1].scatter(timestamps[heat_mask], y_remapped[heat_mask],  color="salmon",  s=4, alpha=0.3, label="Raw — Heating (1)",  zorder=2)
axes[1].scatter(timestamps[cool_mask], y_remapped[cool_mask],  color="skyblue", s=4, alpha=0.3, label="Raw — Cooling (-1)", zorder=2)
axes[1].step(timestamps, schedule, color="crimson", linewidth=2.5, label="Schedule output", where="mid", zorder=3)

axes[1].set_xlabel("Date", fontsize=12)
axes[1].set_ylabel("Schedule label", fontsize=11)
axes[1].set_title("HVAC Schedule Output", fontsize=13)
axes[1].set_ylim(-1.3, 1.3)
axes[1].set_yticks([-1, 0, 1])
axes[1].set_yticklabels(["Cooling (-1)", "Shoulder (0)", "Heating (1)"])
axes[1].legend(loc="lower center", fontsize=8)

# monthly ticks on the shared x-axis
axes[1].xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1))
axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
fig.autofmt_xdate(rotation=45)

plt.tight_layout()
plt.savefig(output_folder / f"schedule_{building_id}.png", dpi=150)
plt.show()

# ─── 6. Save CSV ─────────────────────────────────────────────────────────────
out_df = pd.DataFrame({
    "timestamp":  timestamps,
    "building_id": building_id,
    "schedule":   schedule,
})
csv_path = output_folder / f"schedule_{building_id}.csv"
out_df.to_csv(csv_path, index=False)
print(f"[output] saved {csv_path}")

# ─── 7. Summary ──────────────────────────────────────────────────────────────
print(f"Total timesteps       : {n}")
print(f"Steps per day         : {steps_per_day:.1f}")
print(f"Smoothing window      : {window} timesteps (~14 days)")
print(f"Thresholds            : heat enter/exit {HEAT_ENTER}/{HEAT_EXIT}, cool enter/exit {COOL_ENTER}/{COOL_EXIT}")
print(f"Raw heating labels    : {(y==1).sum()}  ({(y==1).mean()*100:.1f}%)")
print(f"Raw cooling labels    : {(y==0).sum()}  ({(y==0).mean()*100:.1f}%)")
print(f"Schedule — heating(1) : {(schedule==1).sum()}  ({(schedule==1).mean()*100:.1f}%)")
print(f"Schedule — shoulder(0): {(schedule==0).sum()}  ({(schedule==0).mean()*100:.1f}%)")
print(f"Schedule — cooling(-1): {(schedule==-1).sum()}  ({(schedule==-1).mean()*100:.1f}%)")