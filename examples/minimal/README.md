# minimal — 24-hour operational example

Single-family building, Boston (42.36 N, 71.06 W), 1 January, hourly.

| Asset | Configuration |
|---|---|
| Heat pump | 1 × 8 kW heating / 6 kW cooling, COP 3.0/3.5, existing |
| PV | 4 × 400 W monocrystalline, 30° tilt, due south, existing |
| Water heater | 1 × 12 kW tankless, natural gas, existing |
| Grid | import ≤ 100 kW, export ≤ 100 kW, TOU tariff, peak 17:00–20:00 |

`pIT = 0` disables investment: this is a pure dispatch problem. CHP, absorption
chiller, wind, BESS and EV catalogues are populated but deselected in `sp.csv`,
which keeps the "inactive technology" code paths under test.

## Run

```julia
using DECARB
DECARB.run_decarb!("examples/minimal", 1e-2, 300.0, 2, false)  # 2 = HiGHS