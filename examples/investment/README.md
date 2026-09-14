# investment — annual investment-planning example

This case is derived from `examples/annual` and contains no existing equipment.
One investment window is enabled (`pIT = 1`). The air-source heat pump, water
heater, PV, and battery are available as candidate investments.

Unserved electricity, indoor-temperature deviations, and unserved hot water
are disabled by setting their penalty parameters to zero. Because the annual
profiles include space-conditioning and hot-water requirements, the case is
feasible only if the model invests in equipment. PV and battery investment
remain economic choices.

## Run

From the repository root:

```sh
julia --project=. run.jl examples/investment
```

Outputs are written to `examples/investment/out/`.
