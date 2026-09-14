# annual — full-year operational example

Residential building in Houston, Texas (29.99 N, 95.36 W), modeled hourly for
2018. The case includes an existing air-source heat pump, electric water
heater, 30 PV modules, and one battery module. Equipment investment is disabled
for all technologies, making this an annual dispatch case.

## Run

From the repository root:

```sh
julia --project=. run.jl examples/annual
```

Outputs are written to `examples/annual/out/`.
