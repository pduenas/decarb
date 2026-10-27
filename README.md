# DECARB

DECARB (Distributed Energy Consumption in Responsive Buildings) is a Julia
optimization model for planning and operating building- and district-scale
energy systems. It couples electricity, space conditioning, hot water, fuel,
distributed generation, storage, and electric vehicles in a mixed-integer
linear program.

## Capabilities

- Combined heat and power and absorption chillers
- Heating, ventilation, air conditioning, and water heating
- Solar PV and wind generation
- Battery energy storage and electric vehicles
- Thermal and electrical demand and indoor-temperature dynamics
- Time-varying tariffs, peak charges, fuel costs, and emissions
- Existing-equipment dispatch and multi-period investment decisions

## Requirements

- Julia 1.10 or newer
- HiGHS, installed automatically as the default open-source solver
- Optional: Gurobi and a valid Gurobi license

## Install and run

Clone the repository, then instantiate its Julia environment:

```sh
julia --project=. -e "using Pkg; Pkg.instantiate()"
```

Run the bundled 24-hour example:

```sh
julia --project=. run.jl
```

Or call the package API:

```julia
using DECARB, HiGHS

result = DECARB.run_decarb!("examples/minimal";
    optimizer=HiGHS.Optimizer,
    attributes=Dict{String,Any}(),
    mip_gap=1e-4,
    time_limit=300.0,
    duals=true,
    return_result=true,
    relax_integrality=false,
)
```

Progress output is enabled by default; pass `verbose=false` to suppress it. The
function returns `nothing` by default, so a call entered directly at the Julia
prompt does not display a final result. Pass `return_result=true` to receive the
named tuple containing the solved JuMP model, loaded input dictionaries, and
`solve_info` summary. The case results are written to its `out/` directory.

### Other solvers

DECARB currently includes a configuration for Gurobi, detected automatically from
the optimizer, and can be configured for any other compatible JuMP MIP solver 
through the `optimizer` keyword. Solver packages are not DECARB dependencies.
Specific solvers must be installed and, where required, licensed separately.

For a fresh installation, add the chosen solver's Julia package first, then
load it before passing its optimizer to DECARB. For example:

```julia
using Pkg
Pkg.add("Gurobi")

using Gurobi, DECARB
DECARB.run_decarb!("examples/minimal"; optimizer=Gurobi.Optimizer)
```

Installation is only needed once per Julia environment; load the solver with
`using Gurobi` in each new Julia session.

## Case format

A case directory contains an `in/` directory with these CSV files:

| File | Purpose |
|---|---|
| `cfg.csv` | Global model, comfort, investment, and solver-independent settings |
| `tm.csv` | Time series for demand, weather, tariffs, and emissions |
| `bdg_i.csv`, `bdg_ii.csv` | Building geometry and thermal properties |
| `sp.csv` | Existing assets and candidate technology selections |
| `topo.csv` | Thermal connections between equipment and loads |
| `chp.csv`, `abp.csv`, `hvac.csv`, `wh.csv` | Thermal equipment catalogs |
| `pv.csv`, `wind.csv`, `bess.csv`, `ev.csv` | Distributed-energy catalogs |

See [`examples/minimal`](examples/minimal) for a complete dispatch case and
[`examples/investment`](examples/investment) for an annual case that starts
without installed equipment and requires investment.

DECARB writes:

- `out/eq1.csv`: cost and emissions summary
- `out/eq2.csv`: installed equipment and investment summary
- `out/ts.csv`: dispatch, temperature, fuel, state-of-charge, and dual time series
- `out/solve_info.csv`: `name,value` run metadata and MIP solve summary
- `out/balance.csv`: written only when the electrical balance tolerance is exceeded
- `out/iis_constraints.txt`: written when the solver can identify conflicting constraints

Known DECARB outputs from a previous run are deleted from `out/` at the start of each run; other files are kept.

### Batch execution

`bdg_path.txt` lists one case directory per line. As an example, the bundled file now runs:

```text
examples/minimal
examples/annual
examples/investment
```

From the repository root, run all listed cases with:

```sh
julia --project=. run_decarb.jl
```

The batch runner uses HiGHS by default. The `--solver` option currently accepts `highs` and `gurobi`; for other solvers, see [Other solvers](#other-solvers).

Run this command in PowerShell, Command Prompt, or a macOS/Linux terminal—not at the `julia>` prompt. If Julia is already running with the repository project active, use:

```julia
include("run_decarb.jl")
run_batch()
```

Each path may name either a case directory or its `in/` directory. Use forward slashes in `bdg_path.txt` so the same file works on Windows, macOS, and Linux. To run a different list with a selected command-line solver, pass `--solver` followed by the solver name:

```sh
julia --project=. run_decarb.jl --solver gurobi path/to/cases.txt
```

Omit `--solver` to use the default HiGHS solver.

Relative case paths are resolved from the directory containing the case-list file. The runner continues after a case fails and prints a summary table when all cases have been attempted. When invoked as a script, it exits with a nonzero status if any case failed. `run_batch()` returns `true` if one or more cases failed and `false` otherwise.

For another JuMP-compatible solver, install and load it in the active Julia environment, then run the batch from Julia:

```julia
using Gurobi
include("run_decarb.jl")
run_batch(["path/to/cases.txt"]; optimizer=Gurobi.Optimizer)
```

See [Other solvers](#other-solvers) for installation and licensing requirements.

## Tests

```sh
julia --project=. -e "using Pkg; Pkg.test()"
```

The test suite uses HiGHS and does not require proprietary software.

## Documentation

- [Model formulation](docs/formulation.md)
- [Inputs and outputs](docs/inputs_outputs.md)

## DECARB Team

The original version of DECARB was developed by [Pablo Duenas](https://github.com/pduenas)
at the [MIT Energy Initiative](https://energy.mit.edu/), with
contributions from Graham Turk, Karen Tapia-Ahumada, Leslie Norford,
Shaohui Liu, Onur Talu, and Sungho Shin, all affiliated with the [Massachusetts Institute of Technology](https://www.mit.edu/)
at the time of development.

DECARB is currently maintained by [Pablo Duenas](https://github.com/pduenas),
who is affiliated with the [MIT Energy Initiative](https://energy.mit.edu/) and
[Universidad Pontificia Comillas](https://www.comillas.edu/).


## DECARB Team

The original version of DECARB was developed at the [MIT Energy Initiative](https://energy.mit.edu/).
by Pablo Duenas. Other contributors, in order, are Graham
Turk, Karen Tapia-Ahumada, Leslie Norford, Shaohui Liu, Onur Talu, and Sungho
Shin. All contributors are from the MIT Energy Initiative.

## Contributing

Bug reports, documentation fixes, tests, and focused model improvements are
welcome. Before contributing:

- Search existing issues and pull requests, and open an issue before a large
  model, schema, or API change.
- Keep each pull request focused and add tests for behavioral changes.
- Update this README or the formulation documentation when assumptions,
  interfaces, inputs, or outputs change.
- Explain the physical units and source of new parameters. Optimization changes
  should include a small reproducible case and describe effects on feasibility,
  objective value, and relevant balances.
- Do not commit generated outputs, solver dumps, local manifests, confidential
  data, personal data, export-controlled data, or material you cannot license.
- Use four-space indentation, descriptive names, public-function docstrings,
  and platform-independent paths built with `joinpath`.

Run `Pkg.test()` before requesting review. By contributing, you agree that your
work is licensed under the GNU Affero General Public License v3.

## Citing DECARB

If you use DECARB in published work, cite the software release:

> P. Duenas, G. Turk, K. Tapia-Ahumada, L. Norford, S. Liu, O. Talu, and S.
> Shin. *DECARB: Distributed Energy Consumption in Responsive Buildings*.
> Version 1.0.0. <https://github.com/pduenas/decarb>

GitHub's **Cite this repository** feature provides additional formats from the
machine-readable [`CITATION.cff`](CITATION.cff) file.

## Security

Security fixes are applied to the current `main` branch and, when practical,
the latest published release. Do not open a public issue for a suspected
vulnerability. Use GitHub's private vulnerability reporting feature or email
`pduenas@mit.edu` with the subject `DECARB security report`.

Include the affected version or commit, reproduction steps, impact, and any
suggested mitigation. Allow time for acknowledgement and investigation before
public disclosure. See [`SECURITY.md`](SECURITY.md) for GitHub's discoverable
copy of this policy. Ordinary correctness bugs can use the public bug template.

## License

DECARB is licensed under the [GNU Affero General Public License v3](LICENSE).
Contributions are accepted under the same license.
