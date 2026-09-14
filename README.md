# DECARB

DECARB (Distributed Energy Consumption in Responsive Buildings) is a Julia
optimization model for planning and operating building- and district-scale
energy systems. It couples electricity, space conditioning, hot water, fuel,
distributed generation, storage, and electric vehicles in a mixed-integer
linear program built with [JuMP](https://jump.dev/).

> DECARB is research software. Review assumptions and validate input data before
> using results for engineering, policy, or investment decisions.

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
using DECARB

result = DECARB.run_decarb!("examples/minimal";
    mip_gap=1e-2,
    time_limit=300.0,
    solver=:highs,
    relax_integrality=false,
)
```

The returned named tuple contains the solved JuMP model and the loaded input
dictionaries. The case results are written to its `out/` directory.

### Optional Gurobi support

Gurobi is not required to install or use DECARB. Add and load it explicitly
before selecting it:

```julia
using Pkg
Pkg.add("Gurobi")

using Gurobi, DECARB
DECARB.run_decarb!("examples/minimal"; solver=:gurobi)
```

Gurobi is proprietary software and has separate installation and licensing
terms.

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
- `out/balance.csv`: written only when the electrical balance tolerance is exceeded
- `out/iis_constraints.txt`: written when the solver can identify conflicting constraints

For batch execution, list one case directory per line in `bdg_path.txt` and run
`julia --project=. run_decarb.jl`. A different list file may be supplied as the
first argument.

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
work is licensed under the GNU Affero General Public License v3. Participation
is also governed by the [code of conduct](CODE_OF_CONDUCT.md).

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
