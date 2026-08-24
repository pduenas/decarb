# DECARB

DECARB is a Julia-based optimization model for distributed energy systems, building-level thermal and electric coupling, and operational scheduling for decarbonization-oriented energy planning.

## Overview

This project builds and solves a mixed-integer optimization model to evaluate energy system operation and investment decisions across building and district-scale assets, including:

- CHP units
- Heat pumps / HVAC systems
- Absorption chillers
- Water heaters
- PV and wind generation
- Battery storage
- Electric vehicles
- Thermal and electrical demand modeling

## Project structure

```text
.
├── LICENSE
├── Manifest.toml
├── Project.toml
├── run.jl
├── run_decarb.jl
├── in/
│   ├── abs.csv
│   ├── bdg_i.csv
│   ├── bdg_ii.csv
│   ├── bess.csv
│   ├── chp.csv
│   ├── ev.csv
│   ├── hvac.csv
│   ├── in.csv
│   ├── pv.csv
│   ├── sp.csv
│   ├── tm.csv
│   ├── topo.csv
│   ├── wh.csv
│   ├── wind.csv
│   └── ...
├── out/
│   └── generated results and reports
├── src/
│   ├── DECARB.jl
│   ├── case_runner/
│   ├── configure_solver/
│   ├── decarb_model/
│   ├── load_inputs/
│   ├── read_outputs/
│   ├── solve_model/
│   └── write_outputs/
├── docs/
│   └── formulation.md
└── README.md
```

## Requirements

- Julia 1.12 or newer
- Gurobi and/or HiGHS solver support
- Required Julia packages listed in [Project.toml](Project.toml)

## Installation

1. Clone the repository.
2. Open the project in Julia.
3. Activate the environment:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

4. Ensure the solver libraries are available for your environment.

## Usage

Run the model from the project root:

```julia
include("run.jl")
```

or:

```julia
include("run_decarb.jl")
```

The input files in the [in](in) directory drive the simulation and the model writes outputs to the project output location.

## Configuration

- Edit input CSV files under [in](in)
- Adjust model settings in the solver configuration modules under [src/configure_solver](src/configure_solver)
- Update case and scenario logic in the runner components under [src/case_runner](src/case_runner)

## Model notes

The implementation is organized as a modular Julia package that separates:

- input loading
- model construction
- solver configuration
- result reading
- output writing

This keeps the optimization logic easier to extend and maintain as new technologies or constraints are added.

## Documentation

See [docs/formulation.md](docs/formulation.md) for formulation and modeling notes.

## License

This project is licensed under the GNU Affero General Public License v3. See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with a clear explanation of the proposed change.
