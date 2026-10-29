# Contributing to DECARB

Thank you for your interest in contributing to DECARB.

DECARB is an open-source Julia-based optimization model for distributed energy
systems, building-level energy coupling, operational scheduling, and
decarbonization-oriented planning.

Contributions are welcome, including:

- Bug reports
- Documentation improvements
- Test cases
- Input-data improvements
- Performance improvements
- New technologies or constraints
- Improvements to the optimization formulation

Please read the [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) before contributing.

## Reporting issues

Before opening an issue, search the existing issues to avoid duplicates.

Please include:

- A clear description of the problem or proposed improvement
- DECARB version or commit hash
- Julia version
- Operating system
- Solver and solver version
- Relevant input files or configuration
- Commands used to reproduce the issue
- Complete error messages
- Expected and actual behavior

Do not include passwords, API keys, solver credentials, private datasets, or
other confidential information in issues.

## Development setup

Clone the repository and activate the Julia project:

```bash
git clone <repository-url>
cd decarb
julia --project=.
```

Instantiate the project environment:

```julia
using Pkg

Pkg.instantiate()
```

HiGHS is the default open-source solver. Other solvers, including licensed ones, are
welcome. DECARB does not depend on them: users install and load the solver package
themselves and pass its optimizer to `run_decarb!`. Solver-specific settings follow
this pattern:

1. Add `src/configure_solver/configure_<solver>.jl` with a function that sets the
   solver's raw attributes (gap, time limit, and any tuning) on the model, and
   include it in `src/DECARB.jl`.
2. Add a branch in `configure_solver!` (`src/configure_solver/configure_solver.jl`)
   that detects the solver from `solver_name(model)` and calls that function.
   Unrecognized solvers fall back to the generic MOI gap and time-limit attributes.
3. Do not add the solver package to `[deps]` or `[compat]` in `Project.toml`.
4. Add a test using a recording MOI optimizer, as in the existing Gurobi test, and
   document the required installation, loading, and licensing steps in the README.

## Branches

Create a separate branch for each contribution:

```bash
git checkout main
git pull --ff-only
git checkout -b feature/short-description
```

Examples:

```text
feature/add-thermal-storage
fix/input-validation
docs/update-installation
test/solar-potential
```

Keep each branch focused on one logical change.

## Code and model changes

Please follow the existing Julia style and repository organization.

When modifying the optimization model:

- Document the physical meaning and units of new variables and parameters.
- Explain changes to constraints or the objective function.
- Preserve existing indexing and naming conventions where possible.
- Update `docs/formulation.md` when the mathematical formulation changes.
- Update input or output documentation when file formats change.
- Avoid unrelated formatting or refactoring changes.

The main source directories are:

```text
src/case_runner/       # Complete model workflow
src/configure_solver/  # Solver configuration
src/decarb_model/      # Model variables, constraints, and objective
src/load_inputs/       # Input loading and validation
src/solve_model/       # Optimization and feasibility checks
src/read_outputs/      # Result extraction
src/write_outputs/     # Output-file generation
```

## Testing

Run the project test suite:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

The suite includes regression tests that run `examples/minimal` and compare
`total_cost`, `energy_bill` and emissions (direct + indirect) in `eq1.csv` with the
committed `examples/minimal/out/eq1.csv` (relative tolerance `max(1e-3, 2*MIP_GAP)`).

### Regenerating regression fixtures

Only regenerate fixtures when a change intentionally alters results, and explain the
change in the pull request. From the repository root, run each case with the settings
used by `run.jl` (HiGHS, `mip_gap = 1e-4`):

```bash
julia --project=. run.jl examples/minimal
julia --project=. run.jl examples/annual
julia --project=. run.jl examples/investment
```

This overwrites `examples/<case>/out/`. Review the diff of `eq1.csv` (and the other
outputs), then commit the updated files. Do not commit `balance.csv` or other
stray files from failed runs.

For changes affecting model execution, run a representative case:

```julia
using DECARB, HiGHS

DECARB.run_decarb!("examples/minimal"; optimizer=HiGHS.Optimizer)
```

Check that:

- Input validation succeeds.
- The model produces a feasible solution.
- Expected output files are created.
- The electrical balance is acceptable.
- No unexpected errors or warnings occur.

When reporting numerical results, include:

- DECARB version or commit hash
- Julia version
- Solver and solver version
- MIP gap
- Time limit
- Whether integrality was relaxed
- Input-data version

## Input and output files

Input files define the model case and can significantly affect results.

When modifying files in `in/`:

- Preserve the CSV structure.
- Verify units and data types.
- Check timestamps and time-step consistency.
- Confirm that equipment names match the catalogs.
- Explain the effect of the change on reproducibility.

Do not commit private or restricted datasets.

Generated files in `out/` should generally not be committed unless they are
intentionally included as examples or regression-test fixtures.

## Pull requests

Before opening a pull request:

1. Update your branch with the latest target branch.
2. Run the relevant tests.
3. Run a representative model case if appropriate.
4. Review the complete diff.
5. Remove temporary files and generated artifacts.
6. Update the relevant documentation.
7. Confirm that no sensitive information is included.

Pull requests should include:

- A summary of the change
- The motivation for the change
- Testing performed
- Any formulation changes
- Any changes to inputs or outputs
- Potential effects on numerical results
- Known limitations

A suitable commit-message format is:

```text
<type>: <short description>
```

Examples:

```text
fix: validate zero-capacity equipment rows
feat: add thermal storage constraints
docs: update solver instructions
test: add PV availability regression case
```

## Licensing

DECARB is distributed under the GNU Affero General Public License version 3.0.

By contributing to this repository, you agree that your contribution may be
distributed under the same license, unless otherwise agreed with the project
maintainers.

When incorporating third-party code or documentation:

- Verify that its license permits use.
- Preserve required copyright notices.
- Provide attribution where required.
- Document the source and license.

## Questions and sensitive reports

For general questions, open a GitHub issue or discussion.

For abusive, harassing, or otherwise unacceptable behavior, follow the
reporting instructions in [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).

Do not report sensitive incidents through a public GitHub issue or pull request.
