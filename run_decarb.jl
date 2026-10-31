"""
Batch driver for DECARB cases.

Usage:
    julia --project=. run_decarb.jl [--solver highs|gurobi] [case-list.txt]

The default solver is HiGHS. The case list defaults to `bdg_path.txt`.
"""

using DECARB

# Solver packages must be imported at top level, before run_batch starts.
# To add a solver: (1) accept its name in parse_args and usage, (2) add a branch in
# optimizer_for, (3) add an `import` for its package in the block below.
function requested_solver(args::Vector{String})
    for (i, arg) in enumerate(args)
        arg == "--solver" && i < length(args) && return lowercase(args[i + 1])
        startswith(arg, "--solver=") && return lowercase(split(arg, "="; limit=2)[2])
    end
    return "highs"
end

if requested_solver(ARGS) == "gurobi" && Base.find_package("Gurobi") !== nothing
    import Gurobi
end

function usage(io::IO=stdout)
    println(io, """
    Usage:
        julia --project=. run_decarb.jl [--solver highs|gurobi] [case-list.txt]

    Options:
        --solver NAME   Solver to use: highs (default) or gurobi.
        -h, --help      Show this help message.
    """)
end

function parse_args(args::Vector{String})
    solver = "highs"
    list_file = nothing
    i = 1

    while i <= length(args)
        arg = args[i]

        if arg in ("-h", "--help")
            usage()
            return nothing
        elseif arg == "--solver"
            i == length(args) && error("--solver requires a solver name")
            solver = lowercase(args[i + 1])
            i += 2
        elseif startswith(arg, "--solver=")
            solver = lowercase(split(arg, "="; limit=2)[2])
            i += 1
        elseif startswith(arg, "-")
            error("unknown option: $arg")
        elseif isnothing(list_file)
            list_file = arg
            i += 1
        else
            error("expected at most one case-list path")
        end
    end

    solver in ("highs", "gurobi") ||
        error("unsupported solver '$solver'; use highs or gurobi")

    return (
        solver,
        isnothing(list_file) ? joinpath(@__DIR__, "bdg_path.txt") : abspath(list_file),
    )
end

function optimizer_for(solver::String)
    solver == "highs" && return nothing  # DECARB's default optimizer

    Base.find_package("Gurobi") === nothing &&
        error("Gurobi is not available in the active Julia environment. " *
              "Install it there, then rerun with --solver gurobi.")

    @eval import Gurobi

    return () -> Base.invokelatest(
        getfield(
            Base.invokelatest(getfield, @__MODULE__, :Gurobi),
            :Optimizer,
        ),
    )
end

function print_summary(results)
    rows = [
        (result.case, result.status, string(round(result.elapsed; digits=2)), result.error)
        for result in results
    ]
    headers = ("case", "status", "elapsed s", "error message")
    widths = [
        maximum(length(row[i]) for row in (headers, rows...))
        for i in eachindex(headers)
    ]

    function print_row(row)
        println(join((rpad(row[i], widths[i]) for i in eachindex(row)), " | "))
    end

    print_row(headers)
    println(join((repeat("-", width) for width in widths), "-+-"))
    foreach(print_row, rows)
end

function run_batch(args::Vector{String}=ARGS; optimizer=nothing)
    parsed = parse_args(args)
    isnothing(parsed) && return nothing

    solver, list_file = parsed
    isfile(list_file) || error("case list not found: $list_file")

    selected_optimizer = isnothing(optimizer) ? optimizer_for(solver) : optimizer

    lines = strip.(readlines(list_file))
    case_paths = filter(line -> !isempty(line) && !startswith(line, "#"), lines)
    results = NamedTuple[]

    for raw_path in case_paths
        path = strip(raw_path, ['"', '\''])
        path = isabspath(path) ? normpath(path) : normpath(joinpath(dirname(list_file), path))
        case = basename(path) == "in" ? dirname(path) : path
        started = time()

        try
            if isnothing(selected_optimizer)
                DECARB.run_decarb!(case)
            else
                DECARB.run_decarb!(case; optimizer=selected_optimizer)
            end
            push!(results, (case=case, status="success", elapsed=time() - started, error=""))
        catch err
            message = replace(sprint(showerror, err), '\n' => ' ', '\r' => ' ')
            push!(results, (case=case, status="failed", elapsed=time() - started, error=message))
        end
    end

    print_summary(results)
    return any(result -> result.status == "failed", results)
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_batch() === true && exit(1)
end
