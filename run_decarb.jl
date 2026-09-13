"""
Batch driver for DECARB cases.

List one case directory per line in `bdg_path.txt`. A case directory contains
an `in/` subdirectory. Blank lines and lines beginning with `#` are ignored.
Pass a different list file as the first command-line argument if needed.
"""

using DECARB

list_file = isempty(ARGS) ? joinpath(@__DIR__, "bdg_path.txt") : abspath(ARGS[1])
isfile(list_file) || error("case list not found: $(list_file)")

lines = strip.(readlines(list_file))
case_paths = filter(line -> !isempty(line) && !startswith(line, "#"), lines)

for raw_path in case_paths
    path = strip(raw_path, ['\"', '\''])
    case = basename(normpath(path)) == "in" ? dirname(normpath(path)) : normpath(path)
    DECARB.run_decarb!(case; solver=:highs)
end
