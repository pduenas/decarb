"""
Use this file to run DECARB cases by indicating each path to building input
data in file bdg_path.txt, one building per row
"""

# save current path
wd = pwd()      # path to input paths

# load packages
using CSV, DataFrames

# save paths in dataframe
df_bdg_path = CSV.File(joinpath(wd,"bdg_path.txt")) |> DataFrame

# run DECARB for each building
for i in 1:size(df_bdg_path,1)
    bdg = df_bdg_file[i,:]
    empty!(ARGS)
    push!(ARGS,bdg[1])
    include(joinpath(wd,"src","DECARB.jl"))
end
