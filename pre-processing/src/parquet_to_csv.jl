using Parquet, DataFrames
using CSV

wd = pwd()
println("\u2328  type working directory or press <Enter> for ", wd)
dir_name = readline(stdin)
dir_name=="" ? dir_name=joinpath(wd,"in/resstock_files") : isdir(dir_name)==false ? error("\u26A0 ", dir_name," is not a directory.\n") :  println("\u21B3 is now the working directory.\n")

println("Type name of file")
fname = readline(stdin)
fname == "" ? error("$(fname) must not be blank") : println("Reading $(fname) into DataFrame")
println(fname)

df = DataFrame(read_parquet(joinpath(dir_name, fname)))

csv_index = findfirst(".parquet", fname)
fname_prefix = fname[1:(csv_index[1]-1)]
out_fname = fname_prefix * "_converted.csv"
println(out_fname)

CSV.write(joinpath(dir_name, out_fname), df)