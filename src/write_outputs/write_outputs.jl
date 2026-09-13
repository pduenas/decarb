"""
write_outputs(path::AbstractString,tmDate,df_chp,df_hvac,df_abp,df_wh,df_pv,
    df_bess,df_wind,df_elec,balance,df_fuel,df_indoor,df_dual,
    df_econ)

Writes optimization model results to CSV files in the output directory

inputs:
path        string path to working directory
tmDate      date/time vector for time series data
df_chp      combined heat and power dataframe
df_hvac     HVAC units dataframe
df_abp      absorption chillers dataframe
df_wh       water heaters dataframe
df_pv       photovoltaic panels dataframe
df_bess     battery energy storage dataframe
df_wind     wind turbines dataframe
df_elec     electric dispatch dataframe
balance     electrical balance array
df_fuel     fuel consumption dataframe
df_indoor   indoor temperature dataframe
df_dual     dual variables dataframe
df_econ     economic results dataframe

returns:
writes three CSV files to path/out/: eq1.csv (economics), eq2.csv (equipment), ts.csv (time series)
Date columns are formatted as mm/dd/yyyy HH:MM for readability
"""

function write_outputs(path::AbstractString,tmDate,df_chp,df_hvac,df_abp,df_wh,df_pv,
    df_bess,df_wind,df_elec,balance,df_fuel,df_indoor,df_dual,
    df_econ)

    dfEQ1 = df_econ
    dfEQ1 = something.(dfEQ1,"0")                   # fix missing values

    dfEQ2 = vcat(df_chp,df_hvac,df_abp,df_wh,df_pv,df_bess,df_wind)
    deleteat!(dfEQ2,findall(isnothing.(dfEQ2.Eq)))  # fix zero values

    dfTS = hcat(DataFrame(Date=tmDate),df_elec,df_fuel,df_indoor,df_dual)
    dfTS = something.(dfTS,"0")                     # fix missing values

    # write file of imbalances if any (warning: cumulative rounding errors)
    if any(abs.(balance).>0.1)
        println("❗  Model solution is unbalanced. See balance.csv file.")
        dfB = DataFrame(Date=tmDate,B_Q=balance)
        CSV.write(joinpath(path,"out","balance.csv"),dfB,dateformat="mm/dd/yyyy HH:MM"; header=false)
    else
        println("Model solution is balanced.")
    end

    # write outputs in CSV files
    CSV.write(joinpath(path,"out","eq1.csv"),dfEQ1)
    CSV.write(joinpath(path,"out","eq2.csv"),dfEQ2,dateformat="mm/dd/yyyy HH:MM")
    CSV.write(joinpath(path,"out","ts.csv"),dfTS,dateformat="mm/dd/yyyy HH:MM")

end
