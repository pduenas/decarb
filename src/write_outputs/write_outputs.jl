"""
write_outputs(path::String,tmDate,df_chp,df_hvac,df_abp,df_wh,df_pv,
    df_bess,df_wind,df_elec,balance,df_fuel,df_indoor,df_dual,
    df_econ)

Writes optimization model results to CSV files in the output directory

inputs:
path        string path to working directory
tmDate      date/time vector for time series data [DateTime]
df_chp      combined heat and power dataframe (includes Date, Win columns)
df_hvac     HVAC units dataframe (includes Date, Win columns)
df_abp      absorption chillers dataframe (includes Date, Win columns)
df_wh       water heaters dataframe (includes Date, Win columns)
df_pv       photovoltaic panels dataframe (includes Date, Win columns)
df_bess     battery energy storage dataframe (includes Date, Win columns)
df_wind     wind turbines dataframe (includes Date, Win columns)
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

function write_outputs(path::String,tmDate,df_chp,df_hvac,df_abp,df_wh,df_pv,
    df_bess,df_wind,df_elec,balance,df_fuel,df_indoor,df_dual,
    df_econ)

    dfEQ1 = df_econ
    dfEQ1 = something.(dfEQ1,"0")                   # fix missing values

    dfEQ2 = vcat(df_chp,df_hvac,df_abp,df_wh,df_pv,df_bess,df_wind)
    delete!(dfEQ2,findall(isnothing.(dfEQ2.Eq)))    # fix zero values

    dfTS = hcat(DataFrame(Date=tmDate),df_elec,df_fuel,df_indoor,df_dual)
    dfTS = something.(dfTS,"0")                     # fix missing values

    # write file of imbalances if any
    if any(balance.!=0)
        dfB = DataFrame(Date=tmDate,B_Q=balance)
        CSV.write(joinpath(path,"balance.csv"),dfB,dateformat="mm/dd/yyyy HH:MM"; header=false)
    else
        println("Model soluton is balanced.")
    end

    # write outputs in CSV files
    CSV.write(joinpath(path,"out","eq1.csv"),dfEQ1)
    CSV.write(joinpath(path,"out","eq2.csv"),dfEQ2,dateformat="mm/dd/yyyy HH:MM")
    CSV.write(joinpath(path,"out","ts.csv"),dfTS,dateformat="mm/dd/yyyy HH:MM")

end