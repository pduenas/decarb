"""
temperature_variables!(model::Model,tm::Dict)

Creates variables associated to temperature variables

inputs:
model   name of core model
tm      dictionary with time series data

"""
function temperature_variables!(model::Model,tm::Dict)

    # Bounds on extreme indoor temperatures [°C]
    Tup = maximum([tm["Tout"];tm["Tmx"]], dims=1)[1]
    Tlo = minimum([tm["Tout"];tm["Tmn"]], dims=1)[1]

    # indoor temperature [°C]
    @variable(model, Tup >= vTin[t=1:tm["P"]] >= Tlo)
    # excursion up discomfort temperature [°C]
    @variable(model, Tup.-tm["Tmx"][t] >= vTup[t=1:tm["P"]] >= 0)
    # excursion low discomfort temperature [°C]
    @variable(model, tm["Tmn"][t].-Tlo >= vTlo[t=1:tm["P"]] >= 0)

end