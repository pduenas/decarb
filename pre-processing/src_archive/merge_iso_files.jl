using CSV, DataFrames, Dates

wd = pwd()
dfA = CSV.File(joinpath(wd, "in", "isone_files", "da_lmp_2018.csv")) |> DataFrame

statsdf = describe(dfA, :mean, cols="DA Locational Marginal Price")
global mean_da = statsdf.mean[1]
println(mean_da)

dt_vector = []
da_price_vector = []
for row in eachrow(dfA)
    local hr = row."Hour Ending"

    local date_temp = DateTime(row.Date,"mm/dd/yyyy") + Dates.Year(2000)

    true_hr = hr - 1
    dt = DateTime(Dates.year(date_temp), Dates.month(date_temp), Dates.day(date_temp), true_hr)
    push!(dt_vector, dt)

    da_price = row."DA Locational Marginal Price"
    #println(da_price)
    local price_adj = (row."DA Locational Marginal Price" - mean_da) / 1000.0
    push!(da_price_vector, price_adj)
end

write_df = DataFrame(Datetime = dt_vector, Price_Adjustor = da_price_vector)
CSV.write("2018_hourly_da.csv", write_df)
