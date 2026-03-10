using Dates, CSV, DataFrames, HTTP, JSON

weather_key = "6e00acb71be627c9a97e5078b2774fb6"

global this_dt = DateTime(2018,1,1,0)
original_dt = this_dt

off_peak = 0.13
mid_peak = 0.13
peak = 0.30

dt_vector = []
price_vector = []
cap_vector = []

START_AM_PEAK = 7
END_AM_PEAK = 11

START_PM_PEAK = 18
END_PM_PEAK = 22

while Dates.year(this_dt) < 2019
    push!(dt_vector, this_dt)
    if Dates.hour(this_dt) < 6
        push!(price_vector,off_peak)
        push!(cap_vector, 0)
    elseif Dates.hour(this_dt) >= 6 && Dates.hour(this_dt) < 11
        push!(price_vector,peak)
        push!(cap_vector, 1)
    elseif Dates.hour(this_dt) >= 11 && Dates.hour(this_dt) < 17
        push!(price_vector,mid_peak)
        push!(cap_vector, 0)
    elseif Dates.hour(this_dt) >= 17 && Dates.hour(this_dt) < 21
        push!(price_vector, peak)
        push!(cap_vector, 1)
    else
        push!(price_vector, off_peak)
        push!(cap_vector, 0)
    end
    global this_dt = this_dt + Dates.Hour(1)
end

df = DataFrame(Datetime = dt_vector, Price = price_vector, Is_Peak = cap_vector)
CSV.write("tou_file_13_30.csv", df)

resp_city = HTTP.request("GET", "http://api.openweathermap.org/geo/1.0/direct", headers=["Content-Type" => "application/json"], query=["q" => "Boston,MA,USA", "limit" => 1, "appid" => weather_key])
println("Resp City")
try
    global jobj = JSON.parse(String(resp_city.body))
catch e
    println(e)
end

try
    global lat = jobj[1]["lat"]
    println(lat)
    global lon = jobj[1]["lon"]
    println(lon)
catch e
    println("City response dictionary not formatted correctly: $(jobj[1])")
    println(e)
end

resp = HTTP.request("GET", "https://history.openweathermap.org/data/2.5/history/city", headers=["Content-Type" => "application/json"], query=["lat" => lat, "lon" => lon, "type" => "hour", "start"=> Dates.datetime2unix(original_dt), "end" => Dates.datetime2unix(this_dt), "appid" => weather_key])
resp_obj = JSON.parse(String(resp.body))
println(resp_obj)