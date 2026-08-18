"""
load_bdgi(path::AbstractString)

Loads bdg_i.csv file from path directory and stores values in a dictionary object

inputs:
path    string path to working directory

returns bdg-type inputs in dictionary object
"""

function load_bdgi(path::AbstractString)

    # declare dictionary object to store parameters
    bdg = Dict()

    path2file = joinpath(path,"bdg_i.csv")

    # load file into dataframe with predefined types
    df_bdg = CSV.File(path2file;delim=',',types=[Float64,Float64,Float64,String,Int8,
                    Float64,Float64,Float64,Float64,Float64,Int8,Float64,Float64,
                    Float64,Float64,Float64,Float64,Float64,Float64,Float64,Float64,
                    Float64,Float64,Float64,Int8,Int8,Int8,Int8,Float64,Float64,Float64,
                    Int8,Int8,Int8,Float64],transpose=true) |> DataFrame

    bdg["Balt"] = df_bdg.pBalt[1]       # altitude above sea level [m]
    bdg["Blon"] = df_bdg.pBlon[1]		# longitude [°]
    bdg["Blat"] = df_bdg.pBlat[1]		# longitude [°]
    bdg["Btz"] = df_bdg.pBtz[1]		    # time zone
    bdg["Bventy"] = df_bdg.pBventy[1]   # ventilation strategy {0,1}
    bdg["Bvent"] = df_bdg.pBvent[1]     # ventilation requirement [l/s/person]
    bdg["Bihgp"] = df_bdg.pBihgp[1]     # internal heat gain from people [kW/person]
    bdg["Bihgl"] = df_bdg.pBihgl[1]     # internal heat gain from lighting [%]
    bdg["Bihge"] = df_bdg.pBihge[1]     # internal heat gain from equipment [%]
    bdg["Bfoot"] = df_bdg.pBfoot[1]     # building footprint surface [m2]
    bdg["Bslab"] = df_bdg.pBslab[1]     # number of concrete slabs {0,z}
    bdg["Btslab"] = df_bdg.pBtslab[1]   # slab thickness [m]
    bdg["Bkslab"] = df_bdg.pBkslab[1]   # slab heat transfer coefficient [W/m2-°C]
    bdg["Bpslab"] = df_bdg.pBpslab[1]   # density of slab material [kg/m3]
    bdg["Bcpslab"] = df_bdg.pBcpslab[1] # specific heat of slab material [J/kg-°C]
    bdg["Bkroof"] = df_bdg.pBkroof[1]   # roof solar heat gain factor [%]
    bdg["Bwest"] = df_bdg.pBwest[1]     # west lateral surface [m2]
    bdg["Bkwest"] = df_bdg.pBkwest[1]   # west lateral solar heat gain factor [%]
    bdg["Beast"] = df_bdg.pBeast[1]     # east lateral surface [m2]
    bdg["Bkeast"] = df_bdg.pBkeast[1]   # east lateral solar heat gain factor [%]
    bdg["Bnorth"] = df_bdg.pBnorth[1]   # north lateral surface [m2]
    bdg["Bknorth"] = df_bdg.pBknorth[1] # north lateral solar heat gain factor [%]
    bdg["Bsouth"] = df_bdg.pBsouth[1]   # south lateral surface [m2]
    bdg["Bksouth"] = df_bdg.pBksouth[1] # south lateral solar heat gain factor [%]
    bdg["Bchp"] = df_bdg.pBchp[1]       # available CHP space {0,z}
    bdg["Bhvac"] = df_bdg.pBhvac[1]     # available HVAC space {0,z}
    bdg["Babs"] = df_bdg.pBabs[1]       # available ABS space {0,z}
    bdg["Bwh"] = df_bdg.pBwh[1]         # available WH space {0,z}
    bdg["Bpv"] = df_bdg.pBpv[1]         # available PV (roof) surface [m2]
    bdg["Btilt"] = df_bdg.pBtilt[1] 	# PV tilt (roof) angle [°]
    bdg["Bazi"] = df_bdg.pBazi[1]     	# PV azimuth (roof) angle [°]
    bdg["Btck"] = df_bdg.pBtck[1]   	# tracking system {0,2}
    bdg["Bwind"] = df_bdg.pBwind[1]     # available wind space {0,z}
    bdg["Bbess"] = df_bdg.pBbess[1]     # available BESS space {0,z}
    bdg["Bevmx"] = df_bdg.pBevmx[1]     # EV maximum charging capacity [kW]

    # unit transformations
    bdg["Bvent"] = bdg["Bvent"]/1000*3600   # in [m3/h/person]
    bdg["Bkslab"] = bdg["Bkslab"]/1000      # in [kW/m2-°C]
    bdg["Bcpslab"] = bdg["Bcpslab"]/3.6e6   # in [kWh/kg-°C]
    bdg["Btilt"] = bdg["Btilt"]*pi/180      # in [rad]
    bdg["Bazi"] = bdg["Bazi"]*pi/180     	# in [rad]

    return bdg

end