"""
bess_modules!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,sp::Dict,bess::Dict)

Creates variables, expressions and constraints associated to BESS modules

inputs:
model   name of core model
cfg     dictionary with configuration input data
tm      dictionary with time series data
bdg     dictionary with building data
sp      dictionary with equipment selection data
bess    dictionary with BESS data

"""
function bess_modules!(model::Model,cfg::Dict,tm::Dict,bdg::Dict,sp::Dict,bess::Dict)

    # unitary state-of-charge of multiple BESS [0,z]
    @variable(model, bdg["Bbess"] >= vBESSsoc[t=0:tm["P"],s=1:bess["N"]] >= 0)
    # unitary electricity charged in multiple BESS [0,z]
    @variable(model,
        bdg["Bbess"]*tm["TM"][t]*bess["up"][s]/bess["mx"][s] >= vBESSup[t=1:tm["P"],s=1:bess["N"]] >= 0)
    # unitary electricity discharged from multiple BESS [0,z]
    @variable(model,
        bdg["Bbess"]*tm["TM"][t]*bess["dn"][s]/bess["mx"][s] >= vBESSdn[t=1:tm["P"],s=1:bess["N"]] >= 0)
    # charge/discharge mode of BESS {0,1}
    @variable(model, bBESS[t=1:tm["P"],s=1:bess["N"]], Bin)
    # number of installed BESS {0,z}
    @variable(model, 0 >= zBESS[i=1:cfg["IT"],s=1:bess["N"]] >= 0, Int)

    bess["zmx0"] = zeros(Int16, bess["N"])      # maximum number of potential existing BESS modules
    for s in findall(sp["BESSyn"].!="0")
        # fix already installed BESS modules
        if sp["BESSyn"][s] == "NO"
            set_upper_bound.(zBESS[1,bess["ty"].==sp["BESS0"][s]],sp["BESSz0"][s])
            set_lower_bound.(zBESS[1,bess["ty"].==sp["BESS0"][s]],sp["BESSz0"][s])
            bess["zmx0"][bess["ty"].==sp["BESS0"][s]] .= sp["BESSz0"][s]
        # set bounds of potential BESS modules
        elseif sp["BESSyn"][s] == "YES"
            set_upper_bound.(zBESS[:,bess["ty"].==sp["BESS0"][s]],bdg["Bbess"])
            set_lower_bound.(zBESS[:,bess["ty"].==sp["BESS0"][s]],sp["BESSz0"][s])
            bess["zmx0"][bess["ty"].==sp["BESS0"][s]] .= bdg["Bbess"]
        end
    end

    # electricity charged in BESS module [kWh]
    @expression(model, vBESS_UP[t=1:tm["P"],s=1:bess["N"]], bess["mx"][s]*vBESSup[t,s])
    # electricity discharged from BESS module [kWh]
    @expression(model, vBESS_DN[t=1:tm["P"],s=1:bess["N"]], bess["mx"][s]*vBESSdn[t,s])

    # maximum available space for BESS modules {0,z}
    @constraint(model, eBESSbdg, sum(zBESS[i,s] for i=1:cfg["IT"],s=1:bess["N"]) <= bdg["Bbess"])
    for i1=1:cfg["IT"]
    	# maximum electricity stored by BESS [0,z]
    	@constraint(model, eBESSmx[t=tm["IW"][i1]:tm["P"],s=1:bess["N"]; bess["zmx0"][s]>0],
    		vBESSsoc[t,s] <= sum(zBESS[i2,s] for i2=1:i1))
    	# maximum electricity charged to BESS [0,z]
    	@constraint(model, eBESSup[t=tm["IW"][i1]:tm["P"],s=1:bess["N"]; bess["zmx0"][s]>0],
    		vBESSup[t,s] <= sum(zBESS[i2,s] for i2=1:i1)*tm["TM"][t]*bess["up"][s]/bess["mx"][s])
    	# maximum electricity discharged from BESS [0,z]
    	@constraint(model, eBESSdn[t=tm["IW"][i1]:tm["P"],s=1:bess["N"]; bess["zmx0"][s]>0],
    		vBESSdn[t,s] <= sum(zBESS[i2,s] for i2=1:i1)*tm["TM"][t]*bess["dn"][s]/bess["mx"][s])
    end

    # charge/discharge mode of BESS {0,1}
    @constraint(model, eBESSud[t=1:tm["P"],s=1:bess["N"]; bess["zmx0"][s]>0],
    	vBESSup[t,s] <= bess["zmx0"][s]*tm["TM"][t]*bess["up"][s]/bess["mx"][s]*bBESS[t,s])
    @constraint(model, eBESSdu[t=1:tm["P"],s=1:bess["N"]; bess["zmx0"][s]>0],
    	vBESSdn[t,s] <= bess["zmx0"][s]*tm["TM"][t]*bess["dn"][s]/bess["mx"][s]*(1-bBESS[t,s]))
    # electricity stored balance [0,z]
    @constraint(model, eBESSbal[t=1:tm["P"],s=1:bess["N"]; bess["zmx0"][s]>0],
        vBESSsoc[t,s]-vBESSsoc[t-1,s] == vBESSup[t,s]*bess["effu"][s]-vBESSdn[t,s]/bess["effd"][s] +
    	sum(cfg["BESSsoc0"]*zBESS[i,s] for i=2:cfg["IT"] if t==tm["IW"][i]))
    # fix initial SOC [0,z]
    @constraint(model, eBESSi[s=1:bess["N"]; bess["zmx0"][s]>0],
    	vBESSsoc[0,s] == cfg["BESSsoc0"]*zBESS[1,s])
    # fix final SOC [0,z]
    @constraint(model, eBESSf[s=1:bess["N"]; bess["zmx0"][s]>0],
    	vBESSsoc[tm["P"],s] == cfg["BESSsocf"]*sum(zBESS[i,s] for i=1:cfg["IT"]))

end