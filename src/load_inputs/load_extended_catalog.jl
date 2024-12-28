"""
load_extended_catalog(path::AbstractString,tm::Dict,sp::Dict,bdg::Dict)

Loads equipment csv files that might require an extension when multiple units exist
    from path directory and stores values in respective dictionary objects

inputs:
path    string path to working directory
tm      dictionary with time series data
sp      dictionary with equipment selection
bdg     dictionary with building data

returns catalog inputs: chp, abs, hvac, wh in dictionary objects
"""

function load_extended_catalog(path::AbstractString,tm::Dict,sp::Dict,bdg::Dict)

    chp = load_chp(path,sp,bdg)
    abs = load_abs(path,sp,bdg)
    hvac = load_hvac(path,tm,sp,bdg)
    wh = load_wh(path,sp,bdg)
    ev = load_ev(path,tm,sp)

    return chp,abs,hvac,wh,ev

end



"""
extend_catalog(df::DataFrame,ty::String,z0::UInt8,yn::String,bdg::UInt8)
"""

function extend_catalog(df,ty,z0,yn,bdg)

    for i in findall(yn.=="YES" .|| yn.=="NO")
        # for all potential investment in units
        if yn[i] == "YES"
            # replicate new lines of type of unit up to maximum number of units in building
            if bdg > 1
                append!(df,repeat(df[df.ty.==ty[i],:],bdg-1))
            end
            # update name of units of coincident type for new lines
            if bdg > 0
                df.ty[df.ty.==ty[i]] = string.(df.ty[df.ty.==ty[i]],'-',Vector(1:bdg))
            end
        # when no potential investment in units
        elseif yn[i] == "NO"
            # replicate new lines of type of unit up to existing number of units in building
            if z0[i] > 1
                append!(df,repeat(df[df.ty.==ty[i],:],z0[i]-1))
            end
            # update name of units of coincident type for new lines
            if z0[i] > 0
                df.ty[df.ty.==ty[i]] = string.(df.ty[df.ty.==ty[i]],'-',Vector(1:z0[i]))
            end
        end
    end

    return df

end