# Inputs and outputs

DECARB reads a case from a directory containing an `in/` folder and writes
results to an `out/` folder in the same case directory.

```text
case/
|-- in/
|   |-- cfg.csv
|   |-- tm.csv
|   |-- bdg_i.csv
|   |-- bdg_ii.csv
|   |-- sp.csv
|   |-- topo.csv
|   |-- chp.csv
|   |-- abp.csv
|   |-- hvac.csv
|   |-- wh.csv
|   |-- pv.csv
|   |-- wind.csv
|   |-- bess.csv
|   `-- ev.csv
`-- out/
    |-- eq1.csv
    |-- eq2.csv
    `-- ts.csv
```

The current runner opens all 14 input files, even when a technology is not
selected. Use the exact column names and order shown in the bundled
[`examples/minimal/in`](../examples/minimal/in) case. CSV files are
comma-delimited. Names and option values are case-sensitive.

## Input files

### `cfg.csv`: global configuration

This is a two-column, headerless parameter/value file. Timestamps use
`yyyy-mm-ddTHH:MM:SS`; booleans use `true` or `false`.

| Parameter | Type/unit | Description |
|---|---|---|
| `pP0` | local date-time | Start of the first model period. It must precede the first `pP` in `tm.csv`. |
| `pTmode` | boolean | Enables indoor-temperature comfort control. If false, all `pTon` values are disabled. |
| `pBESSsoc0` | fraction | Initial battery state of charge, in `[0,1]`. |
| `pBESSsocf` | fraction | Required final battery state of charge, in `[0,1]`. |
| `pEVsoc0` | fraction | Initial EV state of charge, in `[0,1]`. |
| `pEVmnsoc` | fraction | Minimum preferred EV state of charge, in `[0,1]`; shortfall is allowed at `pNSEVcost`. |
| `pWHsto0` | fraction | Initial water-heater tank state, in `[0,1]`. |
| `pWHstof` | fraction | Required final water-heater tank state, in `[0,1]`. |
| `pTin0` | deg C | Initial indoor temperature. |
| `pQmxBuy` | kW | Maximum grid import; must be positive. |
| `pQmxSell` | kW | Maximum grid export; must be non-negative. |
| `pST` | h | Reserved survival-time setting; currently inactive. |
| `pGco2` | kg/kWh | Direct CO2 emissions factor for gaseous fuel. |
| `pLco2` | kg/kWh | Direct CO2 emissions factor for liquid fuel. |
| `pCO2mx` | metric ton | Reserved emissions limit; currently inactive. |
| `pCO2cost` | $/metric ton | Reserved carbon price; currently inactive. |
| `pCO2ton` | metric ton | Reserved emissions offset; currently inactive. |
| `pQmxTM` | integer | Number of peak-charge groups. Use `0` to disable peak charges. |
| `pNSEcost` | $/kWh | Penalty for unserved electricity. A zero value prevents unserved electricity. |
| `pNSTcost` | $/(deg C h) | Penalty for indoor-temperature deviation. A zero value prevents comfort-bound violations when comfort is active. |
| `pNSHWcost` | $/kWh | Penalty for unserved hot water. A zero value prevents unserved hot water. |
| `pNSEVcost` | $ per unit SOC | Penalty for EV state-of-charge shortfall. |
| `pEVdriver` | `-1`, `0`, or `1` | Driver preference: range-anxious, indifferent, or battery-conscious, respectively. |
| `pEVpen` | $ per unit SOC | Weight applied to the driver-preference term. |
| `pEVv2g` | boolean | Allows EV discharge to the building/grid when true. |
| `pIT` | integer | Number of investment windows. `0` disables investment and creates one internal window for existing assets. |
| `pIR` | fraction/year | Annual interest rate used to annualize capital costs; must be between 0 and 1. |

### `tm.csv`: time series

Each row is one model period. `pP` is the period-end timestamp in the local time
zone declared by `pBtz`. Period duration is calculated from consecutive
timestamps, starting with `pP0`; timestamps must therefore be sorted and
strictly increasing. Variable-duration periods are supported.

| Column | Type/unit | Description |
|---|---|---|
| `pP` | local date-time | Period-end timestamp. |
| `pQcostBuy` | $/kWh | Grid electricity purchase price. |
| `pQcostSell` | $/kWh | Grid electricity sale price. |
| `pQmx` | integer | Peak-charge group for this period; `0` means none, otherwise use `1` through `pQmxTM`. |
| `pQmxCost` | $/kW | Charge for the peak group. Each group should use a consistent value. |
| `pQco2` | kg/kWh | Grid electricity emissions factor. |
| `pGcost` | $/kWh | Gaseous-fuel price. |
| `pLcost` | $/kWh | Liquid-fuel price. |
| `pQlight` | kW | Lighting electric demand. |
| `pQequip` | kW | Other equipment electric demand. |
| `pTout` | deg C | Outdoor temperature. |
| `pTmx` | deg C | Maximum comfortable indoor temperature. |
| `pTmn` | deg C | Minimum comfortable indoor temperature; must not exceed `pTmx`. |
| `pTon` | `0` or `1` | Enforces the comfort bounds in this period when `pTmode` is true. |
| `pHWdem` | kWh/period | Domestic hot-water demand. |
| `pSdni` | kW/m2 | Direct normal solar irradiance. |
| `pSdhi` | kW/m2 | Diffuse horizontal solar irradiance. |
| `pWms` | m/s | Wind speed. |
| `pEVdem` | km | EV trip distance at a departure event. |
| `pEVtm` | h | Plugged-in charging window beginning at an arrival event. |
| `pEVtype` | integer | EV event identifier: `0` for no event, otherwise the 1-based selected EV instance. Paired occurrences mark departure and arrival. |
| `pBppl` | people | Building occupancy, used to calculate heat gains from people. |

### `bdg_i.csv`: site and building

This is a two-column, headerless parameter/value file.

| Parameter | Type/unit | Description |
|---|---|---|
| `pBalt` | m | Altitude above sea level. |
| `pBlon` | degrees | Longitude. |
| `pBlat` | degrees | Latitude. |
| `pBtz` | IANA time-zone name | Local time zone, for example `America/New_York`. |
| `pBihgp` | kW/person | Internal heat gain from people. |
| `pBihgl` | fraction | Fraction of lighting demand converted to internal heat. |
| `pBihge` | fraction | Fraction of equipment demand converted to internal heat. |
| `pBfoot` | m2 | Building footprint/roof surface used for solar heat gain. |
| `pBkroof` | fraction | Roof solar heat-gain factor. |
| `pBwest`, `pBeast`, `pBnorth`, `pBsouth` | m2 | Exterior wall areas by orientation. |
| `pBkwest`, `pBkeast`, `pBknorth`, `pBksouth` | fraction | Wall solar heat-gain factors by orientation. |
| `pBchp` | integer units | Maximum CHP installation count. |
| `pBhvac` | integer units | Maximum HVAC installation count. |
| `pBabp` | integer units | Maximum absorption-chiller installation count. |
| `pBwh` | integer units | Maximum water-heater installation count. |
| `pBpv` | m2 | Roof area available to PV. |
| `pBtilt` | degrees | PV surface tilt. |
| `pBazi` | degrees | PV surface azimuth; the minimal example uses `180` for south. |
| `pBalb` | fraction | Ground/roof albedo used in the PV model. |
| `pBtck` | `0`, `1`, or `2` | PV tracking: fixed, single-axis, or dual-axis. |
| `pBwind` | integer units | Maximum wind-turbine installation count. |
| `pBbess` | integer units | Maximum battery-module installation count. |
| `pBevmx` | kW | Reserved site EV-charging limit; currently inactive. |

### `bdg_ii.csv`: thermal coefficients

This file has one header row and one data row.

| Column | Type/unit | Description |
|---|---|---|
| `pBk1` | dimensionless | Outdoor-to-indoor temperature response coefficient. |
| `pBk2` | deg C/kW | Exterior solar-gain response coefficient. |
| `pBk3` | deg C/kW | Internal heating/cooling response coefficient. |

### `sp.csv`: equipment selection

Each row can select one catalog item in every equipment class. For each prefix
`CHP`, `ABP`, `HVAC`, `WH`, `PV`, `WIND`, and `BESS`, the columns follow this
pattern:

```csv
pCHP0,pCHPz0,pCHPyn,pABP0,pABPz0,pABPyn,pHVAC0,pHVACz0,pHVACyn,pWH0,pWHz0,pWHyn,pPV0,pPVz0,pPVyn,pWIND0,pWINDz0,pWINDyn,pBESS0,pBESSz0,pBESSyn,pEV0,pEVz0,pEVyn
```

| Column pattern | Description |
|---|---|
| `p<PREFIX>0` | Exact `ty` name from the corresponding catalog; use `0` for no selection. |
| `p<PREFIX>z0` | Number of existing units. |
| `p<PREFIX>yn` | `YES` to allow investment in this type, `NO` to limit it to existing units, or `0` for no selection. Investment is forced to `NO` when `pIT=0`. |

EVs use `pEV0` and `pEVz0` in the same way. `pEVyn` must remain in the input
header for schema compatibility, but EV investment is not implemented and its
value is ignored. A trip identifier in `tm.csv` refers to an individual EV, so
two existing vehicles of the same type become two instances.

A selection row whose three fields are all zero is removed. Every selected
name must exist in its catalog and have nonzero capacity.

### `topo.csv`: thermal connections

| Column | Description |
|---|---|
| `up` | Upstream equipment base name, such as a CHP `ty`. |
| `lo` | Downstream equipment base name or the literal `Building`. |
| `in` | Product delivered to the downstream node: `hot air`, `hot water`, or `cooling`. |
| `eff` | Connection efficiency as a fraction. |

The topology currently represents CHP-to-building heating/hot-water links and
CHP-to-absorption-chiller links. Equipment names use catalog base names, without
the numeric suffix DECARB adds for multiple physical units.

### Equipment catalog files

Catalog rows with missing names, duplicate rows, or zero principal capacity
are discarded. Monetary units below use dollars, but a different currency can
be used consistently across every price and cost field.

#### `chp.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique CHP type name. |
| `mx` | kW | Maximum electrical output. |
| `mn` | fraction | Minimum output as a fraction of capacity. |
| `fcf` | fraction | Fuel conversion efficiency in `(0,1]`; fuel input is output divided by this value. |
| `fuel` | code | `G` for gaseous or `L` for liquid fuel. |
| `tank` | kWh | Reserved on-site thermal storage size; currently inactive. |
| `h2p` | ratio | Heat-to-power ratio. |
| `sup` | $/start | Reserved startup cost; currently inactive. |
| `inv` | $/unit | Capital cost. |
| `fom` | $/(unit year) | Fixed operation and maintenance cost. |
| `vom` | $/kWh | Variable operation and maintenance cost. |
| `life` | years | Economic lifetime. |

#### `abp.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique absorption-chiller type name. |
| `mx` | kW | Maximum cooling capacity. |
| `fcf` | fraction | Direct-fuel conversion efficiency in `(0,1]`; use `0` when cooling is supplied only by CHP heat. |
| `fuel` | code | `G` for gaseous or `L` for liquid fuel. |
| `ac` | fraction | Cooling efficiency. |
| `inv` | $/unit | Capital cost. |
| `fom` | $/(unit year) | Fixed operation and maintenance cost. |
| `vom` | $/kWh | Variable operation and maintenance cost. |
| `life` | years | Economic lifetime. |

#### `hvac.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique HVAC type name. |
| `pHVmx`, `pACmx` | kW | Rated heating and cooling capacities. At least one must be nonzero. |
| `pHVeff`, `pACeff` | ratio | Rated heating and cooling efficiencies/COPs. |
| `temp` | deg C | Outdoor design/reference temperature. |
| `pHVmx_`, `pACmx_` | per deg C | Heating and cooling capacity derating coefficients. Values are internally capped at `0.02`. |
| `pHVeff_`, `pACeff_` | per deg C | Heating and cooling efficiency derating coefficients. |
| `inv` | $/unit | Capital cost. |
| `fom` | $/(unit year) | Fixed operation and maintenance cost. |
| `vom` | $/kWh | Variable operation and maintenance cost. |
| `life` | years | Economic lifetime. |

#### `wh.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique water-heater type name. |
| `mx` | kW | Maximum input capacity. |
| `fcf` | fraction | Fuel conversion efficiency in `(0,1]` for fuel-fired units; use `0` when not applicable. |
| `fuel` | code | `0` for electricity, `G` for gaseous fuel, or `L` for liquid fuel. |
| `eff` | fraction | Hot-water conversion efficiency. |
| `tank` | kWh | Thermal tank capacity; use `0` for tankless. |
| `inv` | $/unit | Capital cost. |
| `fom` | $/(unit year) | Fixed operation and maintenance cost. |
| `vom` | $/kWh | Variable operation and maintenance cost. |
| `life` | years | Economic lifetime. |

#### `pv.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique PV module type name. |
| `tech` | `1`, `2`, or `3` | PV performance-coefficient set. |
| `mx` | kW | Rated module capacity. |
| `ar` | m2 | Module area; must be positive. |
| `eff` | fraction | DC-to-AC efficiency. |
| `loss` | fraction | Reserved system loss; currently inactive. |
| `fail` | fraction | Reserved failure rate; currently inactive. |
| `inv` | $/module | Capital cost. |
| `lc` | fraction/year | Reserved learning rate; currently inactive. |
| `fom` | $/(module year) | Fixed operation and maintenance cost. |
| `life` | years | Economic lifetime. |

#### `wind.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique wind-turbine type name. |
| `mx`, `mn` | kW | Maximum and minimum output. |
| `vmn` | m/s | Minimum operating wind speed. |
| `vmx` | m/s | Cutoff wind speed; must exceed `vmn`. |
| `fail` | fraction | Reserved failure rate; currently inactive. |
| `inv` | $/unit | Capital cost. |
| `lc` | fraction/year | Reserved learning rate; currently inactive. |
| `fom` | $/(unit year) | Fixed operation and maintenance cost. |
| `life` | years | Economic lifetime. |

Output is linearly interpolated from `mn` at `vmn` to `mx` at `vmx` and is
zero outside that interval.

#### `bess.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique battery module type name. |
| `mx` | kWh | Energy capacity. |
| `up`, `dn` | kW | Maximum charge and discharge rates. |
| `effu`, `effd` | fraction | Charge and discharge efficiencies. |
| `inv` | $/module | Capital cost. |
| `lc` | fraction/year | Reserved learning rate; currently inactive. |
| `fom` | $/(module year) | Fixed operation and maintenance cost. |
| `life` | years | Economic lifetime. |

#### `ev.csv`

| Column | Type/unit | Description |
|---|---|---|
| `ty` | string | Unique EV type name. |
| `mx` | kWh | Battery capacity. |
| `drv` | kWh/km | Reference driving energy intensity. |
| `cold` | fraction | Reserved cold-weather penalty; currently inactive. Temperature effects are calculated separately from `drv` and `pTout`. |
| `up`, `dn` | kW | Maximum charge and discharge rates. |
| `effu`, `effd` | fraction | Charge and discharge efficiencies. |

## Run-time API inputs and return value

The public API is:

```julia
result = DECARB.run_decarb!(case_path;
    mip_gap=1e-2,
    time_limit=300.0,
    solver=:highs,
    relax_integrality=false,
)
```

| Argument | Description |
|---|---|
| `case_path` | Case directory containing `in/`. |
| `mip_gap` | Solver relative MIP-gap target. |
| `time_limit` | Solver time limit in seconds. |
| `solver` | `:highs` or `:gurobi`; Gurobi must be installed and licensed separately. |
| `relax_integrality` | If true, solves the continuous relaxation. If false, solves the integer model and then fixes integer decisions for a relaxed re-solve that can provide duals. |

The return value is a named tuple containing the solved JuMP `model` and the
loaded/derived dictionaries `cfg`, `tm`, `bdg`, `sp`, `chp`, `abp`, `hvac`,
`wh`, `pv`, `wind`, `bess`, `ev`, and `topo`.

## Output files

A feasible solve writes the following files under `out/`. Dates use
`mm/dd/yyyy HH:MM`. Economic values are rounded to two decimals, emissions to
three decimals, dispatch values to two decimals, and duals to four decimals.

### `eq1.csv`: economics and emissions

This file has columns `name` and `eq`, with one row per metric.

| `name` | Unit | Description |
|---|---|---|
| `total_cost` | $ | Complete objective: energy bill, non-service penalties, annualized equipment cost, and EV driver-preference term. |
| `energy_bill` | $ | Grid purchases minus grid-sale revenue, plus capacity, fuel, variable O&M, and fixed O&M costs. |
| `grid_sales` | $ | Grid-export revenue, reported as a positive amount. |
| `grid_purchases` | $ | Grid-import energy cost. |
| `capacity_charge` | $ | Peak-capacity charges. |
| `fuel_purchases` | $ | Gaseous- and liquid-fuel cost. |
| `variable_om_cost` | $ | Variable operation and maintenance cost. |
| `fixed_om_cost` | $ | Fixed operation and maintenance cost, prorated to the modeled horizon. |
| `equipment_annuity` | $ | Sum of all time-prorated annualized equipment costs. |
| `chp_annuity` | $ | CHP portion of the equipment annuity. |
| `hvac_annuity` | $ | HVAC portion of the equipment annuity. |
| `abp_annuity` | $ | Absorption-chiller portion of the equipment annuity. |
| `water_heater_annuity` | $ | Water-heater portion of the equipment annuity. |
| `pv_annuity` | $ | PV portion of the equipment annuity. |
| `wind_annuity` | $ | Wind portion of the equipment annuity. |
| `battery_annuity` | $ | Battery portion of the equipment annuity. |
| `unserved_electricity_cost` | $ | Electricity non-service penalty. |
| `unserved_thermal_cost` | $ | Indoor-temperature discomfort penalty. |
| `unserved_hot_water_cost` | $ | Hot-water non-service penalty. |
| `unserved_ev_cost` | $ | EV minimum-SOC shortfall penalty. |
| `direct_emissions` | metric ton CO2 | On-site gaseous- and liquid-fuel emissions. |
| `indirect_emissions` | metric ton CO2 | Emissions associated with grid purchases. |

### `eq2.csv`: installed equipment and investment

One row is written for each installed stationary equipment type and investment
window. Existing equipment therefore appears even in a dispatch-only case.
EVs are not included because EV investment is not implemented.

| Column | Unit | Description |
|---|---|---|
| `Eq` | string | Catalog equipment type. Numeric physical-unit suffixes are removed from thermal equipment. |
| `Inv` | $/unit | Catalog capital cost. |
| `Qty` | units | Quantity installed in this investment window, including existing stock in the first window. |
| `New` | units | Newly purchased quantity (`Qty` minus existing stock in the first applicable window). |
| `CAPEX` | $ | Undiscounted capital expenditure, `Inv * New`; this differs from the prorated annuity in `eq1.csv`. |
| `Date` | local date-time | Date at which the existing/new quantity becomes available. |

### `ts.csv`: period results

This file contains one row per input period. Electrical and fuel flows are
reported as average power over the period; thermal service and hot-water flows
are energy per period.

| Columns | Unit | Description |
|---|---|---|
| `Date` | local date-time | Input period-end timestamp. |
| `dem` | kW | Lighting plus equipment demand. |
| `sell`, `buy` | kW | Grid export and import. |
| `hvac_HT`, `hvac_AC` | kW | HVAC electricity used for heating and cooling. |
| `Qwh` | kW | Electricity used by electric water heaters. |
| `bess_UP`, `bess_DN` | kW | Battery charge and discharge power. |
| `ev_UP`, `ev_DN` | kW | EV charge and discharge power. |
| `pv`, `wind`, `chp` | kW | Electrical generation by technology. |
| `nse` | kW | Unserved electrical demand. |
| `bess_SOC`, `ev_SOC` | kWh | Aggregate stored battery and EV energy. |
| `Gchp`, `Gabp`, `Gwh` | kW fuel | Gaseous-fuel input to CHP, absorption chillers, and water heaters. |
| `Lchp`, `Labp`, `Lwh` | kW fuel | Liquid-fuel input to CHP, absorption chillers, and water heaters. |
| `Temp` | deg C | Indoor temperature. |
| `HVACht`, `HVACac` | kWh/period | Space heating and cooling supplied by HVAC. |
| `CHPht` | kWh/period | Space heat supplied by CHP. |
| `ABPac` | kWh/period | Cooling supplied by absorption chillers. |
| `IHGp`, `IHGl`, `IHGe` | kW thermal | Internal heat gain from people, lighting, and equipment. |
| `SHG` | kW thermal | Solar heat gain through the building envelope. |
| `HWdem` | kWh/period | Hot-water demand. |
| `HWns` | kWh/period | Unserved hot-water demand. |
| `WHsoc` | kWh | Aggregate water-heater tank energy. |
| `WHhw` | kWh/period | Hot water supplied by water heaters. |
| `HWchp` | kWh/period | Hot water supplied by CHP. |
| `dualQ` | solver dual | Electrical-balance shadow value. |
| `dualT` | solver dual | Indoor-temperature balance shadow value. |
| `dualHW` | solver dual | Hot-water balance shadow value. |

Dual columns are zero when the selected solve does not provide dual values.
Their sign follows the corresponding JuMP constraint convention.

### Conditional diagnostic outputs

- `balance.csv` is written when any electrical-balance residual calculated
  from the rounded reported flows exceeds `0.1 kW`. It has no header: its two
  columns are `Date` and the residual `B_Q` in kW.
- `iis_constraints.txt` is written when the model is infeasible and the solver
  can compute a conflict. It lists constraints identified as part of the
  irreducible infeasible subsystem; the run then raises an error.

For a complete working input and representative outputs, see
[`examples/minimal`](../examples/minimal).
