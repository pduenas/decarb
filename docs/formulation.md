# Model formulation

This document summarizes the formulation implemented in `src/decarb_model`.
The source code remains authoritative; symbols below are descriptive rather
than a complete algebraic specification.

## Scope and time representation

DECARB is a mixed-integer linear program indexed by time period `t`, equipment
type, physical unit, and—when enabled—investment window. Input timestamps are
period-ending UTC instants; `pP0` is the first period's UTC start. `TM[t]` is
elapsed duration in hours between consecutive UTC boundaries, and `H` is the
total elapsed horizon, equal to `sum(TM)`. Power inputs in kW are period
averages and are multiplied by `TM[t]` when they enter an energy balance.
Demand, weather, and irradiance in a row describe the same interval. Solar
position uses that interval's UTC midpoint at whole-minute precision and site
coordinates as an approximation to period-average geometry. The building
timezone controls local display only and cannot shift solar or demand. See
[the UTC input contract](inputs_outputs.md#tmcsv-time-series) for conversion
rules, including local standard time versus DST-aware civil time.

The model supports:

- grid purchases and sales;
- combined heat and power (CHP);
- absorption chillers, HVAC systems, and water heaters;
- photovoltaic panels and wind turbines;
- battery energy storage systems (BESS); and
- electric-vehicle charging and vehicle-to-grid operation.

Binary variables enforce mutually exclusive operating modes such as grid
buying versus selling, heating versus cooling, and charging versus discharging.
Integer or binary investment variables select new equipment.

## Objective

The model minimizes total cost:

```text
total cost = operating cost
           + unmet-service penalties
           + annualized investment cost
           + EV preference penalty
```

Operating cost includes grid purchases, less grid-sale revenue, fuel,
capacity charges, variable O&M, and fixed O&M. Investment costs use the input
interest rate and asset lifetime to form an annuity (`IR/(1-(1+IR)^-life)`, or
`1/life` when the rate is zero), prorated to the modeled
time horizon. Penalties can be applied to unserved electricity, indoor
temperature deviations, unserved hot water, and EV state-of-charge shortfalls.

Emissions are reported but are not currently part of the objective. Direct
emissions use gaseous and liquid fuel consumption. Indirect emissions use grid
purchases and the time-varying electricity emissions factor.

## Electrical balance

For every period, electrical supply equals electrical demand:

```text
CHP + PV + wind + BESS discharge + EV discharge
+ grid purchases + unserved electricity
= lighting + equipment + HVAC + electric water heating
+ BESS charge + EV charge + grid sales
```

Grid purchases and sales are bounded by the case configuration and cannot
occur simultaneously. Optional peak-demand variables track maximum grid import
within configured tariff periods.

## Building temperature

Indoor temperature follows a first-order data-driven recurrence whose input
coefficients `k1`, `k2`, and `k3` are calibrated over the interval specified by
the required `pBtime` field in `bdg_ii.csv`. For a model period of `TM[t]`
hours, define:

```text
r[t]  = TM[t] / pBtime
k1[t] = 1 - (1 - k1)^r[t]
s[t]  = k1[t] / k1
k2[t] = s[t] k2
k3[t] = s[t] k3
```

The input coefficients `k1`, `k2`, and `k3` are fixed building parameters.
The indexed coefficients `k1[t]`, `k2[t]`, and `k3[t]` depend on `t` only
through the period duration `TM[t]`; they do not vary directly with clock time
or weather. Equal-duration periods therefore use identical coefficients. When
`TM[t]` equals the calibration interval, the indexed coefficients equal the
input coefficients.

When `k1 = 0`, `s[t] = r[t]`. This resampling assumes outdoor temperature,
solar gain, internal gain, and equipment output are constant within each
period. The resulting recurrence is:

```text
Tin[t] = Tin[t-1]
       + k1[t] (Tout[t] - Tin[t-1])
       + k2[t] solar_gain[t]
       + k3[t] (active_heating_cooling_power[t] + internal_gains[t])
```

The first period uses the configured initial indoor temperature. Internal gains
come from occupancy, lighting, and electric equipment. Solar gains are derived
from irradiance, sun position, and building-envelope geometry. All three gain
terms enter as average thermal power in kW; equipment energy produced during a
period is divided by `TM[t]` before entering the recurrence.

Comfort constraints bound indoor temperature when enabled. Nonnegative slack
variables represent excursions above and below the comfort band and are
penalized when `NSTcost` is nonzero.

## Domestic hot water

Water heaters and eligible CHP thermal connections serve hot-water demand.
Tank models carry stored energy between periods and enforce configured initial
and final storage fractions. A penalized slack can represent unserved demand.

## Storage and electric vehicles

BESS state of charge evolves from charging and discharging after their
respective efficiencies. Charge and discharge are mutually exclusive, bounded
by power limits, and linked to installed module counts. Initial and final
state-of-charge fractions are enforced.

EV state of charge additionally accounts for trip energy. Charging is limited
to plugged-in periods, and discharge is disabled unless vehicle-to-grid is
enabled. `vEVtrip[t,e]` represents charging away from home during a trip. It
is bounded in `[0,1]` because it contributes to normalized SOC, and is fixed
to zero outside trips and for inactive EVs.

Away-from-home charging and minimum-SOC shortfalls are penalized by
`NSEVcost`. When `NSEVcost` is zero, both variables are fixed to zero, so
trips and the minimum SOC requirement cannot be softened.

## Investments

Existing assets are fixed in the first investment window. Candidate assets may
be installed in later windows subject to building space, equipment count, and
technology-specific limits. Installed assets remain available in subsequent
periods. Set `pIT = 0` in `cfg.csv` for a dispatch-only case.

## Solving and dual values

DECARB solves the mixed-integer model with HiGHS by default. If the integer
solution is feasible, integer and binary decisions are fixed and the resulting
linear program is solved again. This second solve provides dual values for the
electricity, temperature, and hot-water balances.

## Current limitations

Input validation warns when accepted columns are not active in the current
release. At present these include several lifecycle/failure fields, CHP
supplemental-firing data, the building EV-count limit, and configured carbon
caps or prices. Do not assume a nonzero value in a warned field affects results.
