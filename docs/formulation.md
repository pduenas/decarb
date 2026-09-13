# Model formulation

This document summarizes the formulation implemented in `src/decarb_model`.
The source code remains authoritative; symbols below are descriptive rather
than a complete algebraic specification.

## Scope and time representation

DECARB is a mixed-integer linear program indexed by time period `t`, equipment
type, physical unit, and—when enabled—investment window. Input timestamps are
period-ending local datetimes. `TM[t]` is the period duration in hours, so power
inputs in kW are multiplied by `TM[t]` when they enter an energy balance.

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
interest rate and asset lifetime to form an annuity, prorated to the modeled
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

Indoor temperature follows a first-order data-driven recurrence:

```text
Tin[t] = Tin[t-1]
       + k1 (Tout[t-1] - Tin[t-1])
       + k2 solar_gain[t]
       + k3 (active_heating_cooling[t] + internal_gains[t])
```

The first period uses the configured initial indoor temperature. Internal gains
come from occupancy, lighting, and electric equipment. Solar gains are derived
from irradiance, sun position, and building-envelope geometry.

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
enabled. A slack variable permits a penalized shortfall below the requested
minimum state of charge.

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
