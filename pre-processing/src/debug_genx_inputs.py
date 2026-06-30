import pandas as pd


thermal_path = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\GenX.jl-main\example_systems\ercot_1bus_ed\resources\Thermal.csv"
vre_path = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\GenX.jl-main\example_systems\ercot_1bus_ed\resources\Vre.csv"
demand_path = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\GenX.jl-main\example_systems\ercot_1bus_ed\system\Demand_data.csv"
fuels_path = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\GenX.jl-main\example_systems\ercot_1bus_ed\system\Fuels_data.csv"
gen_var_path = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\GenX.jl-main\example_systems\ercot_1bus_ed\system\Generators_variability.csv"
power_path = r"C:\Users\onurt\OneDrive - Massachusetts Institute of Technology\Documents\research\GenX.jl-main\example_systems\ercot_1bus_ed\results\power.csv"

thermal = pd.read_csv(thermal_path)
print("=== Thermal ===")
print(f"Total Existing_Cap_MW: {thermal['Existing_Cap_MW'].sum():.0f} MW")
print(f"Min_Power range: {thermal['Min_Power'].min():.3f} to {thermal['Min_Power'].max():.3f}")
print(f"Any Min_Power > 1: {(thermal['Min_Power'] > 1).sum()}")
print(f"Any Existing_Cap_MW = 0: {(thermal['Existing_Cap_MW'] == 0).sum()}")

vre = pd.read_csv(vre_path)
print("\n=== VRE ===")
print(f"Total Existing_Cap_MW: {vre['Existing_Cap_MW'].sum():.0f} MW")


demand = pd.read_csv(demand_path)
print("\n=== Demand ===")
print(f"Columns: {list(demand.columns)}")
print(f"Max load: {demand.iloc[:,0].max():.0f} MW")
print(demand.head(3))

fuels = pd.read_csv(fuels_path)
print("\n=== Fuels ===")
print(fuels.head(3))

print(thermal[["Resource","Existing_Cap_MW","Cap_Size","Ramp_Up_Percentage"]].describe())

var = pd.read_csv(gen_var_path, index_col=0)
print(f"Shape: {var.shape}")
print(f"Columns (first 5): {list(var.columns[:5])}")
print(f"Any values > 1: {(var > 1).any().any()}")
print(f"Mean CF across all VRE: {var.mean().mean():.3f}")
print(var.describe())

var_cols = set(var.columns)
vre_resources = set(vre["Resource"])
print(f"In variability but not VRE: {var_cols - vre_resources}")
print(f"In VRE but not variability: {vre_resources - var_cols}")


power = pd.read_csv(power_path)
# Skip Zone and AnnualSum rows, get numeric data rows
data_rows = power[power["Resource"].str.startswith("t")]
total_gen_per_hour = data_rows.iloc[:, 1:].apply(pd.to_numeric, errors="coerce").sum(axis=1)
print(f"Mean hourly generation: {total_gen_per_hour.mean():.0f} MW")
print(f"Max hourly generation: {total_gen_per_hour.max():.0f} MW")
print(f"Min hourly generation: {total_gen_per_hour.min():.0f} MW")
print(f"\nMean hourly demand: {demand['Demand_MW_z1'].mean():.0f} MW")
print(f"Max hourly demand: {demand['Demand_MW_z1'].max():.0f} MW")