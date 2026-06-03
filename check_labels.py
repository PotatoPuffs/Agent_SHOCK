import pandas as pd
df = pd.read_csv("data/labels.csv")

print("=== CROSSHAIR ===")
print(df[["cx_norm", "cy_norm"]].describe())

print("\n=== TARGET ===")
print(df[["tx_norm", "ty_norm"]].describe())

print("\n=== TARGET VARIETY (unique values) ===")
print(f"Unique tx_norm values: {df['tx_norm'].nunique()}")
print(f"Unique ty_norm values: {df['ty_norm'].nunique()}")
print(f"\nTarget x range: {df['tx_norm'].min():.3f} to {df['tx_norm'].max():.3f}")
print(f"Target y range: {df['ty_norm'].min():.3f} to {df['ty_norm'].max():.3f}")