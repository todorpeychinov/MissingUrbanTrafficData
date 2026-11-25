"""
Закръгляне на импутираните стойности:
  - volume     -> закръглено до цяло число (само за редовете, попълнени с RF/GAN)
  - avg_speed  -> закръглено до 2-ри знак след десетичната запетая (само за импутираните)

Реалните (неимпутирани) стойности НЕ се пипат - остават както са в изходния файл.

Вход:  data/traffic_fully_imputed.csv  (изход на 13_speed_imputation.ipynb)
Изход: data/traffic_fully_imputed_rounded.csv
"""

import pandas as pd

INPUT_PATH  = "data/traffic_fully_imputed.csv"
OUTPUT_PATH = "data/traffic_fully_imputed_rounded.csv"

df = pd.read_csv(INPUT_PATH, parse_dates=["datetime"])

# Маски за импутираните редове
volume_imputed_mask = df["imputation_method"].notna()
speed_imputed_mask  = df["speed_imputation_method"].notna()

print(f"Импутирани volume редове    : {volume_imputed_mask.sum():,}")
print(f"Импутирани avg_speed редове : {speed_imputed_mask.sum():,}")

# Закръгляне
df.loc[volume_imputed_mask, "volume"] = (
    df.loc[volume_imputed_mask, "volume"].round(0).astype(int)
)

df.loc[speed_imputed_mask, "avg_speed"] = (
    df.loc[speed_imputed_mask, "avg_speed"].round(2)
)

df.to_csv(OUTPUT_PATH, index=False)

print(f"\nЗапазено в {OUTPUT_PATH}")
print(f"Пример - импутирани volume (първите 5):")
print(df.loc[volume_imputed_mask, "volume"].head())
print(f"\nПример - импутирани avg_speed (първите 5):")
print(df.loc[speed_imputed_mask, "avg_speed"].head())

