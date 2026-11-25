import pandas as pd


files = [
    "data/2022_combined.csv",
    "data/2023_combined.csv",
    "data/2024_combined.csv"
]


READ_PARAMS = {
    "sep": ",",           
    "encoding": "utf-8",  
    "low_memory": False   
}


dfs = []

for f in files:
    print(f"Четене на: {f}")
    df = pd.read_csv(f, **READ_PARAMS)
    print(f"  -> {len(df):,} реда, {len(df.columns)} колони")
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

print(f"\nОбщо редове след комбиниране: {len(combined):,}")
print(f"Колони: {list(combined.columns)}")


combined["date"] = pd.to_datetime(combined["date"])
combined = combined.sort_values("date").reset_index(drop=True)

print(f"\nПърва дата: {combined['date'].min()}")
print(f"Последна дата: {combined['date'].max()}")


output_path = "data/traffic_combined.csv"
combined.to_csv(output_path, index=False, encoding="utf-8")

print(f"\nФайлът е запазен: {output_path}")

