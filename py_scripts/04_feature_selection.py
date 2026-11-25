import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Зареждане
df = pd.read_csv(
    "data/traffic_preprocessed.csv",
    parse_dates=["date"],
    low_memory=False
)

print(f"Редове  : {len(df):,}")
print(f"Колони  : {list(df.columns)}")
print(f"Период  : {df['date'].min()}  →  {df['date'].max()}")
print(f"\nТипове данни:\n{df.dtypes}")

print("=" * 60)
print("АНАЛИЗ 1: КОРЕЛАЦИОНЕН АНАЛИЗ")
print("=" * 60)

# Корелация между числовите колони
numeric_cols = ["speed", "length", "weight", "vehicle_class"]
corr_matrix = df[numeric_cols].corr(method="pearson")

print("Pearson корелационна матрица:")
print(corr_matrix.round(3).to_string())

# Визуализация
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Pearson
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, ax=ax1, square=True,
            cbar_kws={"label": "Корелация"})
ax1.set_title("Pearson корелация", fontsize=13, fontweight="bold")

# Spearman - по-добър за нелинейни връзки
corr_spearman = df[numeric_cols].corr(method="spearman")
sns.heatmap(corr_spearman, annot=True, fmt=".3f", cmap="coolwarm",
            center=0, ax=ax2, square=True,
            cbar_kws={"label": "Корелация"})
ax2.set_title("Spearman корелация", fontsize=13, fontweight="bold")

plt.tight_layout()
plt.savefig("feature_corr.png", dpi=150, bbox_inches="tight")
plt.show()

# Корелация по vehicle_class
print("\nСредни стойности по vehicle_class:")
print(df.groupby("vehicle_class")[["speed","length","weight"]]
      .mean().round(1).to_string())

print("=" * 60)
print("АНАЛИЗ 2: ВАРИАБИЛНОСТ ПО ВРЕМЕВИ ИНТЕРВАЛ")
print("=" * 60)

intervals = {"15min": "15min", "1h": "1h", "1D": "1D"}

for name, freq in intervals.items():
    agg = df.groupby(df["date"].dt.floor(freq)).size()
    print(f"\nИнтервал {name}:")
    print(f"  Общо интервали       : {len(agg):,}")
    
    # Пълна поредица
    full = pd.date_range(df["date"].min().floor(freq),
                         df["date"].max().floor(freq), freq=freq)
    missing = len(full) - len(agg)
    print(f"  Пълни интервали      : {len(full):,}")
    print(f"  Липсващи интервали   : {missing:,} ({missing/len(full)*100:.1f}%)")
    print(f"  Средно записи/интервал: {agg.mean():.1f}")
    print(f"  Min записи/интервал  : {agg.min()}")
    print(f"  Max записи/интервал  : {agg.max()}")

print("=" * 60)
print("АНАЛИЗ 3: ИНФОРМАЦИОННА СТОЙНОСТ НА КОЛОНИТЕ")
print("=" * 60)

# Агрегираме по час и проверяваме вариабилността
df["hour_ts"] = df["date"].dt.floor("h")

hourly = df.groupby("hour_ts").agg(
    volume      = ("speed", "size"),
    avg_speed   = ("speed", "mean"),
    avg_length  = ("length", "mean"),
    avg_weight  = ("weight", "mean"),
    std_speed   = ("speed", "std"),
    std_length  = ("length", "std")
).reset_index()

print("Вариабилност на агрегираните колони (CV = std/mean):")
for col in ["volume", "avg_speed", "avg_length", "avg_weight"]:
    cv = hourly[col].std() / hourly[col].mean()
    print(f"  {col:15}: mean={hourly[col].mean():.1f}, "
          f"std={hourly[col].std():.1f}, CV={cv:.3f}")

# Корелация между агрегираните колони
print(f"\nКорелация между агрегираните колони:")
agg_cols = ["volume", "avg_speed", "avg_length", "avg_weight"]
print(hourly[agg_cols].corr().round(3).to_string())

# Колко от вариацията в avg_speed се обяснява от volume?
from numpy.polynomial import polynomial as P
corr_vol_speed = hourly[["volume", "avg_speed"]].corr().iloc[0,1]
print(f"\nКорелация volume ↔ avg_speed : {corr_vol_speed:.3f}")
print(f"R²                           : {corr_vol_speed**2:.3f}")

print("=" * 60)
print("АНАЛИЗ 4: VEHICLE CLASS РАЗБИВКА")
print("=" * 60)

# Дял на всеки клас по час
df["hour_ts"] = df["date"].dt.floor("h")

vc_hourly = (df.groupby(["hour_ts", "vehicle_class"])
               .size()
               .reset_index(name="count"))

vc_pivot = (vc_hourly.pivot(index="hour_ts", 
                             columns="vehicle_class", 
                             values="count")
                      .fillna(0))

# Корелация между класовете
print("Корелация между броя ПС по клас (почасово):")
print(vc_pivot.corr().round(3).to_string())

# CV за всеки клас
print(f"\nВариабилност по клас (CV):")
for col in vc_pivot.columns:
    cv = vc_pivot[col].std() / vc_pivot[col].mean()
    pct = df[df["vehicle_class"]==col].shape[0] / len(df) * 100
    print(f"  Клас {col:3}: CV={cv:.3f}, дял={pct:.2f}%")

# Корелация с общия volume
vc_pivot["total"] = vc_pivot.sum(axis=1)
print(f"\nКорелация на всеки клас с общия volume:")
for col in [c for c in vc_pivot.columns if c != "total"]:
    corr = vc_pivot[col].corr(vc_pivot["total"])
    print(f"  Клас {col:3}: {corr:.3f}")

print("=" * 60)
print("АНАЛИЗ 5: АНАЛИЗ ПО ПОСОКА")
print("=" * 60)

# Агрегиране по час и посока
hourly_dir = (df.groupby([df["date"].dt.floor("h"), "direction"])
               .size()
               .reset_index(name="count"))

hourly_pivot = (hourly_dir.pivot(index="date", 
                                  columns="direction", 
                                  values="count")
                           .fillna(0))

hourly_pivot.columns = ["Okolovrastno", "Centar"]

# Корелация между посоките
corr_dir = hourly_pivot["Centar"].corr(hourly_pivot["Okolovrastno"])
print(f"Корелация Център ↔ Околовръстно: {corr_dir:.3f}")

# CV по посока
for col, name in [("Centar", "Център"), ("Okolovrastno", "Околовръстно")]:
    cv = hourly_pivot[col].std() / hourly_pivot[col].mean()
    print(f"\n{name}:")
    print(f"  mean={hourly_pivot[col].mean():.1f}, "
          f"std={hourly_pivot[col].std():.1f}, CV={cv:.3f}")

# Среден hourly profile по посока
df["hour"] = df["date"].dt.hour
hourly_profile = (df.groupby(["hour", "direction"])
                    .size()
                    .reset_index(name="count"))

fig, ax = plt.subplots(figsize=(12, 5))
for direction in ["Център", "Околовръстно"]:
    sub = hourly_profile[hourly_profile["direction"] == direction]
    ax.plot(sub["hour"], sub["count"], 
            marker="o", linewidth=2, label=direction)

ax.set_title("Почасов профил по посока", fontsize=13, fontweight="bold")
ax.set_xlabel("Час")
ax.set_ylabel("Брой записи")
ax.set_xticks(range(0, 24))
ax.legend()
plt.tight_layout()
plt.savefig("feature_direction_profile.png", dpi=150, bbox_inches="tight")
plt.show()

print("=" * 60)
print("ФИНАЛНО РЕШЕНИЕ — FEATURE SELECTION")
print("=" * 60)

print("""
КОЛОНИ ЗА ОТПАДАНЕ:
  ✗ plate        — чувствителни данни, отпада при агрегиране
  ✗ country      — 40% null, не е релевантна за трафик анализ
  ✗ weight       — системна повреда на сензора (L1, L4, L5)

КОЛОНИ ЗА ЗАПАЗВАНЕ:
  ✓ date         — задължителна за времевия ред
  ✓ direction    — за агрегиране по посока
  ✓ lane         — за агрегиране по лента
  ✓ vehicle_class— за разбивка по тип ПС
  ✓ speed        — за avg_speed агрегат
  ✓ length       — за avg_length агрегат

АГРЕГИРАНИ ХАРАКТЕРИСТИКИ:
  ✓ volume       — брой преминали ПС (най-важна!)
  ✓ avg_speed    — средна скорост (CV=0.074, ниска но реална)
  ✓ avg_length   — средна дължина (CV=0.163)

ВРЕМЕВИ ИНТЕРВАЛ: 1 час
АГРЕГИРАНЕ ПО:    посока (Център/Околовръстно)
""")

print("=" * 60)
print("ИНФОРМАЦИОННА СТОЙНОСТ НА avg_length")
print("=" * 60)

# Как се променя avg_length по час от денонощието?
df["hour"] = df["date"].dt.hour
length_by_hour = df.groupby("hour")["length"].mean()
print("Средна дължина по час:")
print(length_by_hour.round(0).to_string())

# Как се променя по ден от седмицата?
df["dow"] = df["date"].dt.dayofweek
length_by_dow = df.groupby("dow")["length"].mean()
dow_labels = ["Пон","Вт","Ср","Чет","Пет","Съб","Нед"]
print("\nСредна дължина по ден от седмицата:")
for i, val in enumerate(length_by_dow):
    print(f"  {dow_labels[i]}: {val:.0f} мм")

# Корелация avg_length с volume по час
hourly = df.groupby(df["date"].dt.floor("h")).agg(
    volume     = ("speed", "size"),
    avg_length = ("length", "mean"),
    avg_speed  = ("speed", "mean")
).reset_index()

print(f"\nКорелация avg_length ↔ volume    : {hourly['avg_length'].corr(hourly['volume']):.3f}")
print(f"Корелация avg_length ↔ avg_speed : {hourly['avg_length'].corr(hourly['avg_speed']):.3f}")

# Вариацията на avg_length обяснена ли е от vehicle_class mix?
vc_hourly = df.groupby(df["date"].dt.floor("h")).apply(
    lambda x: (x["vehicle_class"] == 1).mean()
).reset_index(name="pct_class1")

merged = hourly.merge(vc_hourly, on="date")
print(f"Корелация avg_length ↔ % клас 1  : {merged['avg_length'].corr(merged['pct_class1']):.3f}")

