import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Зареждане
df = pd.read_csv(
    "data/traffic_combined.csv",
    parse_dates=["date"],
    low_memory=False
)

print(f"Редове  : {len(df):,}")
print(f"Колони  : {list(df.columns)}")
print(f"Период  : {df['date'].min()}  →  {df['date'].max()}")

print("=" * 60)
print("СТЪПКА 1: ДУБЛИРАНИ ЗАПИСИ")
print("=" * 60)

# Пълни дубликати - всички колони съвпадат
full_duplicates = df.duplicated().sum()
print(f"Пълни дубликати (всички колони): {full_duplicates:,}")

# Дубликати по ключови колони - същото ПС в същия момент
key_duplicates = df.duplicated(subset=["date", "plate", "lane"]).sum()
print(f"Дубликати по date+plate+lane   : {key_duplicates:,}")

# Дубликати само по дата и лента (без plate)
date_lane_duplicates = df.duplicated(subset=["date", "lane"]).sum()
print(f"Дубликати по date+lane         : {date_lane_duplicates:,}")

# Примери на дубликати ако има такива
if full_duplicates > 0:
    print(f"\nПримери на пълни дубликати:")
    print(df[df.duplicated(keep=False)].head(10).to_string())
elif key_duplicates > 0:
    print(f"\nПримери на date+plate+lane дубликати:")
    print(df[df.duplicated(subset=["date","plate","lane"], keep=False)].head(10).to_string())

print("=" * 60)
print("ПРЕМАХВАНЕ НА ДУБЛИКАТИ")
print("=" * 60)

df_clean = df.drop_duplicates().reset_index(drop=True)

print(f"Редове преди : {len(df):,}")
print(f"Редове след  : {len(df_clean):,}")
print(f"Премахнати   : {len(df) - len(df_clean):,} ({(len(df) - len(df_clean))/len(df)*100:.2f}%)")

# Проверка дали са останали дубликати
print(f"\nОстанали пълни дубликати: {df_clean.duplicated().sum():,}")
print(f"Останали date+plate+lane дубликати: {df_clean.duplicated(subset=['date','plate','lane']).sum():,}")

print("=" * 60)
print("АНАЛИЗ НА ОСТНАЛИТЕ ЧАСТИЧНИ ДУБЛИКАТИ")
print("=" * 60)

partial_dups = df_clean[df_clean.duplicated(subset=["date","plate","lane"], keep=False)]
print(f"Записи в частични дубликати: {len(partial_dups):,}")
print(f"\nПримери:")
print(partial_dups.sort_values(["date","plate","lane"]).head(20).to_string())

# Каква е разликата между тях?
print(f"\nРазлики в стойностите при частичните дубликати:")
for col in ["speed", "length", "weight", "direction", "country"]:
    nunique = (partial_dups.groupby(["date","plate","lane"])[col]
               .nunique()
               .gt(1)
               .sum())
    print(f"  {col}: {nunique} групи с различни стойности")

print("=" * 60)
print("РЕШЕНИЕ ЗА ЧАСТИЧНИТЕ ДУБЛИКАТИ")
print("=" * 60)

# Проверка в кои ленти са концентрирани
print("Разпределение по лента:")
print(partial_dups["lane"].value_counts().to_string())

print(f"\nЧастичните дубликати са реални различни преминавания")
print(f"записани в същата секунда -> запазваме ги")
print(f"\nТекущ брой редове: {len(df_clean):,}")

print("=" * 60)
print("СТЪПКА 2: ФИЛТРИРАНЕ НА АНОМАЛИИ")
print("=" * 60)

# Преди филтриране
print(f"Редове преди филтриране: {len(df_clean):,}")

# --- SPEED ---
speed_zero    = (df_clean["speed"] == 0).sum()
speed_max     = (df_clean["speed"] == 255).sum()
print(f"\nSpeed = 0   : {speed_zero:,} ({speed_zero/len(df_clean)*100:.2f}%)")
print(f"Speed = 255 : {speed_max:,} ({speed_max/len(df_clean)*100:.2f}%)")

# --- LENGTH ---
length_invalid = (df_clean["length"] <= 0).sum()
length_max     = (df_clean["length"] >= 30000).sum()
print(f"\nLength <= 0      : {length_invalid:,} ({length_invalid/len(df_clean)*100:.2f}%)")
print(f"Length >= 30,000 : {length_max:,} ({length_max/len(df_clean)*100:.2f}%)")

# --- WEIGHT ---
weight_invalid = (df_clean["weight"] <= 0).sum()
weight_max     = (df_clean["weight"] >= 50000).sum()
print(f"\nWeight <= 0      : {weight_invalid:,} ({weight_invalid/len(df_clean)*100:.2f}%)")
print(f"Weight >= 50,000 : {weight_max:,} ({weight_max/len(df_clean)*100:.2f}%)")

# Общо засегнати записи (union)
anomaly_mask = (
    (df_clean["speed"] == 0) | (df_clean["speed"] == 255) |
    (df_clean["length"] <= 0) | (df_clean["length"] >= 30000) |
    (df_clean["weight"] <= 0) | (df_clean["weight"] >= 50000)
)
print(f"\nОбщо записи с поне 1 аномалия: {anomaly_mask.sum():,} ({anomaly_mask.sum()/len(df_clean)*100:.2f}%)")

print("=" * 60)
print("ИЗСЛЕДВАНЕ НА WEIGHT <= 0")
print("=" * 60)

w_zero = df_clean[df_clean["weight"] <= 0]

print(f"Записи с weight <= 0: {len(w_zero):,}")

print(f"\nРазпределение по vehicle_class:")
print(w_zero["vehicle_class"].value_counts().head(10).to_string())

print(f"\nРазпределение по lane:")
print(w_zero["lane"].value_counts().to_string())

print(f"\nСтатистика за speed при weight <= 0:")
print(w_zero["speed"].describe().round(1).to_string())

print(f"\nСтатистика за length при weight <= 0:")
print(w_zero["length"].describe().round(1).to_string())

# Дали weight=0 корелира с определени години?
print(f"\nРазпределение по година:")
print(w_zero["date"].dt.year.value_counts().sort_index().to_string())

# Процент weight=0 по година
for yr in [2022, 2023, 2024]:
    yr_mask = df_clean["date"].dt.year == yr
    pct = (df_clean[yr_mask]["weight"] <= 0).mean() * 100
    print(f"  {yr}: {pct:.1f}% от записите имат weight <= 0")

print("=" * 60)
print("WEIGHT=0 ПО ЛЕНТА И ГОДИНА")
print("=" * 60)

pivot = (df_clean.assign(year=df_clean["date"].dt.year,
                          weight_zero=(df_clean["weight"] <= 0).astype(int))
         .groupby(["year", "lane"])["weight_zero"]
         .mean()
         .mul(100)
         .round(1)
         .reset_index()
         .pivot(index="lane", columns="year", values="weight_zero"))

print("% записи с weight <= 0 по лента и година:")
print(pivot.to_string())

print("=" * 60)
print("ФИЛТРИРАНЕ НА АНОМАЛИИ - SPEED И LENGTH")
print("=" * 60)

# Преди филтриране
print(f"Редове преди: {len(df_clean):,}")

# --- SPEED ---
# Премахваме speed=0 (спрели/грешка) и speed=255 (sensor overflow)
speed_mask = (df_clean["speed"] > 0) & (df_clean["speed"] < 255)

# --- LENGTH ---
# Премахваме отрицателни/нула и над 30,000мм (30м)
# Реалистичен диапазон: най-малкият мотоциклет ~1000мм, най-дългото МПС ~25,000мм
length_mask = (df_clean["length"] > 0) & (df_clean["length"] < 30000)

# Комбиниран филтър
combined_mask = speed_mask & length_mask
df_clean = df_clean[combined_mask].reset_index(drop=True)

print(f"Редове след : {len(df_clean):,}")
print(f"Премахнати  : {combined_mask.sum() - len(df_clean) + (~combined_mask).sum():,} "
      f"({(~combined_mask).sum()/combined_mask.size*100:.2f}%)")

# Верификация
print(f"\nВерификация след филтриране:")
print(f"Speed min/max   : {df_clean['speed'].min()} / {df_clean['speed'].max()}")
print(f"Length min/max  : {df_clean['length'].min()} / {df_clean['length'].max()}")
print(f"Speed = 0       : {(df_clean['speed'] == 0).sum():,}")
print(f"Speed = 255     : {(df_clean['speed'] == 255).sum():,}")
print(f"Length <= 0     : {(df_clean['length'] <= 0).sum():,}")
print(f"Length >= 30000 : {(df_clean['length'] >= 30000).sum():,}")

print("=" * 60)
print("СТЪПКА 3: НОРМАЛИЗИРАНЕ НА COUNTRY")
print("=" * 60)

print(f"Преди нормализиране:")
print(f"  BG  : {(df_clean['country'] == 'BG').sum():,}")
print(f"  BGR : {(df_clean['country'] == 'BGR').sum():,}")
print(f"  '-' : {(df_clean['country'] == '-').sum():,}")
print(f"  NaN : {df_clean['country'].isnull().sum():,}")

# BGR -> BG
df_clean['country'] = df_clean['country'].replace('BGR', 'BG')

# '-' -> NaN
df_clean['country'] = df_clean['country'].replace('-', np.nan)

print(f"\nСлед нормализиране:")
print(f"  BG  : {(df_clean['country'] == 'BG').sum():,}")
print(f"  BGR : {(df_clean['country'] == 'BGR').sum():,}")
print(f"  '-' : {(df_clean['country'] == '-').sum():,}")
print(f"  NaN : {df_clean['country'].isnull().sum():,}")

print(f"\nТоп 10 държави след нормализиране:")
print(df_clean['country'].value_counts().head(10).to_string())

print("=" * 60)
print("СТЪПКА 4: ИЗСЛЕДВАНЕ НА МАРТ 2023")
print("=" * 60)

mar2023 = df_clean[
    (df_clean["date"].dt.year == 2023) & 
    (df_clean["date"].dt.month == 3)
]

print(f"Записи в Март 2023: {len(mar2023):,}")

# Сравнение с останалите месеци
monthly = (df_clean.groupby([df_clean["date"].dt.year.rename("year"), 
                              df_clean["date"].dt.month.rename("month")])
           .size()
           .reset_index(name="count"))

print(f"\nСредно записи/месец (без Март 2023): "
      f"{monthly[~((monthly['year']==2023) & (monthly['month']==3))]['count'].mean():,.0f}")
print(f"Записи Март 2023                    : {len(mar2023):,}")

# Дневна разбивка на Март 2023
print(f"\nДневни записи в Март 2023:")
daily_mar = mar2023.groupby(mar2023["date"].dt.day).size()
print(daily_mar.to_string())

print("=" * 60)
print("ОБОБЩЕНИЕ НА ПРЕДОБРАБОТКАТА")
print("=" * 60)

print(f"Оригинални записи        : 12,239,528")
print(f"След премахване дубликати: 11,497,502  (-742,026)")
print(f"След филтриране аномалии : 11,271,815  (-225,687)")
print(f"Текущи записи            : {len(df_clean):,}")
print(f"\nПремахнати общо          : {12239528 - len(df_clean):,} "
      f"({(12239528 - len(df_clean))/12239528*100:.1f}%)")

print(f"\nСтатус на колоните:")
print(f"  date          : чиста")
print(f"  direction     : 437,569 null (LN ленти)")
print(f"  plate         : null стойности (ще отпадне при агрегиране)")
print(f"  country       : null стойности (ще отпадне при агрегиране)")
print(f"  speed         : филтрирана (1-254 km/h)")
print(f"  vehicle_class : много рядко срещани класове")
print(f"  lane          : чиста")
print(f"  length        : филтрирана (1-29,999 мм)")
print(f"  weight        : системна повреда на сензора -> feature selection")

print("=" * 60)
print("АНАЛИЗ НА OUTLIER-И")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Boxplot за outlier анализ", fontsize=14, fontweight="bold")

for ax, col in zip(axes, ["speed", "length", "weight"]):
    ax.boxplot(df_clean[col], showfliers=True, 
               flierprops=dict(marker=".", markersize=1, alpha=0.3))
    ax.set_title(col)
    ax.set_ylabel(col)

plt.tight_layout()
plt.savefig("outlier_boxplot.png", dpi=150, bbox_inches="tight")
plt.show()

# IQR метод
print("\nIQR анализ:")
for col in ["speed", "length", "weight"]:
    Q1  = df_clean[col].quantile(0.25)
    Q3  = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    outliers = ((df_clean[col] < lower) | (df_clean[col] > upper)).sum()
    print(f"\n  {col}:")
    print(f"    Q1={Q1:.1f}, Q3={Q3:.1f}, IQR={IQR:.1f}")
    print(f"    Граници: [{lower:.1f}, {upper:.1f}]")
    print(f"    Outlier-и: {outliers:,} ({outliers/len(df_clean)*100:.2f}%)")

print("=" * 60)
print("КАКВО СА OUTLIER-ИТЕ")
print("=" * 60)

# SPEED outliers
speed_low  = df_clean[df_clean["speed"] < 25.5]
speed_high = df_clean[df_clean["speed"] > 69.5]

print(f"SPEED < 25.5 km/h : {len(speed_low):,}")
print(f"SPEED > 69.5 km/h : {len(speed_high):,}")
print(f"\nРазпределение на speed > 69.5 по vehicle_class:")
print(speed_high["vehicle_class"].value_counts().head(8).to_string())
print(f"\nРазпределение на speed < 25.5 по vehicle_class:")
print(speed_low["vehicle_class"].value_counts().head(8).to_string())

# LENGTH outliers
length_low  = df_clean[df_clean["length"] < 1200]
length_high = df_clean[df_clean["length"] > 6800]

print(f"\nLENGTH < 1200 мм (< 1.2м) : {len(length_low):,}")
print(f"LENGTH > 6800 мм (> 6.8м)  : {len(length_high):,}")
print(f"\nРазпределение на length > 6800 по vehicle_class:")
print(length_high["vehicle_class"].value_counts().head(8).to_string())
print(f"\nРазпределение на length < 1200 по vehicle_class:")
print(length_low["vehicle_class"].value_counts().head(8).to_string())

# WEIGHT outliers (само горни - долните са отрицателни вече филтрирани)
weight_high = df_clean[df_clean["weight"] > 4709.5]
print(f"\nWEIGHT > 4709 кг : {len(weight_high):,}")
print(f"Разпределение по vehicle_class:")
print(weight_high["vehicle_class"].value_counts().head(8).to_string())

print("=" * 60)
print("VEHICLE_CLASS — РЕДКИ КЛАСОВЕ")
print("=" * 60)

vc_counts = df_clean["vehicle_class"].value_counts().sort_values(ascending=False)
print("Всички класове:")
print(vc_counts.to_string())

print(f"\nКласове с под 1000 записа:")
rare = vc_counts[vc_counts < 1000]
print(rare.to_string())
print(f"\nОбщо редки класове : {len(rare)}")
print(f"Общо записи в тях  : {rare.sum():,} ({rare.sum()/len(df_clean)*100:.3f}%)")

print("=" * 60)
print("ГРУПИРАНЕ НА РЕДКИ КЛАСОВЕ")
print("=" * 60)

# Класове с под 1000 записа -> 99
rare_classes = vc_counts[vc_counts < 1000].index.tolist()
df_clean["vehicle_class"] = df_clean["vehicle_class"].replace(
    {cls: 99 for cls in rare_classes}
)

print(f"Групирани класове: {rare_classes}")
print(f"\nРезултат след групиране:")
print(df_clean["vehicle_class"].value_counts().sort_values(ascending=False).to_string())

print("=" * 60)
print("ПРОВЕРКА НА WEIGHT")
print("=" * 60)

print(f"Weight <= 0      : {(df_clean['weight'] <= 0).sum():,} ({(df_clean['weight'] <= 0).mean()*100:.2f}%)")
print(f"Weight < 0       : {(df_clean['weight'] < 0).sum():,}")
print(f"Weight = 0       : {(df_clean['weight'] == 0).sum():,}")
print(f"Weight >= 50,000 : {(df_clean['weight'] >= 50000).sum():,}")
print(f"\nMin weight: {df_clean['weight'].min():,}")
print(f"Max weight: {df_clean['weight'].max():,}")
print(f"\nОписателна статистика:")
print(df_clean["weight"].describe().round(1).to_string())

print("=" * 60)
print("ФИЛТРИРАНЕ НА WEIGHT АНОМАЛИИ")
print("=" * 60)

print(f"Редове преди: {len(df_clean):,}")

# Филтрираме само явно невъзможните стойности
# weight=0 НЕ филтрираме — вече знаем че е системна повреда на сензора
# weight < 0 — физически невъзможно -> премахваме
# weight > 50,000 кг (50 тона) — над максималното допустимо за пътищата -> премахваме

weight_mask = (df_clean["weight"] >= 0) & (df_clean["weight"] < 50000)
df_clean = df_clean[weight_mask].reset_index(drop=True)

print(f"Редове след : {len(df_clean):,}")
print(f"Премахнати  : {weight_mask.size - weight_mask.sum():,} ({(~weight_mask).mean()*100:.2f}%)")

print(f"\nВерификация:")
print(f"Weight < 0       : {(df_clean['weight'] < 0).sum():,}")
print(f"Weight >= 50,000 : {(df_clean['weight'] >= 50000).sum():,}")
print(f"Min weight       : {df_clean['weight'].min():,}")
print(f"Max weight       : {df_clean['weight'].max():,}")

print("=" * 60)
print("LN ЛЕНТИ — ФИНАЛНО РЕШЕНИЕ")
print("=" * 60)

ln_mask = df_clean["lane"].str.startswith("LN")
print(f"LN записи общо: {ln_mask.sum():,} ({ln_mask.mean()*100:.2f}%)")
print(f"\nРазпределение по лента:")
print(df_clean[ln_mask]["lane"].value_counts().to_string())

print(f"\nСтатистика за LN записи:")
for col in ["speed", "length", "weight"]:
    print(f"  {col}: mean={df_clean[ln_mask][col].mean():.1f}, "
          f"std={df_clean[ln_mask][col].std():.1f}, "
          f"min={df_clean[ln_mask][col].min()}, "
          f"max={df_clean[ln_mask][col].max()}")

print(f"\nVehicle class при LN:")
print(df_clean[ln_mask]["vehicle_class"].value_counts().to_string())

print(f"\nDirection при LN:")
print(df_clean[ln_mask]["direction"].value_counts(dropna=False).to_string())

print("=" * 60)
print("ДЕТАЙЛЕН АНАЛИЗ НА LN ЛЕНТИ")
print("=" * 60)

# Кога са се появили LN записите - по година и месец
print("LN записи по година:")
print(df_clean[ln_mask]["date"].dt.year.value_counts().sort_index().to_string())

print("\nLN записи по месец (всички години):")
print(df_clean[ln_mask]["date"].dt.month.value_counts().sort_index().to_string())

# Дали LN и L записите се появяват едновременно?
print("\nПример - LN2 и L2 в един и същи час:")
df_clean["hour_ts"] = df_clean["date"].dt.floor("h")
ln2_hours = set(df_clean[df_clean["lane"] == "LN2"]["hour_ts"])
l2_hours  = set(df_clean[df_clean["lane"] == "L2"]["hour_ts"])
overlap   = ln2_hours & l2_hours
print(f"  Часове с LN2 записи : {len(ln2_hours):,}")
print(f"  Часове с L2 записи  : {len(l2_hours):,}")
print(f"  Часове с И ДВЕТЕ    : {len(overlap):,}")
print(f"  Overlap %           : {len(overlap)/len(ln2_hours)*100:.1f}%")

# Дали LN се появява само в определени периоди?
print("\nПърви и последен LN запис:")
print(f"  Първи : {df_clean[ln_mask]['date'].min()}")
print(f"  Последен: {df_clean[ln_mask]['date'].max()}")

# Дали има периоди само с LN без L?
print("\nЧасове САМО с LN (без нито един L запис):")
l_hours  = set(df_clean[~ln_mask]["hour_ts"])
only_ln  = ln2_hours - l_hours
print(f"  {len(only_ln):,} часа само с LN2 записи")

print("=" * 60)
print("ПРЕИМЕНУВАНЕ НА LN ЛЕНТИ")
print("=" * 60)

lane_mapping = {
    "LN2": "L2",
    "LN5": "L5",
    "LN3": "L3",
    "LN6": "L6"
}

df_clean["lane"] = df_clean["lane"].replace(lane_mapping)

print("След преименуване:")
print(df_clean["lane"].value_counts().sort_index().to_string())

# Премахване на временната колона
df_clean = df_clean.drop(columns=["hour_ts"])

print(f"\nNull в direction след преименуване: {df_clean['direction'].isnull().sum():,}")

print("=" * 60)
print("ПОПЪЛВАНЕ НА DIRECTION ПО LANE")
print("=" * 60)

lane_to_direction = {
    "L1": "Център",      "L2": "Център",      "L3": "Център",
    "L4": "Околовръстно", "L5": "Околовръстно", "L6": "Околовръстно"
}

# Попълване само на null стойностите
null_dir_before = df_clean["direction"].isnull().sum()

df_clean["direction"] = df_clean.apply(
    lambda row: lane_to_direction.get(row["lane"], row["direction"]) 
    if pd.isnull(row["direction"]) else row["direction"],
    axis=1
)

null_dir_after = df_clean["direction"].isnull().sum()

print(f"Null преди : {null_dir_before:,}")
print(f"Null след  : {null_dir_after:,}")
print(f"Попълнени  : {null_dir_before - null_dir_after:,}")

print(f"\nРазпределение на direction след попълване:")
print(df_clean["direction"].value_counts(dropna=False).to_string())

print("=" * 60)
print("ПРОВЕРКА НА DATE")
print("=" * 60)

print(f"Период      : {df_clean['date'].min()}  →  {df_clean['date'].max()}")
print(f"Null в date : {df_clean['date'].isnull().sum():,}")

# Записи извън очаквания период 2022-2024
outside = df_clean[
    (df_clean["date"].dt.year < 2022) | 
    (df_clean["date"].dt.year > 2024)
]
print(f"\nЗаписи извън 2022-2024: {len(outside):,}")
if len(outside) > 0:
    print(outside["date"].dt.year.value_counts().sort_index().to_string())

# Дублирани timestamps (различни записи в абсолютно същия момент и лента)
print(f"\nЗаписи по година след предобработката:")
print(df_clean["date"].dt.year.value_counts().sort_index().to_string())

# Проверка за нереалистични timestamps - напр. далеч в бъдещето или миналото
print(f"\nРазпределение по месец и година:")
monthly = (df_clean.groupby([df_clean["date"].dt.year.rename("year"),
                              df_clean["date"].dt.month.rename("month")])
           .size()
           .reset_index(name="count"))
print(monthly.to_string(index=False))

print("=" * 60)
print("1. КОНСИСТЕНТНОСТ LANE ↔ DIRECTION")
print("=" * 60)

lane_dir = (df_clean.groupby(["lane", "direction"])
            .size()
            .reset_index(name="count"))

lane_dir_pivot = lane_dir.pivot(index="lane", columns="direction", values="count").fillna(0)
lane_dir_pivot["total"] = lane_dir_pivot.sum(axis=1)

for col in ["Център", "Околовръстно"]:
    lane_dir_pivot[f"{col}_%"] = (lane_dir_pivot[col] / lane_dir_pivot["total"] * 100).round(2)

print(lane_dir_pivot.to_string())

print("=" * 60)
print("2. АНАЛИЗ НА VEHICLE_CLASS = 0")
print("=" * 60)

vc0 = df_clean[df_clean["vehicle_class"] == 0]
vc1 = df_clean[df_clean["vehicle_class"] == 1]

print(f"Клас 0 записи: {len(vc0):,} ({len(vc0)/len(df_clean)*100:.2f}%)")

print(f"\nСравнение клас 0 vs клас 1:")
for col in ["speed", "length", "weight"]:
    print(f"\n  {col}:")
    print(f"    Клас 0: mean={vc0[col].mean():.1f}, median={vc0[col].median():.1f}, std={vc0[col].std():.1f}")
    print(f"    Клас 1: mean={vc1[col].mean():.1f}, median={vc1[col].median():.1f}, std={vc1[col].std():.1f}")

print(f"\nРазпределение по lane:")
print(vc0["lane"].value_counts().to_string())

print(f"\nРазпределение по година:")
print(vc0["date"].dt.year.value_counts().sort_index().to_string())

print(f"\nWeight = 0 при клас 0:")
print(f"  {(vc0['weight'] == 0).sum():,} ({(vc0['weight'] == 0).mean()*100:.1f}%)")
print(f"Weight = 0 при клас 1:")
print(f"  {(vc1['weight'] == 0).sum():,} ({(vc1['weight'] == 0).mean()*100:.1f}%)")

print("=" * 60)
print("3. ПРОВЕРКА НА PLATE")
print("=" * 60)

print(f"Общо записи       : {len(df_clean):,}")
print(f"Null в plate      : {df_clean['plate'].isnull().sum():,} ({df_clean['plate'].isnull().mean()*100:.2f}%)")
print(f"Уникални номера   : {df_clean['plate'].nunique():,}")

# Дължина на номерата
plate_lengths = df_clean["plate"].dropna().str.len()
print(f"\nДължина на регистрационните номера:")
print(plate_lengths.value_counts().sort_index().to_string())

# Нереалистично къси или дълги
print(f"\nНомера с дължина < 3 символа:")
short = df_clean[df_clean["plate"].str.len() < 3].dropna(subset=["plate"])
print(f"  {len(short):,} записа")
if len(short) > 0:
    print(short["plate"].value_counts().head(10).to_string())

print(f"\nНомера с дължина > 12 символа:")
long_plates = df_clean[df_clean["plate"].str.len() > 12].dropna(subset=["plate"])
print(f"  {len(long_plates):,} записа")
if len(long_plates) > 0:
    print(long_plates["plate"].value_counts().head(10).to_string())

# Специални символи
print(f"\nНомера съдържащи само цифри (подозрително):")
only_digits = df_clean["plate"].dropna().str.match(r"^\d+$").sum()
print(f"  {only_digits:,} записа")

print("=" * 60)
print("ПОЧИСТВАНЕ НА PLATE")
print("=" * 60)

# Преди
print(f"Null преди: {df_clean['plate'].isnull().sum():,}")

# '-' -> NaN
df_clean["plate"] = df_clean["plate"].replace("-", np.nan)

# Дължина < 3 или > 12 -> NaN (нереалистични)
plate_len = df_clean["plate"].str.len()
df_clean.loc[plate_len < 3, "plate"] = np.nan
df_clean.loc[plate_len > 12, "plate"] = np.nan

# След
print(f"Null след : {df_clean['plate'].isnull().sum():,} ({df_clean['plate'].isnull().mean()*100:.2f}%)")
print(f"Уникални номера след: {df_clean['plate'].nunique():,}")

print(f"\nДължина на номерата след почистване:")
print(df_clean["plate"].dropna().str.len().value_counts().sort_index().to_string())

print("=" * 60)
print("4. ВРЕМЕВА КОНСИСТЕНТНОСТ")
print("=" * 60)

# Дали записите са наредени хронологично
is_sorted = df_clean["date"].is_monotonic_increasing
print(f"Хронологично наредени: {is_sorted}")

if not is_sorted:
    df_clean = df_clean.sort_values("date").reset_index(drop=True)
    print("Сортирани по дата")

# Дали има дублирани timestamps изобщо
dup_timestamps = df_clean.duplicated(subset=["date", "lane"]).sum()
print(f"Дублирани date+lane  : {dup_timestamps:,}")

# Минимален интервал между записи в една лента
print(f"\nМинимален интервал между записи по лента (секунди):")
for lane in ["L1", "L2", "L3", "L4", "L5", "L6"]:
    lane_dates = df_clean[df_clean["lane"] == lane]["date"].sort_values()
    min_diff = lane_dates.diff().dt.total_seconds().dropna()
    print(f"  {lane}: min={min_diff.min():.0f}s, "
          f"median={min_diff.median():.0f}s, "
          f"mean={min_diff.mean():.1f}s")

print("=" * 60)
print("5. ФИНАЛНА СТАТИСТИКА ПРЕДИ/СЛЕД")
print("=" * 60)

print(f"{'Метрика':<35} {'Преди':>15} {'След':>15}")
print("-" * 65)
print(f"{'Общо записи':<35} {'12,239,528':>15} {len(df_clean):>15,}")
print(f"{'Уникални ленти':<35} {'10 (L+LN)':>15} {df_clean['lane'].nunique():>15}")
print(f"{'Уникални vehicle_class':<35} {'38':>15} {df_clean['vehicle_class'].nunique():>15}")
print(f"{'Null в direction':<35} {'437,569':>15} {df_clean['direction'].isnull().sum():>15,}")
print(f"{'Null в country':<35} {'4,891,734':>15} {df_clean['country'].isnull().sum():>15,}")
print(f"{'Null в plate':<35} {'3,144,809':>15} {df_clean['plate'].isnull().sum():>15,}")
print(f"{'BGR записи':<35} {'1,139,232':>15} {(df_clean['country'] == 'BGR').sum():>15,}")
print(f"{'Записи със speed=0 или 255':<35} {'31,249':>15} {((df_clean['speed']==0)|(df_clean['speed']==255)).sum():>15,}")
print(f"{'Записи с length<=0':<35} {'213,801':>15} {(df_clean['length']<=0).sum():>15,}")
print(f"{'Записи с weight<0':<35} {'?':>15} {(df_clean['weight']<0).sum():>15,}")

print(f"\nОписателна статистика след предобработката:")
print(df_clean[["speed", "length", "weight"]].describe().round(2).to_string())

print("=" * 60)
print("6. ТЕМПОРАЛНА КОНСИСТЕНТНОСТ")
print("=" * 60)

# Средна скорост по час и ден - търсим аномални стойности
df_clean["hour"] = df_clean["date"].dt.hour
df_clean["dow"]  = df_clean["date"].dt.dayofweek
df_clean["date_only"] = df_clean["date"].dt.date

# Дневен брой записи - търсим аномални дни
daily = df_clean.groupby("date_only").agg(
    count        = ("speed", "size"),
    avg_speed    = ("speed", "mean"),
    avg_length   = ("length", "mean")
).reset_index()

print(f"Дневна статистика:")
print(f"  Средно записи/ден  : {daily['count'].mean():,.0f}")
print(f"  Std записи/ден     : {daily['count'].std():,.0f}")
print(f"  Min записи/ден     : {daily['count'].min():,}")
print(f"  Max записи/ден     : {daily['count'].max():,}")

# Дни с необичайно малко записи (под 10% от средното)
threshold_low  = daily["count"].mean() * 0.10
threshold_high = daily["count"].mean() * 3.0

anomaly_days_low  = daily[daily["count"] < threshold_low]
anomaly_days_high = daily[daily["count"] > threshold_high]

print(f"\nДни с под 10% от средния брой записи ({threshold_low:,.0f}):")
print(f"  {len(anomaly_days_low):,} дни")
print(anomaly_days_low.sort_values("count").head(20).to_string(index=False))

print(f"\nДни с над 300% от средния брой записи ({threshold_high:,.0f}):")
print(f"  {len(anomaly_days_high):,} дни")
print(anomaly_days_high.sort_values("count", ascending=False).head(10).to_string(index=False))

print("=" * 60)
print("АНАЛИЗ НА ДНИТЕ С МАЛКО ЗАПИСИ")
print("=" * 60)

# Дали тези дни са в началото/края на известни gap-ове?
anomaly_days_low["date_only"] = pd.to_datetime(anomaly_days_low["date_only"])

print("Разпределение по година:")
print(anomaly_days_low["date_only"].dt.year.value_counts().sort_index().to_string())

print("\nРазпределение по месец:")
print(anomaly_days_low["date_only"].dt.month.value_counts().sort_index().to_string())

# Средна скорост по час - търсим аномални часове
hourly_speed = (df_clean.groupby("hour")["speed"]
                .agg(["mean", "std", "count"])
                .round(2))

print(f"\nСредна скорост по час:")
print(hourly_speed.to_string())

# Има ли часове с необичайно висока/ниска средна скорост?
speed_mean = hourly_speed["mean"].mean()
speed_std  = hourly_speed["mean"].std()
anomaly_hours = hourly_speed[
    (hourly_speed["mean"] > speed_mean + 2*speed_std) |
    (hourly_speed["mean"] < speed_mean - 2*speed_std)
]
print(f"\nАномални часове (извън 2σ): {len(anomaly_hours)}")
if len(anomaly_hours) > 0:
    print(anomaly_hours.to_string())

# Премахване на помощните колони
df_clean = df_clean.drop(columns=["hour", "dow", "date_only"])

df_clean.to_csv("data/traffic_preprocessed.csv", index=False)
print(f"Запазено в data/traffic_preprocessed.csv")
print(f"Редове : {len(df_clean):,}")
print(f"Колони : {list(df_clean.columns)}")

