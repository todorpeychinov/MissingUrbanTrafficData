import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

# Зареждане
df = pd.read_csv(
    "data/traffic_aggregated.csv",
    parse_dates=["datetime"]
)

print(f"Редове  : {len(df):,}")
print(f"Колони  : {list(df.columns)}")
print(f"Период  : {df['datetime'].min()}  →  {df['datetime'].max()}")
print(f"\nNull стойности:")
print(df.isnull().sum().to_string())

print("=" * 60)
print("ДЕФИНИРАНЕ НА ТЕСТОВИ ПЕРИОДИ")
print("=" * 60)

# Тестови периоди - избираме периоди с непрекъснати данни
# 3 кратки (1-3 дни), 3 средни (1-2 седмици), 3 дълги (1 месец)

test_periods = {
    # Кратки — 3 дни
    "short_1" : ("2023-03-01", "2023-03-03"),
    "short_2" : ("2023-06-01", "2023-06-03"),
    "short_3" : ("2023-11-01", "2023-11-03"),
    # Средни — 2 седмици
    "medium_1": ("2023-03-15", "2023-03-28"),
    "medium_2": ("2023-06-10", "2023-06-23"),
    "medium_3": ("2023-10-15", "2023-10-28"),
    # Дълги — 1 месец
    "long_1"  : ("2023-03-01", "2023-03-31"),
    "long_2"  : ("2023-06-01", "2023-06-30"),
    "long_3"  : ("2023-10-01", "2023-10-31"),
}

# Проверка дали избраните периоди имат данни
print(f"{'Период':<12} {'Начало':<22} {'Край':<22} {'Null volume':>12} {'Общо часа':>10}")
print("-" * 78)

for name, (start, end) in test_periods.items():
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    subset = df[mask]
    null_count = subset["volume"].isnull().sum()
    total = len(subset)
    status = "OK" if null_count == 0 else f"{null_count} null"
    print(f"{name:<12} {start:<22} {end:<22} {status:>12} {total:>10,}")

print("=" * 60)
print("НАМИРАНЕ НА ПОДХОДЯЩИ ПЕРИОДИ")
print("=" * 60)

# Проверяваме месеците в 2023 за непрекъснати периоди
months_2023 = {
    "Яну": ("2023-01-01", "2023-01-31"),
    "Фев": ("2023-02-01", "2023-02-28"),
    "Мар": ("2023-03-01", "2023-03-31"),
    "Апр": ("2023-04-01", "2023-04-30"),
    "Май": ("2023-05-01", "2023-05-31"),
    "Юни": ("2023-06-01", "2023-06-30"),
    "Юли": ("2023-07-01", "2023-07-31"),
    "Авг": ("2023-08-01", "2023-08-31"),
    "Сеп": ("2023-09-01", "2023-09-30"),
    "Окт": ("2023-10-01", "2023-10-31"),
    "Ное": ("2023-11-01", "2023-11-30"),
    "Дек": ("2023-12-01", "2023-12-31"),
}

print(f"{'Месец':<8} {'Null volume':>12} {'Общо часа':>10}")
print("-" * 32)
for name, (start, end) in months_2023.items():
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    subset = df[mask]
    null_count = subset["volume"].isnull().sum()
    total = len(subset)
    status = "OK" if null_count == 0 else f"{null_count}"
    print(f"{name:<8} {status:>12} {total:>10,}")

print("=" * 60)
print("ПРОВЕРКА НА 2022")
print("=" * 60)

months_2022 = {
    "Яну": ("2022-01-01", "2022-01-31"),
    "Фев": ("2022-02-01", "2022-02-28"),
    "Мар": ("2022-03-01", "2022-03-31"),
    "Апр": ("2022-04-01", "2022-04-30"),
    "Май": ("2022-05-01", "2022-05-31"),
    "Юни": ("2022-06-01", "2022-06-30"),
    "Юли": ("2022-07-01", "2022-07-31"),
    "Авг": ("2022-08-01", "2022-08-31"),
    "Сеп": ("2022-09-01", "2022-09-30"),
    "Окт": ("2022-10-01", "2022-10-31"),
    "Ное": ("2022-11-01", "2022-11-30"),
    "Дек": ("2022-12-01", "2022-12-31"),
}

print(f"{'Месец':<8} {'Null volume':>12} {'Общо часа':>10}")
print("-" * 32)
for name, (start, end) in months_2022.items():
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    subset = df[mask]
    null_count = subset["volume"].isnull().sum()
    total = len(subset)
    status = "OK" if null_count == 0 else f"{null_count}"
    print(f"{name:<8} {status:>12} {total:>10,}")

print("=" * 60)
print("ФИНАЛНИ ТЕСТОВИ ПЕРИОДИ")
print("=" * 60)

# Чисти месеци: Юни 2023, Ноември 2023, Декември 2023
# Избираме периоди от тях

test_periods = {
    # Кратки — 3 дни
    "short_1" : ("2023-06-05", "2023-06-07"),
    "short_2" : ("2023-11-05", "2023-11-07"),
    "short_3" : ("2023-12-05", "2023-12-07"),
    # Средни — 2 седмици
    "medium_1": ("2023-06-05", "2023-06-18"),
    "medium_2": ("2023-11-05", "2023-11-18"),
    "medium_3": ("2023-12-05", "2023-12-18"),
    # Дълги — 1 месец
    "long_1"  : ("2023-06-01", "2023-06-30"),
    "long_2"  : ("2023-11-01", "2023-11-30"),
    "long_3"  : ("2023-12-01", "2023-12-31"),
}

print(f"{'Период':<12} {'Начало':<22} {'Край':<22} {'Null volume':>12} {'Общо часа':>10}")
print("-" * 78)

for name, (start, end) in test_periods.items():
    mask = (df["datetime"] >= start) & (df["datetime"] <= end)
    subset = df[mask]
    null_count = subset["volume"].isnull().sum()
    total = len(subset)
    status = "OK" if null_count == 0 else f"{null_count} null"
    print(f"{name:<12} {start:<22} {end:<22} {status:>12} {total:>10,}")

print("=" * 60)
print("EVALUATION ФУНКЦИИ И СИМУЛАЦИЯ НА GAP-ОВЕ")
print("=" * 60)

from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate(y_true, y_pred, period_name):
    """Изчислява MAE и RMSE за даден период."""
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"period": period_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2)}

def create_test_set(df, start, end):
    """
    Извлича период, запазва реалните стойности
    и създава версия с премахнат volume за импутация.
    """
    mask   = (df["datetime"] >= start) & (df["datetime"] <= end)
    subset = df[mask].copy().reset_index(drop=True)
    # Запазваме реалните стойности
    subset["volume_true"] = subset["volume"]
    # Премахваме volume за симулация на gap
    subset["volume"] = np.nan
    return subset

# Тест — проверяваме че работи
sample = create_test_set(df, "2023-06-05", "2023-06-07")
print(f"Тестов период: {len(sample):,} реда")
print(f"volume_true null : {sample['volume_true'].isnull().sum():,}")
print(f"volume null      : {sample['volume'].isnull().sum():,}")
print(f"\nПример:")
print(sample[["datetime", "direction", "volume", "volume_true"]].head(6).to_string(index=False))

print("=" * 60)
print("METHOD 1: MEAN IMPUTATION")
print("=" * 60)

# Средната стойност по посока от целия dataset (без null)
mean_by_direction = (df.groupby("direction")["volume"]
                     .mean()
                     .round(2))

print("Средна стойност по посока:")
print(mean_by_direction.to_string())

def mean_imputation(test_df, mean_values):
    """Попълва с средната стойност по посока."""
    result = test_df.copy()
    for direction, mean_val in mean_values.items():
        mask = result["direction"] == direction
        result.loc[mask, "volume_imputed"] = mean_val
    return result

# Прилагаме за всички 9 периода
results_mean = []

for name, (start, end) in test_periods.items():
    test  = create_test_set(df, start, end)
    imputed = mean_imputation(test, mean_by_direction)
    score = evaluate(
        imputed["volume_true"].values,
        imputed["volume_imputed"].values,
        name
    )
    score["category"] = name.split("_")[0]
    results_mean.append(score)

results_mean_df = pd.DataFrame(results_mean)
print(f"\nРезултати Mean Imputation:")
print(results_mean_df.to_string(index=False))

print(f"\nСредно по категория:")
print(results_mean_df.groupby("category")[["MAE","RMSE"]]
      .mean().round(2).to_string())

print("=" * 60)
print("METHOD 2: HISTORICAL AVERAGE")
print("=" * 60)

# Изчисляваме средния профил: час x ден от седмицата x посока
# Използваме само данните ИЗВЪН тестовите периоди

def historical_average_imputation(df, test_df, start, end):
    """
    Попълва с историческата средна стойност за
    същия час, ден от седмицата и посока.
    Използва само данни извън тестовия период.
    """
    # Тренировъчни данни — без тестовия период
    train = df[
        (df["datetime"] < start) | (df["datetime"] > end)
    ].copy()

    train["hour"] = train["datetime"].dt.hour
    train["dow"]  = train["datetime"].dt.dayofweek

    # Среден профил по час x dow x посока
    profile = (train.groupby(["dow", "hour", "direction"])["volume"]
               .mean()
               .reset_index()
               .rename(columns={"volume": "volume_imputed"}))

    # Прилагаме към тестовия период
    result = test_df.copy()
    result["hour"] = result["datetime"].dt.hour
    result["dow"]  = result["datetime"].dt.dayofweek

    result = result.merge(profile, on=["dow", "hour", "direction"], how="left")
    result["volume_imputed"] = result["volume_imputed"].round(2)

    return result

# Прилагаме за всички 9 периода
results_hist = []

for name, (start, end) in test_periods.items():
    test    = create_test_set(df, start, end)
    imputed = historical_average_imputation(df, test, start, end)
    score   = evaluate(
        imputed["volume_true"].values,
        imputed["volume_imputed"].values,
        name
    )
    score["category"] = name.split("_")[0]
    results_hist.append(score)

results_hist_df = pd.DataFrame(results_hist)
print(f"Резултати Historical Average:")
print(results_hist_df.to_string(index=False))

print(f"\nСредно по категория:")
print(results_hist_df.groupby("category")[["MAE","RMSE"]]
      .mean().round(2).to_string())

print("=" * 60)
print("СРАВНЕНИЕ НА СТАТИСТИЧЕСКИТЕ МЕТОДИ")
print("=" * 60)

comparison = pd.DataFrame({
    "period"    : results_mean_df["period"],
    "category"  : results_mean_df["category"],
    "MAE_mean"  : results_mean_df["MAE"],
    "MAE_hist"  : results_hist_df["MAE"],
    "RMSE_mean" : results_mean_df["RMSE"],
    "RMSE_hist" : results_hist_df["RMSE"],
})

comparison["MAE_improvement_%"]  = ((comparison["MAE_mean"] - comparison["MAE_hist"]) 
                                     / comparison["MAE_mean"] * 100).round(1)
comparison["RMSE_improvement_%"] = ((comparison["RMSE_mean"] - comparison["RMSE_hist"]) 
                                     / comparison["RMSE_mean"] * 100).round(1)

print(comparison.to_string(index=False))

print(f"\nСредно подобрение спрямо Mean:")
print(f"  MAE  : -{comparison['MAE_improvement_%'].mean():.1f}%")
print(f"  RMSE : -{comparison['RMSE_improvement_%'].mean():.1f}%")

print(f"\nСредно по категория:")
cat_summary = pd.DataFrame({
    "Mean MAE"  : results_mean_df.groupby("category")["MAE"].mean().round(2),
    "Hist MAE"  : results_hist_df.groupby("category")["MAE"].mean().round(2),
    "Mean RMSE" : results_mean_df.groupby("category")["RMSE"].mean().round(2),
    "Hist RMSE" : results_hist_df.groupby("category")["RMSE"].mean().round(2),
})
print(cat_summary.to_string())

# Визуализация на сравнението
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Сравнение на статистически методи", fontsize=14, fontweight="bold")

categories = ["short", "medium", "long"]
x = np.arange(len(categories))
width = 0.35

# MAE
mean_mae = [results_mean_df[results_mean_df["category"]==c]["MAE"].mean() for c in categories]
hist_mae = [results_hist_df[results_hist_df["category"]==c]["MAE"].mean() for c in categories]

axes[0].bar(x - width/2, mean_mae, width, label="Mean", color="coral", edgecolor="white")
axes[0].bar(x + width/2, hist_mae, width, label="Historical Avg", color="steelblue", edgecolor="white")
axes[0].set_title("MAE по категория")
axes[0].set_xticks(x)
axes[0].set_xticklabels(["Кратък (3д)", "Среден (2с)", "Дълъг (1м)"])
axes[0].set_ylabel("MAE (ПС/час)")
axes[0].legend()

# RMSE
mean_rmse = [results_mean_df[results_mean_df["category"]==c]["RMSE"].mean() for c in categories]
hist_rmse = [results_hist_df[results_hist_df["category"]==c]["RMSE"].mean() for c in categories]

axes[1].bar(x - width/2, mean_rmse, width, label="Mean", color="coral", edgecolor="white")
axes[1].bar(x + width/2, hist_rmse, width, label="Historical Avg", color="steelblue", edgecolor="white")
axes[1].set_title("RMSE по категория")
axes[1].set_xticks(x)
axes[1].set_xticklabels(["Кратък (3д)", "Среден (2с)", "Дълъг (1м)"])
axes[1].set_ylabel("RMSE (ПС/час)")
axes[1].legend()

plt.tight_layout()
plt.savefig("statistical_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# Запазване на статистическите резултати
results_mean_df.to_csv("data/results_mean.csv", index=False)
results_hist_df.to_csv("data/results_hist.csv", index=False)
print("Запазено в data/results_mean.csv")
print("Запазено в data/results_hist.csv")

