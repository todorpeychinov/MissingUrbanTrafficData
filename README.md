# Traffic Data Imputation

Анализ и попълване на липсващи данни от сензор за трафик за периода 2022–2024.
Суровите записи (по едно превозно средство) се агрегират по час и посока, след което
пропуските във `volume` и `avg_speed` се попълват с различни методи и резултатите се сравняват.

## Структура

| Notebook | Описание |
|---|---|
| `01_combine_data` | Обединяване на годишните файлове в един набор |
| `02_data_exploration` | Първичен анализ, почасови редове, пропуски |
| `03_data_preprocess` | Дубликати, невалидни стойности, почистване |
| `04_feature_selection` | Корелации и избор на признаци |
| `05_data_aggregation` | Агрегиране по час и посока |
| `06_data_imputation_statistics_methods` | Базови методи – средно и исторически стойности |
| `07_data_imputation_knn` | KNN импутация |
| `08_data_imputation_random_forest` | Random Forest импутация |
| `09_data_imputation_vae` | Variational Autoencoder |
| `10_data_imputation_gan` | GAN |
| `11_imputation_comparsion` | Сравнение на методите (MAE, RMSE) по дължина на пропуска |
| `12_final_data_imputation` | Финална хибридна импутация на `volume` (RF за пропуски под 168 ч., GAN за по-дълги) |
| `13_check_outliers` | Проверка за аномални стойности |
| `14_speed_imputation` | Импутация на `avg_speed`, с `volume` като допълнителен предиктор |
| `15_round_data` | Закръгляне на импутираните стойности |

Папката `py_scripts/` съдържа същите notebook-и като `.py` файлове, генерирани с `converter.py`.

## Стартиране

Нужен е Python 3.12. `torch` е CUDA 12.4 версия, затова е нужен допълнителният индекс на PyTorch.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124
jupyter notebook
```

Notebook се пускат поред. Всеки чете от и записва в папка `data/`.

## Данни

Данните не са включени в хранилището. За да се изпълнят notebooks, файловете
`2022_combined.csv`, `2023_combined.csv` и `2024_combined.csv` трябва да са в `data/`.
