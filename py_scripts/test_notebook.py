import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Зареждане
df = pd.read_csv(
    "data/traffic_aggregated.csv",
    parse_dates=["datetime"],
    low_memory=False
)

df

