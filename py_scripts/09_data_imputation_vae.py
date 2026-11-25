import torch
print(f"PyTorch версия: {torch.__version__}")
print(f"CUDA достъпен : {torch.cuda.is_available()}")
print(f"CUDA версия   : {torch.version.cuda}")
print(f"GPU           : {torch.cuda.get_device_name(0)}")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство    : {device}")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Устройство: {device}")

# Зареждане
df = pd.read_csv(
    "data/traffic_aggregated.csv",
    parse_dates=["datetime"]
)

# Feature Engineering
df["hour"]          = df["datetime"].dt.hour
df["dow"]           = df["datetime"].dt.dayofweek
df["month"]         = df["datetime"].dt.month
df["direction_enc"] = (df["direction"] == "Център").astype(int)
df["hour_sin"]      = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"]      = np.cos(2 * np.pi * df["hour"] / 24)
df["dow_sin"]       = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"]       = np.cos(2 * np.pi * df["dow"] / 7)
df["month_sin"]     = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"]     = np.cos(2 * np.pi * df["month"] / 12)

feature_cols = ["hour_sin", "hour_cos", "dow_sin", "dow_cos",
                "month_sin", "month_cos", "direction_enc"]

print(f"Редове  : {len(df):,}")
print(f"Features: {feature_cols}")

print("=" * 60)
print("VAE АРХИТЕКТУРА")
print("=" * 60)

class ConditionalVAE(nn.Module):
    def __init__(self, input_dim, condition_dim, latent_dim, hidden_dims):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder
        encoder_layers = []
        in_dim = input_dim + condition_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu     = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim + condition_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.BatchNorm1d(h_dim),
                nn.Dropout(0.1)
            ])
            in_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def encode(self, x, c):
        h = self.encoder(torch.cat([x, c], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        return self.decoder(torch.cat([z, c], dim=1))
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z          = self.reparameterize(mu, logvar)
        recon      = self.decode(z, c)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """ELBO loss = Reconstruction loss + β * KL divergence"""
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_loss    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# Тест на архитектурата
test_model = ConditionalVAE(
    input_dim     = 1,
    condition_dim = len(feature_cols),
    latent_dim    = 16,
    hidden_dims   = [64, 32]
).to(device)

print(f"Модел архитектура:")
print(test_model)
print(f"\nПараметри: {sum(p.numel() for p in test_model.parameters()):,}")

print("=" * 60)
print("ПОДГОТОВКА НА ДАННИТЕ")
print("=" * 60)

# Скалиране на volume
scaler = StandardScaler()

# Тренировъчни данни
train_data = df[df["volume"].notna()].copy().reset_index(drop=True)

X_train = train_data[feature_cols].values.astype(np.float32)
y_train = train_data["volume"].values.astype(np.float32).reshape(-1, 1)

# Скалираме volume
y_train_scaled = scaler.fit_transform(y_train)

print(f"Тренировъчни редове: {len(train_data):,}")
print(f"Volume mean: {scaler.mean_[0]:.2f}, std: {scaler.scale_[0]:.2f}")

# Конфигурации за тестване
configs = [
    {"latent_dim": 8,  "hidden_dims": [64, 32],       "lr": 0.001,  "batch_size": 64,  "epochs": 100, "beta": 1.0},
    {"latent_dim": 16, "hidden_dims": [64, 32],       "lr": 0.001,  "batch_size": 64,  "epochs": 100, "beta": 1.0},
    {"latent_dim": 32, "hidden_dims": [64, 32],       "lr": 0.001,  "batch_size": 64,  "epochs": 100, "beta": 1.0},
    {"latent_dim": 16, "hidden_dims": [128, 64, 32],  "lr": 0.001,  "batch_size": 64,  "epochs": 100, "beta": 1.0},
    {"latent_dim": 16, "hidden_dims": [64, 32],       "lr": 0.0001, "batch_size": 64,  "epochs": 100, "beta": 1.0},
    {"latent_dim": 16, "hidden_dims": [64, 32],       "lr": 0.001,  "batch_size": 128, "epochs": 100, "beta": 1.0},
    {"latent_dim": 16, "hidden_dims": [64, 32],       "lr": 0.001,  "batch_size": 64,  "epochs": 100, "beta": 0.5},
]

def train_vae(X_train, y_train_scaled, config, verbose=False):
    """Обучава VAE с дадена конфигурация."""
    
    # Dataset
    X_tensor = torch.FloatTensor(X_train).to(device)
    y_tensor = torch.FloatTensor(y_train_scaled).to(device)
    dataset  = TensorDataset(y_tensor, X_tensor)
    loader   = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)
    
    # Модел
    model = ConditionalVAE(
        input_dim     = 1,
        condition_dim = len(feature_cols),
        latent_dim    = config["latent_dim"],
        hidden_dims   = config["hidden_dims"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Обучение
    train_losses = []
    model.train()
    
    for epoch in range(config["epochs"]):
        epoch_loss = 0
        for y_batch, x_batch in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(y_batch, x_batch)
            loss, _, _        = vae_loss(recon, y_batch, mu, logvar, config["beta"])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:>3}: loss={avg_loss:.6f}")
    
    return model, train_losses

def vae_impute(model, scaler, test_df, feature_cols):
    """Импутира с VAE."""
    model.eval()
    with torch.no_grad():
        X_test  = torch.FloatTensor(
            test_df[feature_cols].values.astype(np.float32)
        ).to(device)
        # Сэмплираме от prior
        z       = torch.randn(len(X_test), model.latent_dim).to(device)
        recon   = model.decode(z, X_test)
        imputed = scaler.inverse_transform(recon.cpu().numpy())
        imputed = np.maximum(imputed, 0)  # Не допускаме отрицателни стойности
    return imputed.flatten()

# Evaluation функции
def evaluate(y_true, y_pred, period_name):
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = np.sqrt(mean_squared_error(y_true, y_pred))
    return {"period": period_name, "MAE": round(mae, 2), "RMSE": round(rmse, 2)}

def create_test_set(df, start, end):
    mask   = (df["datetime"] >= start) & (df["datetime"] <= end)
    subset = df[mask].copy().reset_index(drop=True)
    subset["volume_true"] = subset["volume"]
    subset["volume"]      = np.nan
    return subset

test_periods = {
    "short_1" : ("2023-06-05", "2023-06-07"),
    "short_2" : ("2023-11-05", "2023-11-07"),
    "short_3" : ("2023-12-05", "2023-12-07"),
    "medium_1": ("2023-06-05", "2023-06-18"),
    "medium_2": ("2023-11-05", "2023-11-18"),
    "medium_3": ("2023-12-05", "2023-12-18"),
    "long_1"  : ("2023-06-01", "2023-06-30"),
    "long_2"  : ("2023-11-01", "2023-11-30"),
    "long_3"  : ("2023-12-01", "2023-12-31"),
}

print(f"\nКонфигурации за тестване: {len(configs)}")
print(f"Тестови периоди         : {len(test_periods)}")

print("=" * 60)
print("VAE HYPERPARAMETER TUNING")
print("=" * 60)

tuning_results = []

for i, config in enumerate(configs):
    print(f"\nКонфигурация {i+1}/{len(configs)}: "
          f"latent={config['latent_dim']}, "
          f"hidden={config['hidden_dims']}, "
          f"lr={config['lr']}, "
          f"batch={config['batch_size']}, "
          f"beta={config['beta']}")
    
    # Обучаваме модела върху всички тренировъчни данни
    model, losses = train_vae(X_train, y_train_scaled, config, verbose=True)
    
    # Оценяваме върху тестовите периоди
    period_scores = []
    for name, (start, end) in test_periods.items():
        test    = create_test_set(df, start, end)
        imputed = vae_impute(model, scaler, test, feature_cols)
        score   = evaluate(
            test["volume_true"].values,
            imputed,
            name
        )
        score["category"] = name.split("_")[0]
        period_scores.append(score)
    
    scores_df = pd.DataFrame(period_scores)
    avg_mae   = scores_df["MAE"].mean()
    avg_rmse  = scores_df["RMSE"].mean()
    
    tuning_results.append({
        "config_id"  : i + 1,
        "latent_dim" : config["latent_dim"],
        "hidden_dims": str(config["hidden_dims"]),
        "lr"         : config["lr"],
        "batch_size" : config["batch_size"],
        "beta"       : config["beta"],
        "MAE_mean"   : round(avg_mae, 2),
        "RMSE_mean"  : round(avg_rmse, 2),
        "final_loss" : round(losses[-1], 6)
    })
    
    print(f"  → Avg MAE={avg_mae:.2f}, Avg RMSE={avg_rmse:.2f}, "
          f"Final loss={losses[-1]:.6f}")

tuning_df = pd.DataFrame(tuning_results)
print(f"\n{'='*60}")
print(f"РЕЗУЛТАТИ ОТ TUNING:")
print(tuning_df.to_string(index=False))

best_idx    = tuning_df["MAE_mean"].idxmin()
best_config = configs[best_idx]
print(f"\nНай-добра конфигурация: #{best_idx+1}")
print(f"  latent_dim  : {best_config['latent_dim']}")
print(f"  hidden_dims : {best_config['hidden_dims']}")
print(f"  lr          : {best_config['lr']}")
print(f"  batch_size  : {best_config['batch_size']}")
print(f"  beta        : {best_config['beta']}")
print(f"  MAE         : {tuning_df.loc[best_idx, 'MAE_mean']}")
print(f"  RMSE        : {tuning_df.loc[best_idx, 'RMSE_mean']}")

print("=" * 60)
print("VAE — РАЗШИРЕНО ОБУЧЕНИЕ НА НАЙ-ДОБРАТА КОНФИГУРАЦИЯ")
print("=" * 60)

# Тестваме с повече epochs
extended_configs = [
    {"latent_dim": 8, "hidden_dims": [64, 32],      "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
    {"latent_dim": 8, "hidden_dims": [64, 32],      "lr": 0.001, "batch_size": 64, "epochs": 300, "beta": 1.0},
    {"latent_dim": 8, "hidden_dims": [128, 64, 32], "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
]

extended_results = []

for i, config in enumerate(extended_configs):
    print(f"\nКонфигурация {i+1}/{len(extended_configs)}: "
          f"latent={config['latent_dim']}, "
          f"hidden={config['hidden_dims']}, "
          f"epochs={config['epochs']}")
    
    model, losses = train_vae(X_train, y_train_scaled, config, verbose=True)
    
    period_scores = []
    for name, (start, end) in test_periods.items():
        test    = create_test_set(df, start, end)
        imputed = vae_impute(model, scaler, test, feature_cols)
        score   = evaluate(
            test["volume_true"].values,
            imputed,
            name
        )
        score["category"] = name.split("_")[0]
        period_scores.append(score)
    
    scores_df = pd.DataFrame(period_scores)
    avg_mae   = scores_df["MAE"].mean()
    avg_rmse  = scores_df["RMSE"].mean()
    
    extended_results.append({
        "latent_dim" : config["latent_dim"],
        "hidden_dims": str(config["hidden_dims"]),
        "epochs"     : config["epochs"],
        "MAE_mean"   : round(avg_mae, 2),
        "RMSE_mean"  : round(avg_rmse, 2),
        "final_loss" : round(losses[-1], 6)
    })
    
    print(f"  → Avg MAE={avg_mae:.2f}, Avg RMSE={avg_rmse:.2f}, "
          f"Final loss={losses[-1]:.6f}")

extended_df = pd.DataFrame(extended_results)
print(f"\nРезултати от разширеното обучение:")
print(extended_df.to_string(index=False))

print("=" * 60)
print("VAE — ФИНАЛНО ТЪРСЕНЕ")
print("=" * 60)

final_configs = [
    {"latent_dim": 8,  "hidden_dims": [128, 64, 32], "lr": 0.001, "batch_size": 64, "epochs": 300, "beta": 1.0},
    {"latent_dim": 8,  "hidden_dims": [128, 64, 32], "lr": 0.001, "batch_size": 32, "epochs": 200, "beta": 1.0},
    {"latent_dim": 16, "hidden_dims": [128, 64, 32], "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
    {"latent_dim": 8,  "hidden_dims": [256, 128, 64], "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
]

final_results = []

for i, config in enumerate(final_configs):
    print(f"\nКонфигурация {i+1}/{len(final_configs)}: "
          f"latent={config['latent_dim']}, "
          f"hidden={config['hidden_dims']}, "
          f"epochs={config['epochs']}, "
          f"batch={config['batch_size']}")
    
    model, losses = train_vae(X_train, y_train_scaled, config, verbose=True)
    
    period_scores = []
    for name, (start, end) in test_periods.items():
        test    = create_test_set(df, start, end)
        imputed = vae_impute(model, scaler, test, feature_cols)
        score   = evaluate(
            test["volume_true"].values,
            imputed,
            name
        )
        score["category"] = name.split("_")[0]
        period_scores.append(score)
    
    scores_df = pd.DataFrame(period_scores)
    avg_mae   = scores_df["MAE"].mean()
    avg_rmse  = scores_df["RMSE"].mean()
    
    final_results.append({
        "latent_dim" : config["latent_dim"],
        "hidden_dims": str(config["hidden_dims"]),
        "epochs"     : config["epochs"],
        "batch_size" : config["batch_size"],
        "MAE_mean"   : round(avg_mae, 2),
        "RMSE_mean"  : round(avg_rmse, 2),
        "final_loss" : round(losses[-1], 6)
    })
    
    print(f"  → Avg MAE={avg_mae:.2f}, Avg RMSE={avg_rmse:.2f}, "
          f"Final loss={losses[-1]:.6f}")

final_df = pd.DataFrame(final_results)
print(f"\nФинални резултати:")
print(final_df.to_string(index=False))

# Обединяваме всички резултати
all_results = pd.concat([
    tuning_df[["latent_dim","hidden_dims","MAE_mean","RMSE_mean","final_loss"]],
    extended_df[["latent_dim","hidden_dims","MAE_mean","RMSE_mean","final_loss"]],
    final_df[["latent_dim","hidden_dims","MAE_mean","RMSE_mean","final_loss"]]
]).reset_index(drop=True)

best_overall = all_results.loc[all_results["MAE_mean"].idxmin()]
print(f"\nНай-добра конфигурация overall:")
print(best_overall.to_string())

print("=" * 60)
print("VAE — ФИНАЛНА ОПТИМИЗАЦИЯ")
print("=" * 60)

optimization_configs = [
    {"latent_dim": 8,  "hidden_dims": [256, 128, 64], "lr": 0.001, "batch_size": 64,  "epochs": 300, "beta": 1.0},
    {"latent_dim": 8,  "hidden_dims": [256, 128, 64], "lr": 0.001, "batch_size": 32,  "epochs": 200, "beta": 1.0},
    {"latent_dim": 8,  "hidden_dims": [256, 128, 64], "lr": 0.001, "batch_size": 64,  "epochs": 200, "beta": 0.5},
    {"latent_dim": 16, "hidden_dims": [256, 128, 64], "lr": 0.001, "batch_size": 64,  "epochs": 200, "beta": 1.0},
    {"latent_dim": 8,  "hidden_dims": [512, 256, 128], "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
]

optimization_results = []

for i, config in enumerate(optimization_configs):
    print(f"\nКонфигурация {i+1}/{len(optimization_configs)}: "
          f"latent={config['latent_dim']}, "
          f"hidden={config['hidden_dims']}, "
          f"epochs={config['epochs']}, "
          f"batch={config['batch_size']}, "
          f"beta={config['beta']}")
    
    model, losses = train_vae(X_train, y_train_scaled, config, verbose=True)
    
    period_scores = []
    for name, (start, end) in test_periods.items():
        test    = create_test_set(df, start, end)
        imputed = vae_impute(model, scaler, test, feature_cols)
        score   = evaluate(
            test["volume_true"].values,
            imputed,
            name
        )
        score["category"] = name.split("_")[0]
        period_scores.append(score)
    
    scores_df = pd.DataFrame(period_scores)
    avg_mae   = scores_df["MAE"].mean()
    avg_rmse  = scores_df["RMSE"].mean()
    
    optimization_results.append({
        "latent_dim" : config["latent_dim"],
        "hidden_dims": str(config["hidden_dims"]),
        "epochs"     : config["epochs"],
        "batch_size" : config["batch_size"],
        "beta"       : config["beta"],
        "MAE_mean"   : round(avg_mae, 2),
        "RMSE_mean"  : round(avg_rmse, 2),
        "final_loss" : round(losses[-1], 6)
    })
    
    print(f"  → Avg MAE={avg_mae:.2f}, Avg RMSE={avg_rmse:.2f}, "
          f"Final loss={losses[-1]:.6f}")

opt_df = pd.DataFrame(optimization_results)
print(f"\nРезултати от оптимизацията:")
print(opt_df.to_string(index=False))

best_idx = opt_df["MAE_mean"].idxmin()
print(f"\nНай-добра конфигурация: {opt_df.loc[best_idx].to_dict()}")

print("=" * 60)
print("VAE — ПОСЛЕДЕН КРЪГ")
print("=" * 60)

last_configs = [
    {"latent_dim": 8, "hidden_dims": [512, 256, 128], "lr": 0.001, "batch_size": 64, "epochs": 300, "beta": 1.0},
    {"latent_dim": 8, "hidden_dims": [512, 256, 128], "lr": 0.001, "batch_size": 32, "epochs": 200, "beta": 1.0},
    {"latent_dim": 8, "hidden_dims": [1024, 512, 256], "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
]

last_results = []

for i, config in enumerate(last_configs):
    print(f"\nКонфигурация {i+1}/{len(last_configs)}: "
          f"latent={config['latent_dim']}, "
          f"hidden={config['hidden_dims']}, "
          f"epochs={config['epochs']}, "
          f"batch={config['batch_size']}")
    
    model, losses = train_vae(X_train, y_train_scaled, config, verbose=True)
    
    period_scores = []
    for name, (start, end) in test_periods.items():
        test    = create_test_set(df, start, end)
        imputed = vae_impute(model, scaler, test, feature_cols)
        score   = evaluate(
            test["volume_true"].values,
            imputed,
            name
        )
        score["category"] = name.split("_")[0]
        period_scores.append(score)
    
    scores_df = pd.DataFrame(period_scores)
    avg_mae   = scores_df["MAE"].mean()
    avg_rmse  = scores_df["RMSE"].mean()
    
    last_results.append({
        "latent_dim" : config["latent_dim"],
        "hidden_dims": str(config["hidden_dims"]),
        "epochs"     : config["epochs"],
        "batch_size" : config["batch_size"],
        "MAE_mean"   : round(avg_mae, 2),
        "RMSE_mean"  : round(avg_rmse, 2),
        "final_loss" : round(losses[-1], 6)
    })
    
    print(f"  → Avg MAE={avg_mae:.2f}, Avg RMSE={avg_rmse:.2f}, "
          f"Final loss={losses[-1]:.6f}")

last_df = pd.DataFrame(last_results)
print(f"\nРезултати:")
print(last_df.to_string(index=False))

print("=" * 60)
print("VAE — ФИНАЛЕН КРЪГ")
print("=" * 60)

final_round_configs = [
    {"latent_dim": 8, "hidden_dims": [1024, 512, 256], "lr": 0.001, "batch_size": 64, "epochs": 300, "beta": 1.0},
    {"latent_dim": 8, "hidden_dims": [1024, 512, 256], "lr": 0.001, "batch_size": 32, "epochs": 200, "beta": 1.0},
    {"latent_dim": 8, "hidden_dims": [2048, 1024, 512], "lr": 0.001, "batch_size": 64, "epochs": 200, "beta": 1.0},
]

final_round_results = []

for i, config in enumerate(final_round_configs):
    print(f"\nКонфигурация {i+1}/{len(final_round_configs)}: "
          f"latent={config['latent_dim']}, "
          f"hidden={config['hidden_dims']}, "
          f"epochs={config['epochs']}, "
          f"batch={config['batch_size']}")
    
    model, losses = train_vae(X_train, y_train_scaled, config, verbose=True)
    
    period_scores = []
    for name, (start, end) in test_periods.items():
        test    = create_test_set(df, start, end)
        imputed = vae_impute(model, scaler, test, feature_cols)
        score   = evaluate(
            test["volume_true"].values,
            imputed,
            name
        )
        score["category"] = name.split("_")[0]
        period_scores.append(score)
    
    scores_df = pd.DataFrame(period_scores)
    avg_mae   = scores_df["MAE"].mean()
    avg_rmse  = scores_df["RMSE"].mean()
    
    final_round_results.append({
        "latent_dim" : config["latent_dim"],
        "hidden_dims": str(config["hidden_dims"]),
        "epochs"     : config["epochs"],
        "batch_size" : config["batch_size"],
        "MAE_mean"   : round(avg_mae, 2),
        "RMSE_mean"  : round(avg_rmse, 2),
        "final_loss" : round(losses[-1], 6)
    })
    
    print(f"  → Avg MAE={avg_mae:.2f}, Avg RMSE={avg_rmse:.2f}, "
          f"Final loss={losses[-1]:.6f}")

final_round_df = pd.DataFrame(final_round_results)
print(f"\nРезултати:")
print(final_round_df.to_string(index=False))

best_idx = final_round_df["MAE_mean"].idxmin()
print(f"\nНай-добра конфигурация: {final_round_df.loc[best_idx].to_dict()}")

print("=" * 60)
print("VAE — ФИНАЛНИ РЕЗУЛТАТИ")
print("=" * 60)

# Обучаваме финалния модел с най-добрата конфигурация
best_vae_config = {
    "latent_dim" : 8,
    "hidden_dims": [2048, 1024, 512],
    "lr"         : 0.001,
    "batch_size" : 64,
    "epochs"     : 200,
    "beta"       : 1.0
}

print(f"Финална конфигурация: {best_vae_config}")
final_vae_model, _ = train_vae(X_train, y_train_scaled, best_vae_config, verbose=True)

# Оценяваме по всички тестови периоди
results_vae = []
for name, (start, end) in test_periods.items():
    test    = create_test_set(df, start, end)
    imputed = vae_impute(final_vae_model, scaler, test, feature_cols)
    score   = evaluate(
        test["volume_true"].values,
        imputed,
        name
    )
    score["category"] = name.split("_")[0]
    results_vae.append(score)

results_vae_df = pd.DataFrame(results_vae)
print(f"\nФинални резултати VAE:")
print(results_vae_df.to_string(index=False))
print(f"\nСредно по категория:")
print(results_vae_df.groupby("category")[["MAE","RMSE"]]
      .mean().round(2).to_string())

# Запазване
results_vae_df.to_csv("data/results_vae.csv", index=False)
print(f"\nЗапазено в data/results_vae.csv")

