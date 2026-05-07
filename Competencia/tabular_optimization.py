# %%
"""
Optimización Bayesiana de Modelos Tabulares
===========================================
Esta libreta utiliza Optuna para encontrar los mejores hiperparámetros 
para los modelos LightGBM y XGBoost del ensamble multimodal.

La métrica objetivo a minimizar es el MAPE (Mean Absolute Percentage Error).
"""

# %% [markdown]
# ## 1. Configuración e Imports

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb
import optuna

# Importar el pipeline de feature engineering existente para garantizar consistencia
from multimodal_ensemble import feature_engineering

# Configuración
BASE = Path(".")
DATA_TAB = BASE / "data" / "tabular"
TARGET = "log_price"
N_FOLDS = 5
RANDOM_STATE = 42

# Límites de la optimización (a petición del usuario)
N_TRIALS = 15

print(f"Optuna version: {optuna.__version__}")

# %% [markdown]
# ## 2. Carga de Datos y Feature Engineering

# %%
print("Cargando datasets...")
train = pd.read_csv(DATA_TAB / "train.csv")
test  = pd.read_csv(DATA_TAB / "test.csv")

print("Generando features (reutilizando la función de multimodal_ensemble.py)...")
train_fe, test_fe = feature_engineering(train, test)

FEATURES_TAB = [
    # Originales clave
    "bedrooms", "bathrooms", "livingArea", "yearBuilt",
    "latitude", "longitude", "lotAreaValue", "photoCount",
    "taxAssessedValue", "propertyTaxRate", "has_hoa", "hoa_fee_monthly",
    "has_pool", "has_garage", "has_waterfront",
    "tag_price_cut", "tag_new_construction", "tag_foreclosure",
    "avg_school_rating", "max_school_rating", "num_nearby_schools", "min_school_distance",
    "num_tax_records", "num_sales", "num_price_changes", "last_listing_price",
    "latest_tax_value", "latest_tax_paid",
    "desc_length", "desc_word_count", "desc_is_boilerplate",
    "desc_mentions_renovated", "desc_mentions_pool", "desc_mentions_view", "desc_mentions_new",
    "log_living_area", "log_lot_area",
    "bath_to_bed_ratio", "property_age",
    # Engineered
    "total_rooms", "area_per_room", "density_ratio", "tax_per_sqft",
    "listing_vs_tax", "hoa_per_sqft", "photos_per_room",
    "age", "age_x_area", "property_age_bucket",
    "desc_quality_score", "premium_flag",
    "school_x_area", "waterfront_x_area", "pool_x_area",
    "tax_rate_x_value", "assessed_ratio",
    "zip_median_price_enc", "zip_mean_price_enc", "zip3d_median_enc",
    "geo_cluster", "area_rank_in_zip",
]

CAT_FEATURES = ["homeType", "zipcode", "zip_3digit", "geo_cluster"]
for col in CAT_FEATURES:
    train_fe[col] = train_fe[col].astype("category")

# Filtrar las columnas que existan en el dataframe
FEATURES_TAB = [f for f in FEATURES_TAB if f in train_fe.columns]
FEATURES_TAB += [c for c in CAT_FEATURES if c in train_fe.columns and c not in FEATURES_TAB]

X = train_fe[FEATURES_TAB]
y = train_fe[TARGET]

print(f"Dataset listo. Dimensiones X: {X.shape}, y: {y.shape}")

# %% [markdown]
# ## 3. Función Objetivo: LightGBM

# %%
lgb_cats = [c for c in CAT_FEATURES if c in FEATURES_TAB]

def objective_lgb(trial):
    # Espacio de búsqueda de hiperparámetros
    param = {
        "n_estimators": 2000,  # Fijo, usamos early stopping
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 512),
        "max_depth": trial.suggest_int("max_depth", 5, 15),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "random_state": RANDOM_STATE,
        "verbosity": -1,
        "categorical_feature": lgb_cats
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(**param)
        
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
        ]
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks
        )

        val_pred = model.predict(X_val)
        # Evaluamos el MAPE en el dominio real del precio (expm1)
        fold_mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(val_pred))
        mapes.append(fold_mape)

    return np.mean(mapes)

# %% [markdown]
# ## 4. Ejecutar Optimización LightGBM

# %%
print("="*50)
print("▶ Iniciando Optimización de LightGBM")
print("="*50)

study_lgb = optuna.create_study(direction="minimize", study_name="LGBM_Tuning")
study_lgb.optimize(objective_lgb, n_trials=N_TRIALS)

print("\n🏆 Mejores hiperparámetros para LightGBM:")
for key, value in study_lgb.best_params.items():
    print(f"    {key}: {value}")
print(f"📉 Mejor OOF MAPE: {study_lgb.best_value:.4%}")

# %% [markdown]
# ## 5. Función Objetivo: XGBoost

# %%
X_xgb = X.copy()
# XGBoost maneja categóricas nativamente si usamos `enable_categorical=True` o hacemos encoding
for col in CAT_FEATURES:
    if col in X_xgb.columns:
        X_xgb[col] = X_xgb[col].cat.codes

def objective_xgb(trial):
    param = {
        "n_estimators": 2000, # Fijo, usamos early stopping
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        "random_state": RANDOM_STATE,
        "verbosity": 0,
        "tree_method": "hist",
        "early_stopping_rounds": 100
    }

    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    mapes = []

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_xgb)):
        X_tr, X_val = X_xgb.iloc[tr_idx], X_xgb.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        model = xgb.XGBRegressor(**param)
        
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        val_pred = model.predict(X_val)
        fold_mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(val_pred))
        mapes.append(fold_mape)

    return np.mean(mapes)

# %% [markdown]
# ## 6. Ejecutar Optimización XGBoost

# %%
print("="*50)
print("▶ Iniciando Optimización de XGBoost")
print("="*50)

study_xgb = optuna.create_study(direction="minimize", study_name="XGB_Tuning")
study_xgb.optimize(objective_xgb, n_trials=N_TRIALS)

print("\n🏆 Mejores hiperparámetros para XGBoost:")
for key, value in study_xgb.best_params.items():
    print(f"    {key}: {value}")
print(f"📉 Mejor OOF MAPE: {study_xgb.best_value:.4%}")

# %% [markdown]
# ## 7. Resumen de Hiperparámetros (Copia y Pega)

# %%
print("\n" + "="*80)
print("Copia y pega estos diccionarios en multimodal_ensemble.py")
print("="*80)

print("\n--- LightGBM ---")
print("def lgbm_fn():")
print("    return lgb.LGBMRegressor(")
print("        n_estimators=3000,")
for k, v in study_lgb.best_params.items():
    if isinstance(v, float):
        print(f"        {k}={v:.5f},")
    else:
        print(f"        {k}={v},")
print("        random_state=RANDOM_STATE, verbosity=-1,")
print("        categorical_feature=lgb_cats")
print("    )")

print("\n--- XGBoost ---")
print("def xgb_fn():")
print("    return xgb.XGBRegressor(")
print("        n_estimators=3000,")
for k, v in study_xgb.best_params.items():
    if isinstance(v, float):
        print(f"        {k}={v:.5f},")
    else:
        print(f"        {k}={v},")
print("        random_state=RANDOM_STATE, verbosity=0, tree_method='hist',")
print("        early_stopping_rounds=100")
print("    )")
