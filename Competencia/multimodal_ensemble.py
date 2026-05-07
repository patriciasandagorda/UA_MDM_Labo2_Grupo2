# %%
"""
Multimodal Ensemble — Predicción de Precios Inmobiliarios
==========================================================
Pipelines:
  1. Tabular  — Feature Engineering + LightGBM / XGBoost / CatBoost / Ridge
  2. Texto    — DistilBERT embeddings (CPU) + Ridge
  3. Imágenes — ResNet50 embeddings + SVR / XGBoost
  4. Ensamble — Ridge Stacking sobre predicciones OOF

Ejecutar desde: participant/
  python multimodal_ensemble.py
"""

# %% [markdown]
# ## 0. Imports y Configuración

# %%
import warnings
warnings.filterwarnings("ignore")

import os, gc
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.svm import SVR
from sklearn.cluster import KMeans

import lightgbm as lgb
import xgboost as xgb

# Rutas base
BASE = Path(".")
DATA_TAB = BASE / "data" / "tabular"
DATA_IMG = BASE / "data" / "images"
SUBMISSIONS = BASE / "submissions"
CACHE = BASE / "data" / "cache"
CACHE.mkdir(exist_ok=True)
SUBMISSIONS.mkdir(exist_ok=True)

TARGET = "log_price"
PRICE_COL = "lastSoldPrice_hpi_adjusted"
N_FOLDS = 5
RANDOM_STATE = 42

# ── Flags de ejecución ────────────────────────────────────────────────────────
# Poner en False para cargar predicciones de la corrida anterior (caché).
# Útil para iterar rápido sobre una sola modalidad sin re-correr las demás.
RUN_TABULAR = True
RUN_TEXT    = True
RUN_IMAGE   = True

print("✅ Imports OK")
print(f"   Pipelines a ejecutar: Tabular={RUN_TABULAR} | Texto={RUN_TEXT} | Imagen={RUN_IMAGE}")

# %% [markdown]
# ## 1. Carga de Datos

# %%
train = pd.read_csv(DATA_TAB / "train_processed.csv")
test  = pd.read_csv(DATA_TAB / "test_processed.csv")

print(f"Train: {train.shape} | Test: {test.shape}")
print(f"Columnas: {train.columns.tolist()}")

# %% [markdown]
# ## 2. Feature Engineering Tabular

# %%
def feature_engineering(df_train: pd.DataFrame, df_test: pd.DataFrame):
    """
    Genera features con sentido de negocio para ambos datasets.
    Aplica target encoding con K-Fold solo en train para evitar leakage.
    """
    tr = df_train.copy()
    te = df_test.copy()

    current_year = 2024

    for df in [tr, te]:
        # ── Relaciones de tamaño y calidad ────────────────────────────────
        df["total_rooms"]        = df["bedrooms"].fillna(0) + df["bathrooms"].fillna(0)
        df["area_per_room"]      = df["livingArea"] / (df["total_rooms"] + 1)
        df["density_ratio"]      = df["livingArea"] / (df["lotAreaValue"].replace(0, np.nan))
        df["tax_per_sqft"]       = df["taxAssessedValue"] / (df["livingArea"].replace(0, np.nan))
        df["listing_vs_tax"]     = df["last_listing_price"] / (df["taxAssessedValue"].replace(0, np.nan))
        df["hoa_per_sqft"]       = df["hoa_fee_monthly"] / (df["livingArea"].replace(0, np.nan))
        df["photos_per_room"]    = df["photoCount"] / (df["total_rooms"] + 1)

        # ── Antigüedad y ciclo de vida ─────────────────────────────────────
        df["age"]                = current_year - df["yearBuilt"].fillna(df["yearBuilt"].median())
        df["age_x_area"]         = df["age"] * df["livingArea"]
        df["property_age_bucket"] = pd.cut(
            df["age"], bins=[-1, 5, 15, 30, 50, 9999],
            labels=[0, 1, 2, 3, 4]
        ).astype(float)

        # ── Percepción y marketing ─────────────────────────────────────────
        df["desc_quality_score"] = (
            df["desc_word_count"]
            + df["desc_mentions_renovated"] * 5
            + df["desc_mentions_view"] * 3
            + df["desc_mentions_pool"] * 2
            - df["desc_is_boilerplate"] * 10
        )
        df["premium_flag"]       = ((df["has_pool"] + df["has_waterfront"] + df["has_garage"]) >= 2).astype(int)

        # ── Interacciones con amenidades ───────────────────────────────────
        df["school_x_area"]      = df["avg_school_rating"].fillna(0) * df["livingArea"]
        df["waterfront_x_area"]  = df["has_waterfront"] * df["livingArea"]
        df["pool_x_area"]        = df["has_pool"] * df["livingArea"]

        # ── Ratios financieros ─────────────────────────────────────────────
        df["tax_rate_x_value"]   = df["propertyTaxRate"].fillna(0) * df["taxAssessedValue"].fillna(0)
        df["assessed_ratio"]     = df["taxAssessedValue"] / (df["last_listing_price"].replace(0, np.nan))

    # ── Target encoding por zipcode (K-Fold en train) ─────────────────────
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    tr["zip_median_price_enc"]   = np.nan
    tr["zip_mean_price_enc"]     = np.nan

    for fold, (idx_tr, idx_val) in enumerate(kf.split(tr)):
        fold_map_med = tr.iloc[idx_tr].groupby("zipcode")[TARGET].median()
        fold_map_avg = tr.iloc[idx_tr].groupby("zipcode")[TARGET].mean()
        tr.loc[tr.index[idx_val], "zip_median_price_enc"] = tr.iloc[idx_val]["zipcode"].map(fold_map_med)
        tr.loc[tr.index[idx_val], "zip_mean_price_enc"]   = tr.iloc[idx_val]["zipcode"].map(fold_map_avg)

    # En test: usar mapa global de train
    global_map_med = tr.groupby("zipcode")[TARGET].median()
    global_map_avg = tr.groupby("zipcode")[TARGET].mean()
    te["zip_median_price_enc"] = te["zipcode"].map(global_map_med).fillna(tr[TARGET].median())
    te["zip_mean_price_enc"]   = te["zipcode"].map(global_map_avg).fillna(tr[TARGET].mean())
    tr["zip_median_price_enc"] = tr["zip_median_price_enc"].fillna(tr[TARGET].median())
    tr["zip_mean_price_enc"]   = tr["zip_mean_price_enc"].fillna(tr[TARGET].mean())

    # ── Target encoding por zip_3digit ────────────────────────────────────
    global_3d_med = tr.groupby("zip_3digit")[TARGET].median()
    te["zip3d_median_enc"] = te["zip_3digit"].map(global_3d_med).fillna(tr[TARGET].median())
    tr["zip3d_median_enc"] = tr["zip_3digit"].map(global_3d_med).fillna(tr[TARGET].median())

    # ── Clúster geoespacial K-Means ───────────────────────────────────────
    coords_tr = tr[["latitude", "longitude"]].fillna(0).values
    coords_te = te[["latitude", "longitude"]].fillna(0).values
    km = KMeans(n_clusters=30, random_state=RANDOM_STATE, n_init=10)
    tr["geo_cluster"] = km.fit_predict(coords_tr)
    te["geo_cluster"] = km.predict(coords_te)

    # ── Ranking de área dentro del zipcode ────────────────────────────────
    tr["area_rank_in_zip"] = tr.groupby("zipcode")["livingArea"].rank(pct=True)
    zip_area_rank = tr.groupby("zipcode")["livingArea"].mean()  # proxy para test
    te["area_rank_in_zip"] = te["zipcode"].map(zip_area_rank).rank(pct=True) / len(te)

    return tr, te


print("Aplicando feature engineering...")
train_fe, test_fe = feature_engineering(train, test)
print(f"✅ Features generadas | Train: {train_fe.shape} | Test: {test_fe.shape}")

# %% [markdown]
# ## 3. Pipeline Tabular — Entrenamiento Multi-Modelo

# %%
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
    test_fe[col]  = test_fe[col].astype("category")

FEATURES_TAB = [f for f in FEATURES_TAB if f in train_fe.columns]
FEATURES_TAB += [c for c in CAT_FEATURES if c in train_fe.columns and c not in FEATURES_TAB]

X = train_fe[FEATURES_TAB]
y = train_fe[TARGET]
X_test = test_fe[FEATURES_TAB]

print(f"Features tabulares totales: {len(FEATURES_TAB)}")


# ── run_kfold (se define siempre) ───────────────────────────────────────────
def run_kfold(model_fn, X, y, X_test, n_folds=N_FOLDS, label="Model", early_stop_rounds=0):
    """Entrena con K-Fold. Soporta early stopping para LGB/XGB."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    oof = np.zeros(len(X)); test_preds = np.zeros(len(X_test)); scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        model = model_fn()
        if early_stop_rounds > 0:
            fp = dict(eval_set=[(X_val, y_val)])
            mn = type(model).__name__
            if "LGBM" in mn:
                fp["callbacks"] = [lgb.early_stopping(early_stop_rounds, verbose=False), lgb.log_evaluation(0)]
            elif "XGB" in mn:
                fp["verbose"] = False
            model.fit(X_tr, y_tr, **fp)
        else:
            model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val); oof[val_idx] = val_pred
        test_preds += model.predict(X_test) / n_folds
        bi = ""
        if hasattr(model, "best_iteration_"): bi = f"  iters={model.best_iteration_}"
        elif hasattr(model, "best_iteration"): bi = f"  iters={model.best_iteration}"
        mae = mean_absolute_error(np.expm1(y_val), np.expm1(val_pred))
        mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(val_pred))
        r2 = r2_score(y_val, val_pred); scores.append(mae)
        print(f"  [{label}] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}  R²={r2:.4f}{bi}")
    print(f"  [{label}] OOF MAE=${np.mean(scores):,.0f} ± {np.std(scores):,.0f}\n")
    return oof, test_preds


if RUN_TABULAR:
    print("▶ LightGBM...")
    lgb_cats = [c for c in CAT_FEATURES if c in FEATURES_TAB]
    def lgbm_fn():
        return lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.01, num_leaves=255,
            min_child_samples=10, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=0.5, random_state=RANDOM_STATE, verbosity=-1,
            categorical_feature=lgb_cats)
    oof_lgb, test_lgb = run_kfold(lgbm_fn, X, y, X_test, label="LightGBM", early_stop_rounds=100)

    print("▶ XGBoost...")
    X_xgb = X.copy(); X_test_xgb = X_test.copy()
    for col in CAT_FEATURES:
        if col in X_xgb.columns:
            X_xgb[col] = X_xgb[col].cat.codes; X_test_xgb[col] = X_test_xgb[col].cat.codes
    def xgb_fn():
        return xgb.XGBRegressor(
            n_estimators=3000, learning_rate=0.01, max_depth=8,
            min_child_weight=1, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=0.5, gamma=0.01,
            random_state=RANDOM_STATE, verbosity=0, tree_method="hist",
            early_stopping_rounds=100)
    oof_xgb, test_xgb = run_kfold(xgb_fn, X_xgb, y, X_test_xgb, label="XGBoost", early_stop_rounds=100)

    print("▶ Ridge...")
    num_cols = [c for c in FEATURES_TAB if c not in CAT_FEATURES]
    X_ridge = X[num_cols].fillna(0); X_test_ridge = X_test[num_cols].fillna(0)
    scaler = StandardScaler()
    X_ridge_sc = scaler.fit_transform(X_ridge); X_test_ridge_sc = scaler.transform(X_test_ridge)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_ridge = np.zeros(len(X)); test_ridge = np.zeros(len(X_test))
    for fold, (tri, vli) in enumerate(kf.split(X_ridge_sc)):
        ridge = Ridge(alpha=10.0); ridge.fit(X_ridge_sc[tri], y.iloc[tri])
        oof_ridge[vli] = ridge.predict(X_ridge_sc[vli])
        test_ridge += ridge.predict(X_test_ridge_sc) / N_FOLDS
        mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_ridge[vli]))
        mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_ridge[vli]))
        print(f"  [Ridge] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")
    print(f"  [Ridge] OOF MAE=${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}  MAPE={mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ridge)):.2%}\n")

    print("✅ Pipeline Tabular completo")
    print(f"  LightGBM OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_lgb)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_lgb)):.2%}")
    print(f"  XGBoost  OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_xgb)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_xgb)):.2%}")
    print(f"  Ridge    OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ridge)):.2%}")
    np.savez(CACHE / "preds_tabular.npz", oof_lgb=oof_lgb, test_lgb=test_lgb,
             oof_xgb=oof_xgb, test_xgb=test_xgb, oof_ridge=oof_ridge, test_ridge=test_ridge)
    print("  💾 Predicciones tabulares guardadas en caché")
else:
    cached = np.load(CACHE / "preds_tabular.npz")
    oof_lgb, test_lgb = cached["oof_lgb"], cached["test_lgb"]
    oof_xgb, test_xgb = cached["oof_xgb"], cached["test_xgb"]
    oof_ridge, test_ridge = cached["oof_ridge"], cached["test_ridge"]
    print("📦 Predicciones tabulares cargadas desde caché")
    print(f"  LightGBM OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_lgb)):,.0f}")
    print(f"  XGBoost  OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_xgb)):,.0f}")
    print(f"  Ridge    OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}")

# %% [markdown]
# ## 4. Pipeline de Texto — DistilBERT (CPU)

# %%
DISTILBERT_CACHE      = CACHE / "distilbert_train.npy"
DISTILBERT_TEST_CACHE = CACHE / "distilbert_test.npy"

def clean_text(text: str) -> str:
    import re
    text = str(text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:1000]

def make_fallback_text(row) -> str:
    beds = int(row.get("bedrooms", 0) or 0)
    baths = int(row.get("bathrooms", 0) or 0)
    area = int(row.get("livingArea", 0) or 0)
    return f"property with {beds} beds and {baths} baths and {area} square feet"

def get_distilbert_embeddings(df: pd.DataFrame, cache_path: Path):
    if cache_path.exists():
        print(f"  Cargando desde caché: {cache_path.name}")
        return np.load(str(cache_path))
    try:
        from transformers import DistilBertTokenizer, DistilBertModel
        import torch
    except ImportError:
        print("  ⚠ 'transformers' no instalado. Usando TF-IDF como fallback.")
        return None
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertModel.from_pretrained("distilbert-base-uncased"); model.eval()
    texts = []
    for _, row in df.iterrows():
        desc = clean_text(str(row.get("description", "") or ""))
        if len(desc.strip()) < 20: desc = make_fallback_text(row)
        texts.append(desc)
    all_embs = []; batch_size = 16
    print(f"  Extrayendo embeddings: {len(texts)} textos | batch={batch_size}")
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad(): out = model(**enc)
        cls = out.last_hidden_state[:, 0, :].numpy(); all_embs.append(cls)
        if (i // batch_size) % 20 == 0: print(f"    {i}/{len(texts)}...")
    embs = np.vstack(all_embs); np.save(str(cache_path), embs)
    print(f"  Guardado: {cache_path.name} | shape={embs.shape}")
    return embs

def get_tfidf_embeddings(df_tr, df_te, n_components=64):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tr_t = df_tr["description"].fillna("").apply(clean_text)
    te_t = df_te["description"].fillna("").apply(clean_text)
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", min_df=3)
    tr_m = tfidf.fit_transform(tr_t); te_m = tfidf.transform(te_t)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    tr_s = svd.fit_transform(tr_m); te_s = svd.transform(te_m)
    print(f"  TF-IDF SVD variance: {svd.explained_variance_ratio_.sum():.2%}")
    return tr_s, te_s

if RUN_TEXT:
    print("\n" + "="*60)
    print("▶ Pipeline Texto — DistilBERT (CPU)")
    print("="*60)
    train_embs_txt = get_distilbert_embeddings(train_fe, DISTILBERT_CACHE)
    test_embs_txt = get_distilbert_embeddings(test_fe, DISTILBERT_TEST_CACHE)
    N_TXT = 64
    if train_embs_txt is None:
        print("  Usando TF-IDF + SVD como fallback...")
        train_embs_txt, test_embs_txt = get_tfidf_embeddings(train_fe, test_fe, n_components=N_TXT)
    else:
        pca_txt = PCA(n_components=N_TXT, random_state=RANDOM_STATE)
        train_embs_txt = pca_txt.fit_transform(train_embs_txt)
        test_embs_txt = pca_txt.transform(test_embs_txt)
        print(f"  PCA texto: 768 → {N_TXT} | var={pca_txt.explained_variance_ratio_.sum():.2%}")
    cols_txt = [f"txt_{i}" for i in range(N_TXT)]
    X_txt = pd.DataFrame(train_embs_txt, columns=cols_txt)
    X_test_txt = pd.DataFrame(test_embs_txt, columns=cols_txt)
    sc_txt = StandardScaler()
    X_txt_sc = sc_txt.fit_transform(X_txt); X_test_txt_sc = sc_txt.transform(X_test_txt)
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    oof_txt = np.zeros(len(X_txt)); test_txt = np.zeros(len(X_test_txt))
    for fold, (tri, vli) in enumerate(kf.split(X_txt_sc)):
        r = Ridge(alpha=50.0); r.fit(X_txt_sc[tri], y.iloc[tri])
        oof_txt[vli] = r.predict(X_txt_sc[vli])
        test_txt += r.predict(X_test_txt_sc) / N_FOLDS
        mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_txt[vli]))
        mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_txt[vli]))
        print(f"  [Texto-Ridge] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")
    print(f"✅ Texto OOF MAE: ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_txt)):.2%}")
    np.savez(CACHE / "preds_text.npz", oof_txt=oof_txt, test_txt=test_txt)
    print("  💾 Predicciones de texto guardadas en caché")
else:
    cached = np.load(CACHE / "preds_text.npz")
    oof_txt, test_txt = cached["oof_txt"], cached["test_txt"]
    print(f"📦 Predicciones de texto cargadas desde caché")
    print(f"  Texto OOF MAE: ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):,.0f}")

# %% [markdown]
# ## 5. Pipeline de Imágenes — ResNet50 + SVR / XGBoost

# %%
IMG_TRAIN_CACHE = CACHE / "resnet50_train.npy"
IMG_TEST_CACHE = CACHE / "resnet50_test.npy"
ZPID_TRAIN_CACHE = CACHE / "img_zpid_train.npy"
ZPID_TEST_CACHE = CACHE / "img_zpid_test.npy"

def extract_resnet_embeddings(meta_df, cache_emb, cache_zpid, max_imgs=3, batch_size=64):
    if cache_emb.exists() and cache_zpid.exists():
        print(f"  Cargando desde caché...")
        return np.load(str(cache_emb)), np.load(str(cache_zpid))
    try:
        import torch; import torchvision.models as tvm; import torchvision.transforms as T
        from PIL import Image
    except ImportError:
        print("  ⚠ torch/torchvision no disponible."); return None, None
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {DEVICE}")
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    backbone = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1], torch.nn.Flatten())
    backbone = backbone.to(DEVICE).eval()
    grouped = meta_df.sort_values("image_index").groupby("zpid").head(max_imgs)
    paths = grouped["image_path"].tolist(); zpids = grouped["zpid"].tolist()
    all_embs = []
    print(f"  Extrayendo embeddings: {len(paths)} imágenes...")
    for i in range(0, len(paths), batch_size):
        batch_t = []
        for p in paths[i:i + batch_size]:
            try: img = Image.open(p).convert("RGB"); batch_t.append(transform(img))
            except Exception: batch_t.append(torch.zeros(3, 224, 224))
        batch_t = torch.stack(batch_t).to(DEVICE)
        with torch.no_grad(): embs = backbone(batch_t).cpu().numpy()
        all_embs.append(embs)
        if i % (batch_size * 20) == 0: print(f"    {i}/{len(paths)}...")
    embs_arr = np.vstack(all_embs); zpids_arr = np.array(zpids)
    np.save(str(cache_emb), embs_arr); np.save(str(cache_zpid), zpids_arr)
    print(f"  Guardado | shape={embs_arr.shape}")
    return embs_arr, zpids_arr

def aggregate_by_zpid(embs, zpids):
    df_e = pd.DataFrame(embs); df_e["zpid"] = zpids
    agg = df_e.groupby("zpid").mean().reset_index()
    return agg["zpid"].values, agg.drop("zpid", axis=1).values

if RUN_IMAGE:
    print("\n" + "="*60)
    print("▶ Pipeline Imágenes — ResNet50 Transfer Learning")
    print("="*60)
    train_meta = pd.read_csv(BASE / "data" / "train_photo_metadata.csv")
    test_meta = pd.read_csv(BASE / "data" / "test_photo_metadata.csv")
    raw_tr_embs, raw_tr_zpids = extract_resnet_embeddings(train_meta, IMG_TRAIN_CACHE, ZPID_TRAIN_CACHE)
    raw_te_embs, raw_te_zpids = extract_resnet_embeddings(test_meta, IMG_TEST_CACHE, ZPID_TEST_CACHE)
    img_ok = raw_tr_embs is not None
    if img_ok:
        tr_zpids_img, tr_embs_img = aggregate_by_zpid(raw_tr_embs, raw_tr_zpids)
        te_zpids_img, te_embs_img = aggregate_by_zpid(raw_te_embs, raw_te_zpids)
        N_IMG = 64
        pca_img = PCA(n_components=N_IMG, random_state=RANDOM_STATE)
        tr_pca = pca_img.fit_transform(tr_embs_img); te_pca = pca_img.transform(te_embs_img)
        print(f"  PCA imagen: 2048 → {N_IMG} | var={pca_img.explained_variance_ratio_.sum():.2%}")
        tr_pca_map = dict(zip(tr_zpids_img, tr_pca))
        te_pca_map = dict(zip(te_zpids_img, te_pca))
        median_vec = np.median(tr_pca, axis=0)
        X_img = np.array([tr_pca_map.get(z, median_vec) for z in train_fe["zpid"].values])
        X_test_img = np.array([te_pca_map.get(z, median_vec) for z in test_fe["zpid"].values])
        print("  ▶ SVR sobre imagen PCA...")
        sc_img = StandardScaler()
        X_img_sc = sc_img.fit_transform(X_img); X_test_img_sc = sc_img.transform(X_test_img)
        kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        oof_img_svr = np.zeros(len(X_img)); test_img_svr = np.zeros(len(X_test_img))
        for fold, (tri, vli) in enumerate(kf.split(X_img_sc)):
            svr = SVR(kernel="rbf", C=10.0, epsilon=0.01)
            svr.fit(X_img_sc[tri], y.iloc[tri])
            oof_img_svr[vli] = svr.predict(X_img_sc[vli])
            test_img_svr += svr.predict(X_test_img_sc) / N_FOLDS
            mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_img_svr[vli]))
            mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_img_svr[vli]))
            print(f"    [Img-SVR] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")
        mae_svr = mean_absolute_error(np.expm1(y), np.expm1(oof_img_svr))
        print("  ▶ XGBoost sobre imagen PCA...")
        cols_img = [f"img_{i}" for i in range(N_IMG)]
        X_img_df = pd.DataFrame(X_img, columns=cols_img)
        X_test_img_df = pd.DataFrame(X_test_img, columns=cols_img)
        def xgb_img_fn():
            return xgb.XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbosity=0, tree_method="hist")
        oof_img_xgb, test_img_xgb = run_kfold(xgb_img_fn, X_img_df, y, X_test_img_df, label="Img-XGB")
        mae_xgb = mean_absolute_error(np.expm1(y), np.expm1(oof_img_xgb))
        if mae_svr <= mae_xgb:
            oof_img = oof_img_svr; test_img = test_img_svr
            print(f"✅ Imagen: SVR ganó (MAE=${mae_svr:,.0f} vs XGB=${mae_xgb:,.0f})")
        else:
            oof_img = oof_img_xgb; test_img = test_img_xgb
            print(f"✅ Imagen: XGBoost ganó (MAE=${mae_xgb:,.0f} vs SVR=${mae_svr:,.0f})")
    else:
        print("⚠ Pipeline imagen no disponible — usando predicción tabular como fallback.")
        oof_img = oof_lgb.copy(); test_img = test_lgb.copy()
    np.savez(CACHE / "preds_image.npz", oof_img=oof_img, test_img=test_img)
    print("  💾 Predicciones de imagen guardadas en caché")
else:
    cached = np.load(CACHE / "preds_image.npz")
    oof_img, test_img = cached["oof_img"], cached["test_img"]
    print(f"📦 Predicciones de imagen cargadas desde caché")
    print(f"  Imagen OOF MAE: ${mean_absolute_error(np.expm1(y), np.expm1(oof_img)):,.0f}")
# %% [markdown]
# ## 6. Ensamble Final — Ridge Stacking OOF

# %%
print("\n" + "="*60)
print("▶ Ensamble Final — Ridge Stacking")
print("="*60)

X_oof_stack  = np.column_stack([oof_lgb, oof_xgb, oof_ridge, oof_txt, oof_img])
X_test_stack = np.column_stack([test_lgb, test_xgb, test_ridge, test_txt, test_img])
stack_names  = ["lgb", "xgb", "ridge_tab", "txt", "img"]

print(f"Stack shape: {X_oof_stack.shape}")
oof_corr = pd.DataFrame(X_oof_stack, columns=stack_names).corr().round(3)
print("\nCorrelación entre modelos (OOF):")
print(oof_corr.to_string())

sc_stack      = StandardScaler()
X_oof_sc      = sc_stack.fit_transform(X_oof_stack)
X_test_sc     = sc_stack.transform(X_test_stack)

kf            = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
oof_ensemble  = np.zeros(len(X_oof_sc))
test_ensemble = np.zeros(len(X_test_sc))

for fold, (tri, vli) in enumerate(kf.split(X_oof_sc)):
    meta = xgb.XGBRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=4,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=0, tree_method="hist"
    )
    meta.fit(X_oof_sc[tri], y.iloc[tri])
    oof_ensemble[vli] = meta.predict(X_oof_sc[vli])
    test_ensemble    += meta.predict(X_test_sc) / N_FOLDS
    mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_ensemble[vli]))
    mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_ensemble[vli]))
    print(f"  [Ensemble] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")

mae_final = mean_absolute_error(np.expm1(y), np.expm1(oof_ensemble))
r2_final  = r2_score(y, oof_ensemble)
mape_final = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ensemble))
print(f"\n✅ Ensemble OOF MAE: ${mae_final:,.0f} | MAPE: {mape_final:.2%} | R²: {r2_final:.4f}")

meta_all = xgb.XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8,
    random_state=RANDOM_STATE, verbosity=0, tree_method="hist"
).fit(X_oof_sc, y)
print("\nImportancia de variables (meta-modelo XGBoost):")
for name, imp in zip(stack_names, meta_all.feature_importances_):
    print(f"  {name:12s}: {imp:.4f}")

# %% [markdown]
# ## 7. Generación de Submissions

# %%
print("\n" + "="*60)
print("▶ Generando Submissions")
print("="*60)

zpid_test = test_fe["zpid"].values

mae_lgb = mean_absolute_error(np.expm1(y), np.expm1(oof_lgb))
mae_xgb_ = mean_absolute_error(np.expm1(y), np.expm1(oof_xgb))
best_tab_preds = test_lgb if mae_lgb <= mae_xgb_ else test_xgb
best_tab_mae   = min(mae_lgb, mae_xgb_)

pd.DataFrame({"zpid": zpid_test, "predicted_price": np.expm1(best_tab_preds)}) \
  .to_csv(SUBMISSIONS / "my_team_tabular.csv", index=False)
print(f"  ✅ my_team_tabular.csv   (OOF MAE≈${best_tab_mae:,.0f})")

pd.DataFrame({"zpid": zpid_test, "predicted_price": np.expm1(test_ensemble)}) \
  .to_csv(SUBMISSIONS / "my_team_ensemble.csv", index=False)
print(f"  ✅ my_team_ensemble.csv  (OOF MAE≈${mae_final:,.0f})")

# ── Submissions OOF sobre Train (para evaluar en competencia) ────────────────
zpid_train = train_fe["zpid"].values
best_tab_oof = oof_lgb if mae_lgb <= mae_xgb_ else oof_xgb

pd.DataFrame({"zpid": zpid_train, "predicted_price": np.expm1(best_tab_oof)}) \
  .to_csv(SUBMISSIONS / "my_team_tabular_oof_train.csv", index=False)
print(f"  ✅ my_team_tabular_oof_train.csv   (OOF MAE≈${best_tab_mae:,.0f})")

pd.DataFrame({"zpid": zpid_train, "predicted_price": np.expm1(oof_ensemble)}) \
  .to_csv(SUBMISSIONS / "my_team_ensemble_oof_train.csv", index=False)
print(f"  ✅ my_team_ensemble_oof_train.csv  (OOF MAE≈${mae_final:,.0f})")

print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                    RESUMEN DE RESULTADOS OOF                  ║
╠═══════════════════════════════════════════════════════════════╣
║  LightGBM Tab : ${mae_lgb:>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_lgb)):>6.2%} ║
║  XGBoost Tab  : ${mae_xgb_:>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_xgb)):>6.2%} ║
║  Ridge Tab    : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ridge)):>6.2%} ║
║  Texto (BERT) : ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_txt)):>6.2%} ║
║  Imagen       : ${mean_absolute_error(np.expm1(y), np.expm1(oof_img)):>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_img)):>6.2%} ║
║  ───────────────────────────────────────────────────────────  ║
║  ENSEMBLE     : ${mae_final:>10,.0f}  | MAPE: {mape_final:>6.2%}  ← USAR ESTE ║
╚═══════════════════════════════════════════════════════════════╝
""")
