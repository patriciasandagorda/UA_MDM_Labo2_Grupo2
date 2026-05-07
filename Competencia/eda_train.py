# %%
"""
EDA — Exploración del Dataset de Entrenamiento
===============================================
Ejecutar desde: participant/
  python eda_train.py
O abrir en VSCode/Jupyter como notebook interactivo (# %% = celdas)
"""

# %% [markdown]
# ## 0. Imports y Carga

# %%
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#444466",
    "axes.labelcolor":  "#ccccee",
    "xtick.color":      "#ccccee",
    "ytick.color":      "#ccccee",
    "text.color":       "#e0e0f0",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "DejaVu Sans",
})
PALETTE = ["#7b5ea7", "#e05c97", "#f5a623", "#4ecdc4", "#45b7d1", "#96e072"]

BASE    = Path(".")
OUTDIR  = BASE / "eda_output"
OUTDIR.mkdir(exist_ok=True)

train = pd.read_csv(BASE / "data" / "tabular" / "train_processed.csv")
print(f"Shape: {train.shape}")
print(f"Columnas ({len(train.columns)}): {train.columns.tolist()}")

# %% [markdown]
# ## 1. Resumen General del Dataset

# %%
print("\n=== TIPOS DE DATOS ===")
print(train.dtypes.value_counts())

print("\n=== VALORES NULOS (%) ===")
null_pct = (train.isnull().sum() / len(train) * 100).sort_values(ascending=False)
print(null_pct[null_pct > 0].round(2).to_string())

print("\n=== ESTADÍSTICAS DESCRIPTIVAS (numéricas) ===")
desc = train.describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).T
print(desc[["min","1%","5%","25%","50%","75%","95%","99%","max"]].round(2).to_string())

# %% [markdown]
# ## 2. Distribución del Target: Precio de Venta

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Distribución del Precio de Venta", fontsize=15, fontweight="bold", y=1.02)

price = train["lastSoldPrice_hpi_adjusted"].dropna()
log_p = train["log_price"].dropna()

# Precio original
axes[0].hist(price.clip(upper=price.quantile(0.99)), bins=80,
             color=PALETTE[0], edgecolor="none", alpha=0.85)
axes[0].set_title("Precio Original ($)")
axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
axes[0].set_xlabel("lastSoldPrice_hpi_adjusted")

# Log-precio
axes[1].hist(log_p, bins=80, color=PALETTE[1], edgecolor="none", alpha=0.85)
axes[1].set_title("log1p(Precio) — TARGET")
axes[1].set_xlabel("log_price")

# Boxplot por tipo de propiedad
home_order = (train.groupby("homeType")["lastSoldPrice_hpi_adjusted"]
              .median().sort_values(ascending=False).index.tolist())
train_box = train[train["lastSoldPrice_hpi_adjusted"] < price.quantile(0.99)].copy()
bp_data = [train_box[train_box["homeType"] == ht]["lastSoldPrice_hpi_adjusted"].dropna()
           for ht in home_order]
bp = axes[2].boxplot(bp_data, patch_artist=True, notch=True, vert=True,
                     medianprops=dict(color="white", linewidth=2))
for patch, color in zip(bp["boxes"], PALETTE * 3):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
axes[2].set_xticklabels(home_order, rotation=30, ha="right", fontsize=8)
axes[2].set_title("Precio por Tipo de Propiedad")
axes[2].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))

plt.tight_layout()
plt.savefig(OUTDIR / "01_target_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nPrecio mediano: ${price.median():,.0f}")
print(f"Precio medio:   ${price.mean():,.0f}")
print(f"Precio p95:     ${price.quantile(0.95):,.0f}")
print(f"Precio max:     ${price.max():,.0f}")

# %% [markdown]
# ## 3. Variables Numéricas Clave — Distribuciones

# %%
NUM_COLS = [
    "bedrooms", "bathrooms", "livingArea", "lotAreaValue",
    "yearBuilt", "taxAssessedValue", "propertyTaxRate",
    "avg_school_rating", "hoa_fee_monthly", "photoCount",
    "num_nearby_schools", "min_school_distance",
]
NUM_COLS = [c for c in NUM_COLS if c in train.columns]

fig, axes = plt.subplots(3, 4, figsize=(20, 13))
axes = axes.flatten()
fig.suptitle("Distribuciones de Variables Numéricas", fontsize=14, fontweight="bold")

for i, col in enumerate(NUM_COLS):
    data = train[col].dropna()
    p99  = data.quantile(0.99)
    axes[i].hist(data.clip(upper=p99), bins=60,
                 color=PALETTE[i % len(PALETTE)], edgecolor="none", alpha=0.85)
    axes[i].set_title(col, fontsize=10)
    axes[i].set_xlabel("")
    med = data.median()
    axes[i].axvline(med, color="white", linestyle="--", linewidth=1, alpha=0.7)
    axes[i].text(0.97, 0.93, f"med={med:.1f}", transform=axes[i].transAxes,
                 ha="right", fontsize=8, color="white")

for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(OUTDIR / "02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 4. Variables Categóricas

# %%
CAT_COLS = ["homeType", "zip_3digit"]
CAT_COLS = [c for c in CAT_COLS if c in train.columns]

fig, axes = plt.subplots(1, len(CAT_COLS), figsize=(16, 5))
fig.suptitle("Variables Categóricas — Frecuencia y Precio Mediano", fontsize=13, fontweight="bold")

if len(CAT_COLS) == 1:
    axes = [axes]

for ax, col in zip(axes, CAT_COLS):
    vc = train[col].value_counts().head(20)
    bars = ax.barh(vc.index.astype(str), vc.values, color=PALETTE[0], alpha=0.85)
    ax.set_title(col)
    ax.set_xlabel("Frecuencia")
    ax.invert_yaxis()
    for bar, val in zip(bars, vc.values):
        ax.text(val + max(vc)*0.01, bar.get_y() + bar.get_height()/2,
                f"{val:,}", va="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTDIR / "03_categorical.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Correlación con el Target

# %%
NUM_FEATS = [
    "bedrooms", "bathrooms", "livingArea", "lotAreaValue", "yearBuilt",
    "taxAssessedValue", "last_listing_price", "latest_tax_value",
    "avg_school_rating", "photoCount", "hoa_fee_monthly",
    "has_pool", "has_garage", "has_waterfront",
    "desc_length", "desc_word_count", "num_price_changes",
    "property_age", "bath_to_bed_ratio", "log_living_area",
]
NUM_FEATS = [c for c in NUM_FEATS if c in train.columns]

corrs = (train[NUM_FEATS + ["log_price"]]
         .corr()["log_price"]
         .drop("log_price")
         .sort_values())

fig, ax = plt.subplots(figsize=(10, 9))
fig.suptitle("Correlación de Pearson con log_price", fontsize=13, fontweight="bold")

colors = [PALETTE[1] if v > 0 else PALETTE[0] for v in corrs.values]
bars = ax.barh(corrs.index, corrs.values, color=colors, alpha=0.85)
ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
ax.set_xlabel("Correlación de Pearson")

for bar, val in zip(bars, corrs.values):
    ax.text(val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height()/2,
            f"{val:.3f}", va="center",
            ha="left" if val >= 0 else "right", fontsize=8)

plt.tight_layout()
plt.savefig(OUTDIR / "04_correlation_target.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 6. Mapa de Calor de Correlaciones entre Variables Clave

# %%
KEY_VARS = [
    "log_price", "livingArea", "bedrooms", "bathrooms",
    "taxAssessedValue", "last_listing_price", "avg_school_rating",
    "has_pool", "has_waterfront", "property_age", "photoCount",
    "hoa_fee_monthly",
]
KEY_VARS = [c for c in KEY_VARS if c in train.columns]

corr_mat = train[KEY_VARS].corr()

fig, ax = plt.subplots(figsize=(12, 10))
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(
    corr_mat, mask=mask, annot=True, fmt=".2f",
    cmap="coolwarm", center=0, linewidths=0.3,
    linecolor="#1a1a2e", ax=ax,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 8},
)
ax.set_title("Matriz de Correlación — Variables Clave", fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(OUTDIR / "05_correlation_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 7. Precio vs Variables Clave (Scatter)

# %%
SCATTER_VARS = [
    ("livingArea",        "Área de Vivienda (sqft)"),
    ("taxAssessedValue",  "Valor Catastral ($)"),
    ("avg_school_rating", "Calificación Escolar Promedio"),
    ("yearBuilt",         "Año de Construcción"),
]
SCATTER_VARS = [(c, l) for c, l in SCATTER_VARS if c in train.columns]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()
fig.suptitle("Precio vs Variables Clave", fontsize=14, fontweight="bold")

for ax, (col, label) in zip(axes, SCATTER_VARS):
    sample = train[[col, "log_price", "homeType"]].dropna().sample(
        min(3000, len(train)), random_state=42)
    sc = ax.scatter(
        sample[col].clip(upper=sample[col].quantile(0.99)),
        sample["log_price"],
        alpha=0.3, s=8, c=sample["log_price"],
        cmap="plasma", edgecolors="none"
    )
    ax.set_xlabel(label)
    ax.set_ylabel("log_price")
    ax.set_title(f"log_price vs {col}")
    plt.colorbar(sc, ax=ax, label="log_price", shrink=0.8)

plt.tight_layout()
plt.savefig(OUTDIR / "06_scatter_price_vs_features.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Análisis Geoespacial — Precio por Coordenadas

# %%
geo = train[["latitude", "longitude", "log_price"]].dropna()
geo = geo[(geo["latitude"].between(24, 50)) & (geo["longitude"].between(-125, -65))]
sample_geo = geo.sample(min(5000, len(geo)), random_state=42)

fig, ax = plt.subplots(figsize=(16, 9))
sc = ax.scatter(
    sample_geo["longitude"], sample_geo["latitude"],
    c=sample_geo["log_price"], cmap="plasma",
    alpha=0.5, s=6, edgecolors="none"
)
plt.colorbar(sc, ax=ax, label="log_price (más amarillo = más caro)", shrink=0.8)
ax.set_title("Distribución Geoespacial de Precios (USA)", fontsize=13, fontweight="bold")
ax.set_xlabel("Longitud")
ax.set_ylabel("Latitud")
plt.tight_layout()
plt.savefig(OUTDIR / "07_geo_price_map.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 9. Análisis de Descripciones de Texto

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Análisis del Campo 'description'", fontsize=13, fontweight="bold")

# Longitud de descripciones
axes[0].hist(train["desc_length"].clip(upper=5000), bins=60,
             color=PALETTE[2], edgecolor="none", alpha=0.85)
axes[0].set_title("Longitud de descripción (chars)")
axes[0].axvline(train["desc_length"].median(), color="white", linestyle="--", linewidth=1)

# Boilerplate vs real
bp_counts = train["desc_is_boilerplate"].value_counts()
axes[1].bar(["Real", "Boilerplate"],
            [bp_counts.get(0, 0), bp_counts.get(1, 0)],
            color=[PALETTE[3], PALETTE[4]], alpha=0.85)
axes[1].set_title("Descripciones: Real vs Boilerplate")

# Menciones clave
mention_cols = ["desc_mentions_renovated", "desc_mentions_pool",
                "desc_mentions_view", "desc_mentions_new"]
mention_cols = [c for c in mention_cols if c in train.columns]
mention_vals = [train[c].sum() for c in mention_cols]
labels = [c.replace("desc_mentions_", "") for c in mention_cols]
axes[2].bar(labels, mention_vals, color=PALETTE[:len(labels)], alpha=0.85)
axes[2].set_title("Menciones en descripciones")

plt.tight_layout()
plt.savefig(OUTDIR / "08_text_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

# Correlación descripción con precio
txt_corrs = {}
for col in ["desc_length", "desc_word_count", "desc_is_boilerplate"] + mention_cols:
    if col in train.columns:
        txt_corrs[col] = train[col].corr(train["log_price"])
print("\nCorrelación campos de texto con log_price:")
for k, v in sorted(txt_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f"  {k:35s}: {v:+.4f}")

# %% [markdown]
# ## 10. Variables Booleanas / Indicadores

# %%
BOOL_COLS = [
    "has_pool", "has_garage", "has_waterfront", "has_hoa",
    "tag_price_cut", "tag_new_construction", "tag_foreclosure",
]
BOOL_COLS = [c for c in BOOL_COLS if c in train.columns]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()
fig.suptitle("Impacto de Indicadores Binarios en el Precio", fontsize=13, fontweight="bold")

for i, col in enumerate(BOOL_COLS):
    data0 = train[train[col] == 0]["log_price"].dropna()
    data1 = train[train[col] == 1]["log_price"].dropna()
    axes[i].hist(data0, bins=40, alpha=0.6, color=PALETTE[0],
                 density=True, label=f"{col}=0 (n={len(data0):,})")
    axes[i].hist(data1, bins=40, alpha=0.6, color=PALETTE[1],
                 density=True, label=f"{col}=1 (n={len(data1):,})")
    axes[i].set_title(col, fontsize=9)
    axes[i].legend(fontsize=7)
    delta = data1.median() - data0.median()
    axes[i].set_xlabel(f"Diferencia mediana: {delta:+.3f} log pts")

for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(OUTDIR / "09_binary_features_impact.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 11. Análisis de Valores Faltantes

# %%
null_data = null_pct[null_pct > 0].sort_values(ascending=True)

if len(null_data) > 0:
    fig, ax = plt.subplots(figsize=(10, max(4, len(null_data)*0.35)))
    bars = ax.barh(null_data.index, null_data.values,
                   color=[PALETTE[1] if v > 20 else PALETTE[0] for v in null_data.values],
                   alpha=0.85)
    ax.axvline(5, color="white", linestyle="--", linewidth=0.8, alpha=0.5, label="5%")
    ax.axvline(20, color=PALETTE[1], linestyle="--", linewidth=0.8, alpha=0.5, label="20%")
    ax.set_xlabel("% Valores Nulos")
    ax.set_title("Porcentaje de Valores Faltantes por Variable", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    for bar, val in zip(bars, null_data.values):
        ax.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTDIR / "10_missing_values.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No hay valores faltantes en el dataset.")

# %% [markdown]
# ## 12. Resumen Final

# %%
print("\n" + "="*60)
print("RESUMEN EDA — DATASET DE ENTRENAMIENTO")
print("="*60)
print(f"Filas:                  {len(train):,}")
print(f"Columnas:               {train.shape[1]}")
print(f"Precio mediano:         ${train['lastSoldPrice_hpi_adjusted'].median():,.0f}")
print(f"Precio medio:           ${train['lastSoldPrice_hpi_adjusted'].mean():,.0f}")
print(f"Rango precio:           ${train['lastSoldPrice_hpi_adjusted'].min():,.0f} — ${train['lastSoldPrice_hpi_adjusted'].max():,.0f}")
print(f"Tipos de propiedad:     {train['homeType'].nunique()} únicos")
print(f"Zipcodes únicos:        {train['zipcode'].nunique()}")
print(f"% con descripción real: {(train['desc_is_boilerplate']==0).mean()*100:.1f}%")
print(f"% con pool:             {train['has_pool'].mean()*100:.1f}%")
print(f"% con waterfront:       {train['has_waterfront'].mean()*100:.1f}%")
print(f"% con garage:           {train['has_garage'].mean()*100:.1f}%")
print(f"\nGráficos guardados en: {OUTDIR.resolve()}")
print("="*60)
