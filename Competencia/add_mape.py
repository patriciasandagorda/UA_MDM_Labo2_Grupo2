import re

with open('multimodal_ensemble.py', 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace(
    'from sklearn.metrics import mean_absolute_error, r2_score',
    'from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error'
)

# En run_kfold
content = content.replace(
    'mae = mean_absolute_error(np.expm1(y_val), np.expm1(val_pred))\n        r2  = r2_score(y_val, val_pred)\n        scores.append(mae)\n        print(f"  [{label}] Fold {fold+1}: MAE=${mae:,.0f}  R²={r2:.4f}")',
    'mae = mean_absolute_error(np.expm1(y_val), np.expm1(val_pred))\n        mape = mean_absolute_percentage_error(np.expm1(y_val), np.expm1(val_pred))\n        r2  = r2_score(y_val, val_pred)\n        scores.append(mae)\n        print(f"  [{label}] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}  R²={r2:.4f}")'
)

# En Ridge
content = content.replace(
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_ridge[vli]))\n    print(f"  [Ridge] Fold {fold+1}: MAE=${mae:,.0f}")',
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_ridge[vli]))\n    mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_ridge[vli]))\n    print(f"  [Ridge] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")'
)

content = content.replace(
    'print(f"  [Ridge] OOF MAE=${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}\\n")',
    'print(f"  [Ridge] OOF MAE=${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}  MAPE={mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ridge)):.2%}\\n")'
)

# Resumen Tabular
content = content.replace(
    'print(f"  LightGBM OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_lgb)):,.0f}")',
    'print(f"  LightGBM OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_lgb)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_lgb)):.2%}")'
)
content = content.replace(
    'print(f"  XGBoost  OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_xgb)):,.0f}")',
    'print(f"  XGBoost  OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_xgb)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_xgb)):.2%}")'
)
content = content.replace(
    'print(f"  Ridge    OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}")',
    'print(f"  Ridge    OOF MAE : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ridge)):.2%}")'
)

# En Texto
content = content.replace(
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_txt[vli]))\n    print(f"  [Texto-Ridge] Fold {fold+1}: MAE=${mae:,.0f}")',
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_txt[vli]))\n    mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_txt[vli]))\n    print(f"  [Texto-Ridge] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")'
)
content = content.replace(
    'print(f"✅ Texto OOF MAE: ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):,.0f}")',
    'print(f"✅ Texto OOF MAE: ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):,.0f}  MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_txt)):.2%}")'
)

# En Imagen SVR
content = content.replace(
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_img_svr[vli]))\n        print(f"    [Img-SVR] Fold {fold+1}: MAE=${mae:,.0f}")',
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_img_svr[vli]))\n        mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_img_svr[vli]))\n        print(f"    [Img-SVR] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")'
)
content = content.replace(
    'print(f"  [Img-SVR] OOF MAE=${mae_svr:,.0f}")',
    'print(f"  [Img-SVR] OOF MAE=${mae_svr:,.0f}  MAPE={mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_img_svr)):.2%}")'
)

# En Ensemble
content = content.replace(
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_ensemble[vli]))\n    print(f"  [Ensemble] Fold {fold+1}: MAE=${mae:,.0f}")',
    'mae = mean_absolute_error(np.expm1(y.iloc[vli]), np.expm1(oof_ensemble[vli]))\n    mape = mean_absolute_percentage_error(np.expm1(y.iloc[vli]), np.expm1(oof_ensemble[vli]))\n    print(f"  [Ensemble] Fold {fold+1}: MAE=${mae:,.0f}  MAPE={mape:.2%}")'
)
content = content.replace(
    'print(f"\\n✅ Ensemble OOF MAE: ${mae_final:,.0f} | R²: {r2_final:.4f}")',
    'mape_final = mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ensemble))\nprint(f"\\n✅ Ensemble OOF MAE: ${mae_final:,.0f} | MAPE: {mape_final:.2%} | R²: {r2_final:.4f}")'
)

# En Resumen Final
content = content.replace(
    'print(f"""\n╔══════════════════════════════════════════════════╗\n║           RESUMEN DE RESULTADOS OOF              ║\n╠══════════════════════════════════════════════════╣\n║  LightGBM Tab : ${mae_lgb:>10,.0f}                 ║\n║  XGBoost Tab  : ${mae_xgb_:>10,.0f}                 ║\n║  Ridge Tab    : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):>10,.0f}                 ║\n║  Texto (BERT) : ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):>10,.0f}                 ║\n║  Imagen       : ${mean_absolute_error(np.expm1(y), np.expm1(oof_img)):>10,.0f}                 ║\n║  ────────────────────────────────────────────── ║\n║  ENSEMBLE     : ${mae_final:>10,.0f}  ← USAR ESTE   ║\n╚══════════════════════════════════════════════════╝\n""")',
    'print(f"""\n╔═══════════════════════════════════════════════════════════════╗\n║                    RESUMEN DE RESULTADOS OOF                  ║\n╠═══════════════════════════════════════════════════════════════╣\n║  LightGBM Tab : ${mae_lgb:>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_lgb)):>6.2%} ║\n║  XGBoost Tab  : ${mae_xgb_:>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_xgb)):>6.2%} ║\n║  Ridge Tab    : ${mean_absolute_error(np.expm1(y), np.expm1(oof_ridge)):>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_ridge)):>6.2%} ║\n║  Texto (BERT) : ${mean_absolute_error(np.expm1(y), np.expm1(oof_txt)):>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_txt)):>6.2%} ║\n║  Imagen       : ${mean_absolute_error(np.expm1(y), np.expm1(oof_img)):>10,.0f}  | MAPE: {mean_absolute_percentage_error(np.expm1(y), np.expm1(oof_img)):>6.2%} ║\n║  ───────────────────────────────────────────────────────────  ║\n║  ENSEMBLE     : ${mae_final:>10,.0f}  | MAPE: {mape_final:>6.2%}  ← USAR ESTE ║\n╚═══════════════════════════════════════════════════════════════╝\n""")'
)

with open('multimodal_ensemble.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Archivos actualizados con MAPE')
