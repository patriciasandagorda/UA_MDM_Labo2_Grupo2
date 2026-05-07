import pandas as pd
import numpy as np

files = ['ensamblado_p0.csv','ensamblado_p25.csv','ensamblado_p50.csv','ensamblado_p75.csv','ensamblad_p100.csv']
labels = ['P0 (peor)', 'P25', 'P50 (mediana)', 'P75', 'P100 (mejor)']
base = r'C:\Users\keive\Documents\Archivos\4_Maestria_Datascience\Laboratorio_II\Competencia Labo2\Corridas en train\Primera'

all_dfs = {}
for f, label in zip(files, labels):
    df = pd.read_csv(f'{base}\\{f}')
    all_dfs[label] = df
    print(f'\n{"="*60}')
    print(f'  {label}: {f} | Rows: {len(df)}')
    print(f'{"="*60}')
    
    err = df['Pred Error (%)'].abs()
    signed_err = df['Pred Error (%)']
    print(f'  Avg |Pred Error|: {err.mean():.2f}%  Median: {err.median():.2f}%')
    print(f'  Signed Error mean: {signed_err.mean():+.2f}% (sesgo)')
    print(f'  Error percentiles: p25={err.quantile(.25):.1f}%  p50={err.quantile(.5):.1f}%  p75={err.quantile(.75):.1f}%  p95={err.quantile(.95):.1f}%')
    
    bids = df[df['Decision'].str.startswith('Bid', na=False)]
    won = df[df['Won?'] == 'Yes']
    skip = df[df['Decision'].str.startswith('Skip', na=False)]
    print(f'  Total: {len(df)} | Bids: {len(bids)} | Won: {len(won)} | Skipped: {len(skip)}')
    
    if len(won) > 0:
        profit = pd.to_numeric(won['Profit'], errors='coerce')
        total_profit = profit.sum()
        losses = profit[profit < 0]
        gains = profit[profit >= 0]
        print(f'  Total Profit (won): ${total_profit:,.0f}  Avg: ${profit.mean():,.0f}')
        print(f'  Profitable: {len(gains)} | Loss: {len(losses)}')
        if len(losses) > 0:
            print(f'  Total losses: ${losses.sum():,.0f}  Avg loss: ${losses.mean():,.0f}')

# Analisis de errores grandes
print(f'\n\n{"="*60}')
print(f'  ANALISIS DE ERRORES EXTREMOS (P50 - mediana)')
print(f'{"="*60}')

df_med = all_dfs['P50 (mediana)']
df_med['abs_error_pct'] = df_med['Pred Error (%)'].abs()
df_med['true_val'] = pd.to_numeric(df_med['True Value'], errors='coerce')
df_med['pred_val'] = pd.to_numeric(df_med['Prediction'], errors='coerce')

# Peores predicciones
worst = df_med.nlargest(20, 'abs_error_pct')
print(f'\n  Top 20 peores predicciones:')
for _, row in worst.iterrows():
    print(f'    zpid={row["zpid"]}  Real=${row["true_val"]:,.0f}  Pred=${row["pred_val"]:,.0f}  Error={row["Pred Error (%)"]:.1f}%')

# Error por rango de precio
df_med['price_bucket'] = pd.cut(df_med['true_val'], 
    bins=[0, 100000, 250000, 500000, 750000, 1000000, float('inf')],
    labels=['<100K', '100-250K', '250-500K', '500-750K', '750K-1M', '>1M'])
print(f'\n  Error por rango de precio:')
bucket_stats = df_med.groupby('price_bucket', observed=True).agg(
    count=('abs_error_pct', 'count'),
    mean_error=('abs_error_pct', 'mean'),
    median_error=('abs_error_pct', 'median'),
    p75_error=('abs_error_pct', lambda x: x.quantile(0.75))
).round(2)
print(bucket_stats.to_string())

# Sesgo por rango
print(f'\n  Sesgo (error con signo) por rango de precio:')
signed_stats = df_med.groupby('price_bucket', observed=True)['Pred Error (%)'].agg(['mean', 'median']).round(2)
signed_stats.columns = ['sesgo_mean', 'sesgo_median']
print(signed_stats.to_string())

# Proporcion de sobre vs sub estimacion
over = (df_med['Pred Error (%)'] > 0).sum()
under = (df_med['Pred Error (%)'] < 0).sum()
print(f'\n  Sobreestima: {over} ({over/len(df_med):.1%}) | Subestima: {under} ({under/len(df_med):.1%})')
