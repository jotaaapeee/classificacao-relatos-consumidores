import json
import pandas as pd
import numpy as np

CAMINHO_JSON = "dados2025.json"

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.DataFrame(data)
df["nota_num"] = pd.to_numeric(df["nota"], errors="coerce")
df = df[df["nota_num"].isin([1, 2, 3, 4, 5])].copy()

df["chars"]    = df["relato"].str.len()
df["palavras"] = df["relato"].str.split().str.len()

def mapear_label(nota):
    if nota <= 2:   return "Negativo"
    elif nota == 3: return "Neutro"
    else:           return "Positivo"

df["classe"] = df["nota_num"].apply(mapear_label)

SEP = "=" * 55

# ── 1. Visão geral
print(SEP)
print("VISÃO GERAL")
print(SEP)
print(f"Total de registros:      {len(df):,}")
print(f"Empresas únicas:         {df['empresa'].nunique():,}")
print(f"Período:                 {df['data'].min()} → {df['data'].max()}")
print(f"Colunas:                 {', '.join(df.columns.tolist())}")

# ── 2. Distribuição de classes
print(f"\n{SEP}")
print("DISTRIBUIÇÃO DE CLASSES")
print(SEP)
dist = df["classe"].value_counts()
for classe, n in dist.items():
    print(f"  {classe:<10} {n:>7,}  ({n/len(df)*100:.1f}%)")

# ── 3. Comprimento dos relatos
print(f"\n{SEP}")
print("COMPRIMENTO DOS RELATOS (chars)")
print(SEP)
stats_chars = df["chars"].describe(percentiles=[.25, .5, .75, .90, .95])
for k, v in stats_chars.items():
    print(f"  {k:<10} {v:>8.1f}")

print(f"\n{'─'*30}")
print("POR CLASSE (média de chars):")
print(df.groupby("classe")["chars"].mean().round(1).to_string())

# ── 4. Palavras por relato
print(f"\n{SEP}")
print("PALAVRAS POR RELATO")
print(SEP)
stats_palavras = df["palavras"].describe(percentiles=[.25, .5, .75, .90, .95])
for k, v in stats_palavras.items():
    print(f"  {k:<10} {v:>8.1f}")

print(f"\n{'─'*30}")
print("POR CLASSE (média de palavras):")
print(df.groupby("classe")["palavras"].mean().round(1).to_string())

# ── 5. Top 10 empresas
print(f"\n{SEP}")
print("TOP 10 EMPRESAS MAIS RECLAMADAS")
print(SEP)
top = df["empresa"].value_counts().head(10)
for empresa, n in top.items():
    print(f"  {empresa:<40} {n:>6,}  ({n/len(df)*100:.1f}%)")

# ── 6. Distribuição de notas
print(f"\n{SEP}")
print("DISTRIBUIÇÃO DE NOTAS (1–5)")
print(SEP)
notas = df["nota_num"].value_counts().sort_index()
for nota, n in notas.items():
    barra = "█" * int(n / len(df) * 50)
    print(f"  Nota {int(nota)}  {barra:<50} {n:>7,}  ({n/len(df)*100:.1f}%)")

# ── 7. Relatos muito curtos / muito longos
print(f"\n{SEP}")
print("RELATOS EXTREMOS")
print(SEP)
print(f"  Muito curtos (<50 chars):    {(df['chars'] < 50).sum():,}")
print(f"  Muito longos (>2000 chars):  {(df['chars'] > 2000).sum():,}")
print(f"  Acima do truncamento (1000): {(df['chars'] > 1000).sum():,}  ({(df['chars'] > 1000).mean()*100:.1f}%)")

print(f"\n{SEP}")
print("Análise concluída.")
print(SEP)