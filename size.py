import json
import pandas as pd

CAMINHO_JSON = "dados2025.json"

SEED = 42
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMB_SAVE_PATH = "embeddings_20k.npy"

print("=" * 55)
print("ETAPA 1 — Carregamento e amostragem")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)

N_AMOSTRAS = df_full.shape

print(N_AMOSTRAS)
print("\n")
print(len(df_full))