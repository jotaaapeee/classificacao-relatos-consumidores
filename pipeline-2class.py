import os
import re
import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

# CONFIG
CAMINHO_JSON = "dados2025.json"
N_AMOSTRAS = 10_000 # para rodar na apresentacao
SEED = 42
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMB_SAVE_PATH = "embeddings_10k.npy"
# EMB_SAVE_PATH = "embeddings_full.npy"

print("=" * 55)
print("ETAPA 1 — Carregamento e amostragem")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)
print(f"Dataset completo: {len(df_full):,} registros")
print(df_full["status"].value_counts().to_string())

# N_AMOSTRAS = len(df_full)

# Amostragem estratificada
grupos = []
for status, grupo in df_full.groupby("status"):
    n = round(N_AMOSTRAS * len(grupo) / len(df_full))
    grupos.append(grupo.sample(n=n, random_state=SEED))

df = pd.concat(grupos).reset_index(drop=True)

print(f"\nAmostra: {len(df):,} registros")
print(df["status"].value_counts().to_string())

print("\n" + "=" * 55)
print("ETAPA 2 — Pré-processamento")
print("=" * 55)

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"http\S+", "", texto) # remove URLs
    texto = re.sub(r"\s+", " ", texto).strip() # normaliza espaços
    return texto

df["texto"] = df["relato"].apply(limpar_texto)
df["label"] = (df["status"] == "Resolvido").astype(int) # 1=Resolvido, 0=Não Resolvido

# Checagem
vazios = (df["texto"].str.len() == 0).sum()
print(f"Textos vazios após limpeza: {vazios}")
print(f"Tamanho médio do relato: {df['texto'].str.len().mean():.0f} chars")
print(f"\nDistribuição de labels:")
print(df["label"].value_counts().rename({1: "Resolvido (1)", 0: "Não Resolvido (0)"}).to_string())

print("\n" + "=" * 55)
print("ETAPA 3 — Embeddings")
print("=" * 55)

if os.path.exists(EMB_SAVE_PATH):
    print(f"Carregando embeddings salvos de '{EMB_SAVE_PATH}'...")
    embeddings = np.load(EMB_SAVE_PATH)
else:
    print(f"Modelo: {MODEL_NAME}")
    print("Vetorizando... (pode demorar alguns minutos no notebook)")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["texto"].tolist(),
        batch_size=64,
        show_progress_bar=True,
    )
    np.save(EMB_SAVE_PATH, embeddings)
    print(f"Embeddings salvos em '{EMB_SAVE_PATH}'")

print(f"Shape dos embeddings: {embeddings.shape}")

print("\n" + "=" * 55)
print("ETAPA 4 — Baseline ML")
print("=" * 55)

X = embeddings
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
print(f"Treino: {len(X_train):,} | Teste: {len(X_test):,}")

modelos = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "LinearSVC": LinearSVC(max_iter=2000, random_state=SEED),
}

resultados_f1 = {}
for nome, clf in modelos.items():
    print(f"\n--- {nome} ---")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred, average="weighted")
    resultados_f1[nome] = round(f1, 4)
    print(classification_report(
        y_test, y_pred,
        target_names=["Não Resolvido", "Resolvido"]
    ))

print("\n" + "=" * 55)
print("RESUMO")
print("=" * 55)
for nome, f1 in resultados_f1.items():
    print(f"  {nome}: {f1}")