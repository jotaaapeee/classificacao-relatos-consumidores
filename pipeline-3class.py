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

# config
CAMINHO_JSON = "dados2025.json"
N_AMOSTRAS = 10_000
SEED = 42
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
EMB_SAVE_PATH = "embeddings_3class_20k.npy"

print("=" * 55)
print("ETAPA 1 — Carregamento e amostragem")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)

# converte nota para numérico e descarta não numéricos e outliers
df_full["nota_num"] = pd.to_numeric(df_full["nota"], errors="coerce")
df_full = df_full[df_full["nota_num"].isin([1, 2, 3, 4, 5])].copy()

# cria label de 3 classes
def mapear_label(nota):
    if nota <= 2:
        return 0  # Negativo
    elif nota == 3:
        return 1  # Neutro
    else:
        return 2  # Positivo

df_full["label"] = df_full["nota_num"].apply(mapear_label)

print(f"Dataset completo (após filtro): {len(df_full):,} registros")
print(df_full["label"].value_counts().rename({0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}).to_string())

# amostragem estratificada por label
grupos = []
for label, grupo in df_full.groupby("label"):
    n = round(N_AMOSTRAS * len(grupo) / len(df_full))
    grupos.append(grupo.sample(n=n, random_state=SEED))

df = pd.concat(grupos).reset_index(drop=True)

print(f"\nAmostra: {len(df):,} registros")
print(df["label"].value_counts().rename({0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}).to_string())

print("\n" + "=" * 55)
print("ETAPA 2 — Pré-processamento")
print("=" * 55)

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"http\S+", "", texto)
    return re.sub(r"\s+", " ", texto).strip()

df["texto"] = df["relato"].apply(limpar_texto)

vazios = (df["texto"].str.len() == 0).sum()
print(f"Textos vazios após limpeza: {vazios}")
print(f"Tamanho médio do relato: {df['texto'].str.len().mean():.0f} chars")
print(f"\nDistribuição de labels:")
print(df["label"].value_counts().rename({0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}).to_string())

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
        target_names=["Negativo", "Neutro", "Positivo"]
    ))

print("\n" + "=" * 55)
print("RESUMO")
print("=" * 55)
for nome, f1 in resultados_f1.items():
    print(f"  {nome}: {f1}")