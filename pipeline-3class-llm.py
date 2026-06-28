import re
import json
import time
import ollama
import numpy as np
import pandas as pd
from typing import Literal
from pydantic import BaseModel, field_validator
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from nltk.corpus import stopwords
import nltk

# ========================================
# Downloads NLTK necessários
# ========================================
nltk.download("stopwords", quiet=True)
 
# ========================================
# SCHEMA PYDANTIC – Features extraídas pelo LLM
# ========================================
class FeaturasRelato(BaseModel):
    categoria_problema: Literal[
        "cobrança indevida",
        "produto com defeito",
        "atraso na entrega",
        "atendimento ruim",
        "cancelamento",
        "fraude",
        "outro"
    ]
    tom: Literal["neutro", "frustrado", "furioso", "satisfeito"]
    menciona_valor_financeiro: bool
    menciona_prazo: bool
    complexidade: Literal["baixa", "média", "alta"]

    @field_validator("categoria_problema", "tom", "complexidade", mode="before")
    @classmethod
    def lowercase_strip(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

# ========================================
# Configurações globais
# ========================================
CAMINHO_JSON   = "dados2025.json"
EMB_SAVE_PATH  = "embeddings_3class_balanced.npy"
N_SUBAMOSTRA   = 500   # ajuste conforme GPU/CPU disponível
SEED           = 42
OLLAMA_MODEL   = "llama3.2"
 
with open("prompts/extracao_features.txt", "r", encoding="utf-8") as f:
    PROMPT_TEMPLATE = f.read()
 
# ========================================
# Helpers
# ========================================
def mapear_label(nota):
    if nota <= 2:
        return 0
    elif nota == 3:
        return 1
    else:
        return 2
 
def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"http\S+", "", texto)
    return re.sub(r"\s+", " ", texto).strip()
 
def extrair_features(relato: str) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(relato=relato[:1000])
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0},
        )
        raw = resp["message"]["content"].strip()
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        parsed = json.loads(raw[start:end])
        return FeaturasRelato(**parsed).model_dump()
    except Exception:
        return None
 
def features_para_vetor(feat: dict) -> list:
    categorias  = ["cobrança indevida", "produto com defeito", "atraso na entrega",
                   "atendimento ruim", "cancelamento", "fraude", "outro"]
    tons        = ["neutro", "frustrado", "furioso", "satisfeito"]
    complexidades = ["baixa", "média", "alta"]
    vec  = [1 if feat["categoria_problema"] == c else 0 for c in categorias]
    vec += [1 if feat["tom"]               == t else 0 for t in tons]
    vec += [1 if feat["complexidade"]      == c else 0 for c in complexidades]
    vec += [int(feat["menciona_valor_financeiro"])]
    vec += [int(feat["menciona_prazo"])]
    return vec
 
NOME_LABEL = {0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}

# ========================================
# Carregamento e pré-processamento
# ========================================
print("=" * 55)
print("Carregando dataset e embeddings...")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)

# converte nota para numérico e cria label de 3 classes (0=Neg,1=Neutro,2=Pos)
df_full["nota_num"] = pd.to_numeric(df_full["nota"], errors="coerce")
df_full = df_full[df_full["nota_num"].isin([1, 2, 3, 4, 5])].copy()
df_full["label"] = df_full["nota_num"].apply(mapear_label)

grupos = []
for label, grupo in df_full.groupby("label"):
    n = min(5_000, len(grupo))
    grupos.append(grupo.sample(n=n, random_state=SEED))

df = pd.concat(grupos).reset_index(drop=True)
df["texto"] = df["relato"].apply(limpar_texto)

print(df["label"].value_counts().rename(NOME_LABEL).to_string())

embeddings = np.load(EMB_SAVE_PATH)
print(f"Dataset: {len(df):,} linhas | Embeddings: {embeddings.shape}")

# ========================================
# Extração de features com LLM (subamostra)
# ========================================
print(f"\n{'=' * 55}")
print(f"Extraindo features com LLM ({N_SUBAMOSTRA} relatos)...")
print(f"Modelo: {OLLAMA_MODEL}")
print("=" * 55)

grupos_sub = []
for label, grupo in df.groupby("label"):
    n = round(N_SUBAMOSTRA * len(grupo) / len(df))
    grupos_sub.append(grupo.sample(n=n, random_state=SEED))

sub_idx = pd.concat(grupos_sub).index.tolist()

features_list = []
erros = 0
inicio = time.time()

for i, idx in enumerate(sub_idx):
    feat = extrair_features(df.loc[idx, "texto"])
    if feat:
        features_list.append({"df_idx": idx, **feat})
    else:
        erros += 1

    if (i + 1) % 10 == 0:
        decorrido = time.time() - inicio
        media = decorrido / (i + 1)
        restante = media * (len(sub_idx) - (i + 1))
        print(f"  {i+1}/{len(sub_idx)} | erros: {erros} | {media:.1f}s/req | ~{restante/60:.1f}min restantes")

    time.sleep(0.05)

print(f"\nExtraídos: {len(features_list)} | Erros/ignorados: {erros}")

df_feat = pd.DataFrame(features_list).set_index("df_idx")

idxs_validos = df_feat.index.tolist()
emb_sub = embeddings[idxs_validos]
labels_sub = df.loc[idxs_validos, "label"].values
feat_vecs = np.array([features_para_vetor(row) for row in df_feat.to_dict("records")])

# ========================================
# TF-IDF sobre a subamostra
# ========================================
print(f"\n{'=' * 60}")
print("Construindo TF-IDF sobre a subamostra...")
print("=" * 60)
 
stop_pt = stopwords.words("portuguese")
tfidf   = TfidfVectorizer(
    max_features=15_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words=stop_pt,
    min_df=2,
)
X_tfidf_sub = tfidf.fit_transform(textos_sub).toarray()
print(f"Vocabulário TF-IDF (subamostra): {X_tfidf_sub.shape[1]:,} termos")

# ========================================
# Comparação – 4 configurações
# ========================================
print(f"\n{'=' * 60}")
print("Comparação – Embedding / TF-IDF / LLM features / Combinações")
print("=" * 60)
 
configuracoes = {
    "Embedding puro"                   : emb_sub,
    "TF-IDF puro"                      : X_tfidf_sub,
    "Embedding + LLM features"         : np.hstack([emb_sub,      feat_vecs]),
    "TF-IDF   + LLM features"          : np.hstack([X_tfidf_sub,  feat_vecs]),
    "Embedding + TF-IDF"               : np.hstack([emb_sub,      X_tfidf_sub]),
    "Embedding + TF-IDF + LLM features": np.hstack([emb_sub,      X_tfidf_sub, feat_vecs]),
}
 
resultados = {}
for nome, X in configuracoes.items():
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels_sub, test_size=0.2, random_state=SEED, stratify=labels_sub
    )
    clf = LogisticRegression(max_iter=1_000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1     = f1_score(y_te, y_pred, average="weighted")
    resultados[nome] = round(f1, 4)
 
    print(f"\n--- {nome} (F1 weighted: {f1:.4f}) ---")
    print(classification_report(
        y_te, y_pred,
        target_names=["Negativo", "Neutro", "Positivo"],
        zero_division=0,
    ))
 
# ========================================
# Resumo final de F1
# ========================================
print("=" * 60)
print("RESUMO FINAL – F1 Weighted por configuração")
print("=" * 60)
for nome, f1 in sorted(resultados.items(), key=lambda x: -x[1]):
    print(f"  {nome:<45}: {f1:.4f}")
 
# ========================================
# Distribuição das features LLM
# ========================================
print("\n" + "=" * 60)
print("Distribuição das features extraídas pelo LLM")
print("=" * 60)
for col in ["categoria_problema", "tom", "complexidade"]:
    print(f"\n{col}:")
    print(df_feat[col].value_counts().to_string())
 
# ========================================
# TF-IDF: termos mais discriminativos por classe
# (usando configuração TF-IDF puro para interpretabilidade)
# ========================================
print("\n" + "=" * 60)
print("TF-IDF – termos mais discriminativos por classe (Regressão Logística)")
print("=" * 60)
 
X_tr_i, X_te_i, y_tr_i, y_te_i = train_test_split(
    X_tfidf_sub, labels_sub, test_size=0.2, random_state=SEED, stratify=labels_sub
)
clf_interp = LogisticRegression(max_iter=1_000, random_state=SEED)
clf_interp.fit(X_tr_i, y_tr_i)
 
TOP_TERMOS   = 15
feature_names = np.array(tfidf.get_feature_names_out())
classes_nomes = ["Negativo", "Neutro", "Positivo"]
coef          = clf_interp.coef_
 
for i, nome_classe in enumerate(classes_nomes):
    top_idx    = np.argsort(coef[i])[-TOP_TERMOS:][::-1]
    termos_top = feature_names[top_idx]
    pesos_top  = coef[i][top_idx]
    print(f"\nClasse: {nome_classe} – top {TOP_TERMOS} termos com maior coeficiente")
    print(f"{'Rank':<6} {'Termo':<35} {'Coef':>8}")
    print("-" * 51)
    for rank, (termo, peso) in enumerate(zip(termos_top, pesos_top), 1):
        print(f"{rank:<6} {termo:<35} {peso:>8.4f}")