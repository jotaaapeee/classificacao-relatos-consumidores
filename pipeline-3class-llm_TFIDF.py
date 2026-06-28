import re
import json
import time
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Literal
from pydantic import BaseModel, field_validator
from dotenv import load_dotenv
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score

load_dotenv()

# ============================================================
# CONFIGURAÇÃO
# ============================================================
CAMINHO_JSON  = "dados2025.json"
N_POR_CLASSE  = 5_000
N_SUBAMOSTRA  = 500
SEED          = 42
GROQ_MODEL    = "llama-3.1-8b-instant"
PROMPT_FILE   = os.path.join(os.path.dirname(__file__), "prompts", "extract_feats.txt")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Prompt combinado — features + zero-shot numa única chamada
try:
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        PROMPT_COMBINADO = f.read()
except FileNotFoundError:
    raise FileNotFoundError(
        f"Prompt file not found: {PROMPT_FILE}. "
        "Certifique-se de que o arquivo existe e está na pasta prompts."
    )

# ============================================================
# SCHEMA PYDANTIC — combinado
# ============================================================
LABEL_MAP = {"negativo": 0, "neutro": 1, "positivo": 2}

class FeaturasRelato(BaseModel):
    classificacao: Literal["Negativo", "Neutro", "Positivo"]
    categoria_problema: Literal[
        "cobrança indevida", "produto com defeito", "atraso na entrega",
        "atendimento ruim", "cancelamento", "fraude", "outro"
    ]
    tom: Literal["neutro", "frustrado", "furioso", "satisfeito"]
    menciona_valor_financeiro: bool
    menciona_prazo: bool
    complexidade: Literal["baixa", "média", "alta"]

    @field_validator("classificacao", mode="before")
    @classmethod
    def normalize_classificacao(cls, v):
        return v.strip().capitalize() if isinstance(v, str) else v

    @field_validator("categoria_problema", "tom", "complexidade", mode="before")
    @classmethod
    def lowercase_strip(cls, v):
        return v.strip().lower() if isinstance(v, str) else v

# ============================================================
# FUNÇÕES
# ============================================================
def limpar_texto(texto):
    if not isinstance(texto, str): return ""
    return re.sub(r"\s+", " ", re.sub(r"http\S+", "", texto)).strip()

def mapear_label(nota):
    if nota <= 2:   return 0
    elif nota == 3: return 1
    else:           return 2

def chamar_llm(relato: str) -> FeaturasRelato | None:
    """Uma única chamada retorna features + zero-shot."""
    prompt = PROMPT_COMBINADO.format(relato=relato[:1000])
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        raw   = resp.choices[0].message.content.strip()
        start = raw.find("{")
        end   = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None
        parsed = json.loads(raw[start:end])
        return FeaturasRelato(**parsed)
    except Exception:
        return None

def features_para_vetor(feat: FeaturasRelato) -> list:
    categorias    = ["cobrança indevida", "produto com defeito", "atraso na entrega",
                     "atendimento ruim", "cancelamento", "fraude", "outro"]
    tons          = ["neutro", "frustrado", "furioso", "satisfeito"]
    complexidades = ["baixa", "média", "alta"]
    vec  = [1 if feat.categoria_problema == c else 0 for c in categorias]
    vec += [1 if feat.tom == t else 0 for t in tons]
    vec += [1 if feat.complexidade == c else 0 for c in complexidades]
    vec += [int(feat.menciona_valor_financeiro)]
    vec += [int(feat.menciona_prazo)]
    return vec

# ============================================================
# ETAPA 1 — Carregamento e undersampling balanceado
# ============================================================
print("=" * 55)
print("ETAPA 1 — Carregamento e undersampling balanceado")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)
df_full["nota_num"] = pd.to_numeric(df_full["nota"], errors="coerce")
df_full = df_full[df_full["nota_num"].isin([1, 2, 3, 4, 5])].copy()
df_full["label"] = df_full["nota_num"].apply(mapear_label)

grupos = []
for label, grupo in df_full.groupby("label"):
    grupos.append(grupo.sample(n=min(N_POR_CLASSE, len(grupo)), random_state=SEED))

df = pd.concat(grupos).sample(frac=1, random_state=SEED).reset_index(drop=True)
df["texto"] = df["relato"].apply(limpar_texto)

print(f"Dataset balanceado: {len(df):,} registros")
print(df["label"].value_counts().rename({0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}).to_string())

# ============================================================
# ETAPA 2 — TF-IDF
# ============================================================
print("\n" + "=" * 55)
print("ETAPA 2 — Vetorização TF-IDF")
print("=" * 55)

tfidf  = TfidfVectorizer(max_features=10_000, sublinear_tf=True, min_df=2, ngram_range=(1, 2))
X_full = tfidf.fit_transform(df["texto"])
y_full = df["label"].values
print(f"Shape TF-IDF: {X_full.shape}")

# ============================================================
# ETAPA 3 — Baseline: TF-IDF + Logistic Regression
# ============================================================
print("\n" + "=" * 55)
print("ETAPA 3 — Baseline: TF-IDF + Logistic Regression")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=SEED, stratify=y_full
)
print(f"Treino: {X_train.shape[0]:,} | Teste: {X_test.shape[0]:,}")

clf_baseline = LogisticRegression(max_iter=1000, random_state=SEED)
clf_baseline.fit(X_train, y_train)
y_pred_base  = clf_baseline.predict(X_test)
f1_baseline  = f1_score(y_test, y_pred_base, average="weighted")

print(classification_report(y_test, y_pred_base,
    target_names=["Negativo", "Neutro", "Positivo"], zero_division=0))
print(f"F1 weighted baseline: {f1_baseline:.4f}")

# ============================================================
# ETAPA 4 — Subamostra + chamada única ao LLM
# ============================================================
print("\n" + "=" * 55)
print(f"ETAPA 4 — Chamada LLM combinada ({N_SUBAMOSTRA} relatos)")
print(f"Modelo: {GROQ_MODEL}")
print("=" * 55)

grupos_sub = []
for label, grupo in df.groupby("label"):
    n = round(N_SUBAMOSTRA * len(grupo) / len(df))
    grupos_sub.append(grupo.sample(n=n, random_state=SEED))

sub_idx = pd.concat(grupos_sub).index.tolist()

resultados_llm = []  # guarda FeaturasRelato por índice
erros          = 0
inicio         = time.time()

for i, idx in enumerate(sub_idx):
    feat = chamar_llm(df.loc[idx, "texto"])
    if feat:
        resultados_llm.append({"df_idx": idx, "obj": feat})
    else:
        erros += 1

    if (i + 1) % 10 == 0:
        decorrido = time.time() - inicio
        media     = decorrido / (i + 1)
        restante  = media * (len(sub_idx) - (i + 1))
        print(f"  {i+1}/{len(sub_idx)} | erros: {erros} | {media:.1f}s/req | ~{restante/60:.1f}min restantes")

    time.sleep(0.1)

print(f"\nExtraídos: {len(resultados_llm)} | Erros: {erros}")

# ============================================================
# ETAPA 5 — TF-IDF + LLM features
# ============================================================
print("\n" + "=" * 55)
print("ETAPA 5 — TF-IDF + LLM features")
print("=" * 55)

idxs_validos = [r["df_idx"] for r in resultados_llm]
labels_sub   = df.loc[idxs_validos, "label"].values
feat_vecs    = np.array([features_para_vetor(r["obj"]) for r in resultados_llm])

X_tfidf_sub = X_full[idxs_validos]
X_combined  = sp.hstack([X_tfidf_sub, sp.csr_matrix(feat_vecs)])

X_tr, X_te, y_tr, y_te = train_test_split(
    X_combined, labels_sub, test_size=0.2, random_state=SEED, stratify=labels_sub
)
clf_feat    = LogisticRegression(max_iter=1000, random_state=SEED)
clf_feat.fit(X_tr, y_tr)
y_pred_feat = clf_feat.predict(X_te)
f1_features = f1_score(y_te, y_pred_feat, average="weighted")

print(classification_report(y_te, y_pred_feat,
    target_names=["Negativo", "Neutro", "Positivo"], zero_division=0))
print(f"F1 weighted: {f1_features:.4f}")

# salva features
df_feat = pd.DataFrame([
    {"df_idx": r["df_idx"], **r["obj"].model_dump()} for r in resultados_llm
]).set_index("df_idx")
df_feat.to_csv("features_llm.csv")
print("Features salvas em 'features_llm.csv'")

# ============================================================
# ETAPA 6 — Zero-shot (extraído da mesma chamada)
# ============================================================
print("\n" + "=" * 55)
print("ETAPA 6 — Zero-shot LLM (sem treinamento)")
print("=" * 55)

y_true_zero = labels_sub
y_pred_zero = np.array([LABEL_MAP[r["obj"].classificacao.lower()] for r in resultados_llm])
f1_zeroshot = f1_score(y_true_zero, y_pred_zero, average="weighted")

print(classification_report(y_true_zero, y_pred_zero,
    target_names=["Negativo", "Neutro", "Positivo"], zero_division=0))
print(f"F1 weighted: {f1_zeroshot:.4f}")

# ============================================================
# RESUMO COMPARATIVO
# ============================================================
print("\n" + "=" * 55)
print("RESUMO COMPARATIVO")
print("=" * 55)
print(f"  Baseline (TF-IDF + LR):       {f1_baseline:.4f}  (15k relatos)")
print(f"  TF-IDF + LLM features:        {f1_features:.4f}  ({len(idxs_validos)} relatos)")
print(f"  Zero-shot LLM:                {f1_zeroshot:.4f}  ({len(idxs_validos)} relatos)")
print(f"\n  Ganho features LLM vs base:   {f1_features - f1_baseline:+.4f}")
print(f"  Ganho zero-shot vs base:      {f1_zeroshot - f1_baseline:+.4f}")

print("\n" + "=" * 55)
print("Distribuição das features extraídas")
print("=" * 55)
for col in ["categoria_problema", "tom", "complexidade"]:
    print(f"\n{col}:")
    print(df_feat[col].value_counts().to_string())