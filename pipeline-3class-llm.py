import re
import json
import time
import pandas as pd
import numpy as np
import ollama
from pydantic import BaseModel, field_validator
from typing import Literal
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

# SCHEMA PYDANTIC
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

# config
CAMINHO_JSON = "dados2025.json"
EMB_SAVE_PATH = "embeddings_3class_full.npy" # gerado na etapa 3
N_SUBAMOSTRA = 500 # linhas para extração LLM
# N_SUBAMOSTRA = 10 # linhas para extração LLM
SEED = 42
OLLAMA_MODEL = "llama3.2"

PROMPT_TEMPLATE = """Você é um analisador de reclamações de consumidores brasileiros.
Analise o relato abaixo e responda APENAS com um JSON válido, sem explicações.

Relato:
\"\"\"{relato}\"\"\"

Responda com este schema exato:
{{
  "categoria_problema": "cobrança indevida" | "produto com defeito" | "atraso na entrega" | "atendimento ruim" | "cancelamento" | "fraude" | "outro",
  "tom": "neutro" | "frustrado" | "furioso" | "satisfeito",
  "menciona_valor_financeiro": true | false,
  "menciona_prazo": true | false,
  "complexidade": "baixa" | "média" | "alta"
}}"""

def extrair_features(relato: str) -> dict | None:
    # chama o Ollama e valida a resposta com Pydantic.
    prompt = PROMPT_TEMPLATE.format(relato=relato[:1000])  # trunca pra não estourar contexto
    try:
        resp = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}
        )
        raw = resp["message"]["content"].strip()

        # extrai só o JSON caso o modelo adicione texto extra
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        parsed = json.loads(raw[start:end])
        features = FeaturasRelato(**parsed)
        return features.model_dump()

    except Exception as e:
        return None

def features_para_vetor(feat: dict) -> list:
    # converte o dict Pydantic em vetor numérico para o classificador.
    categorias = [
        "cobrança indevida", "produto com defeito", "atraso na entrega",
        "atendimento ruim", "cancelamento", "fraude", "outro"
    ]
    tons = ["neutro", "frustrado", "furioso", "satisfeito"]
    complexidades = ["baixa", "média", "alta"]

    vec = []
    vec += [1 if feat["categoria_problema"] == c else 0 for c in categorias]
    vec += [1 if feat["tom"] == t else 0 for t in tons]
    vec += [1 if feat["complexidade"] == c else 0 for c in complexidades]
    vec += [int(feat["menciona_valor_financeiro"])]
    vec += [int(feat["menciona_prazo"])]
    return vec

print("=" * 55)
print("Carregando dataset e embeddings...")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)

# converte nota para numérico e cria label de 3 classes (0=Neg,1=Neutro,2=Pos)
df_full["nota_num"] = pd.to_numeric(df_full.get("nota", None), errors="coerce")
df_full = df_full[df_full["nota_num"].isin([1, 2, 3, 4, 5])].copy()

def mapear_label(nota):
    if nota <= 2:
        return 0
    elif nota == 3:
        return 1
    else:
        return 2

df_full["label"] = df_full["nota_num"].apply(mapear_label)

grupos = []
for label, grupo in df_full.groupby("label"):
    n = round(N_SUBAMOSTRA * len(grupo) / len(df_full))
    grupos.append(grupo.sample(n=n, random_state=SEED))

df = pd.concat(grupos).reset_index(drop=True)

def limpar_texto(texto):
    if not isinstance(texto, str):
        return ""
    texto = re.sub(r"http\S+", "", texto)
    return re.sub(r"\s+", " ", texto).strip()

df["texto"] = df["relato"].apply(limpar_texto)
# df["label"] = (df["status"] == "Resolvido").astype(int)
print(df["label"].value_counts().rename({0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}).to_string())

embeddings = np.load(EMB_SAVE_PATH)
print(f"Dataset: {len(df):,} linhas | Embeddings: {embeddings.shape}")

print(f"\n{'=' * 55}")
print(f"Extraindo features com LLM ({N_SUBAMOSTRA} relatos)...")
print(f"Modelo: {OLLAMA_MODEL}")
print("=" * 55)

sub_idx = (
    df.groupby("label", group_keys=False)
    .apply(lambda x: x.sample(frac=N_SUBAMOSTRA / len(df), random_state=SEED))
    .index.tolist()
)

features_list = []
erros = 0
for i, idx in enumerate(sub_idx):
    feat = extrair_features(df.loc[idx, "texto"])
    if feat:
        features_list.append({"df_idx": idx, **feat})
    else:
        erros += 1

    if (i + 1) % 50 == 0:
        print(f"  {i + 1}/{len(sub_idx)} | erros: {erros}")
    time.sleep(0.05) # respira entre chamadas

print(f"\nExtraídos: {len(features_list)} | Erros/ignorados: {erros}")

df_feat = pd.DataFrame(features_list).set_index("df_idx")

idxs_validos = df_feat.index.tolist()
emb_sub = embeddings[idxs_validos]
labels_sub = df.loc[idxs_validos, "label"].values
feat_vecs = np.array([features_para_vetor(row) for row in df_feat.to_dict("records")])

X_combined = np.hstack([emb_sub, feat_vecs])
X_emb_only = emb_sub

print(f"\n{'=' * 55}")
print("Comparação — baseline vs. enriquecido com LLM")
print("=" * 55)

for nome, X in [("Embedding puro", X_emb_only), ("Embedding + LLM features", X_combined)]:
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels_sub, test_size=0.2, random_state=SEED, stratify=labels_sub
    )
    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    f1 = f1_score(y_te, y_pred, average="weighted")
    print(f"\n--- {nome} (F1 weighted: {f1:.4f}) ---")
    print(classification_report(
        y_te,
        y_pred,
        target_names=["Negativo", "Neutro", "Positivo"],
        zero_division=0,
    ))

print("=" * 55)
print("Distribuição das features extraídas pelo LLM")
print("=" * 55)
for col in ["categoria_problema", "tom", "complexidade"]:
    # print(f"\n{col}:")
    print(f"\n", df_feat[col].value_counts().to_string())
