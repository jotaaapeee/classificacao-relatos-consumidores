import json
import re
import os
import nltk
import spacy
import numpy as np
import pandas as pd
from collections import Counter
from sentence_transformers import SentenceTransformer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

# ========================================
# Downloads NLTK necessários
# ========================================
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# ========================================
# Configurações globais
# ========================================
CAMINHO_JSON   = "dados2025.json"
N_POR_CLASSE   = 5_000
SEED           = 42
MODEL_NAME     = "paraphrase-multilingual-mpnet-base-v2"
EMB_SAVE_PATH  = "embeddings_3class_balanced.npy"
TOP_N_PALAVRAS = 30   # palavras mais frequentes exibidas na análise NLTK vs spaCy
 
# ========================================
# Rótulo de 3 classes
# ========================================
def mapear_label(nota):
    if nota <= 2:
        return 0   # Negativo
    elif nota == 3:
        return 1   # Neutro
    else:
        return 2   # Positivo
 
NOME_LABEL = {0: "Negativo (0)", 1: "Neutro (1)", 2: "Positivo (2)"}

# ========================================
# ETAPA 1 – Carregamento e undersampling
# ========================================
print("=" * 55)
print("ETAPA 1 - Carregamento e undersampling balanceado")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

df_full = pd.DataFrame(data)

# converte nota para numérico e descarta não numéricos e outliers
df_full["nota_num"] = pd.to_numeric(df_full["nota"], errors="coerce")
df_full = df_full[df_full["nota_num"].isin([1, 2, 3, 4, 5])].copy()
df_full["label"] = df_full["nota_num"].apply(mapear_label)

print(f"Dataset completo (após filtro): {len(df_full):,} registros")
print(df_full["label"].value_counts().rename(NOME_LABEL).to_string())

# undersampling balanceado
grupos = []
for label, grupo in df_full.groupby("label"):
    n = min(N_POR_CLASSE, len(grupo))  # respeita limite da classe minoritária
    grupos.append(grupo.sample(n=n, random_state=SEED))

df = pd.concat(grupos).sample(frac=1, random_state=SEED).reset_index(drop=True)

print(f"\nAmostra balanceada: {len(df):,} registros ({N_POR_CLASSE} por classe)")
print(df["label"].value_counts().rename(NOME_LABEL).to_string())

# ========================================
# ETAPA 2 – Pré-processamento básico
# ========================================
print("\n" + "=" * 55)
print("ETAPA 2 - Pré-processamento")
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
print(df["label"].value_counts().rename(NOME_LABEL).to_string())

# ========================================
# ETAPA 3 – Embeddings (sentence-transformers)
# ========================================
print("\n" + "=" * 55)
print("ETAPA 3 - Embeddings")
print("=" * 55)

if os.path.exists(EMB_SAVE_PATH):
    print(f"Carregando embeddings salvos de '{EMB_SAVE_PATH}'...")
    embeddings = np.load(EMB_SAVE_PATH)
else:
    print(f"Modelo: {MODEL_NAME}")
    print("Vetorizando...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        df["texto"].tolist(),
        batch_size=64,
        show_progress_bar=True,
    )
    np.save(EMB_SAVE_PATH, embeddings)
    print(f"Embeddings salvos em '{EMB_SAVE_PATH}'")

print(f"Shape dos embeddings: {embeddings.shape}")

# ========================================
# ETAPA 4 – TF-IDF + Regressão Logística
# ========================================
print("\n" + "=" * 60)
print("ETAPA 4 – TF-IDF + Regressão Logística")
print("=" * 60)
 
stop_pt = stopwords.words("portuguese")
 
# Vetorizador TF-IDF (unigramas + bigramas, sem stopwords em português)
tfidf = TfidfVectorizer(
    max_features15_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    stop_words=stop_pt,
    min_df=2,
)
 
X_tfidf = tfidf.fit_transform(df["texto"])
y = df["label"].values
 
X_tr_tfidf, X_te_tfidf, y_tr, y_te = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=SEED, stratify=y
)
 
print(f"Vocabulário TF-IDF  : {X_tfidf.shape[1]:,} termos")
print(f"Treino: {X_tr_tfidf.shape[0]:,} | Teste: {X_te_tfidf.shape[0]:,}")
 
lr_tfidf = LogisticRegression(max_iter=1_000, random_state=SEED, C=1.0)
lr_tfidf.fit(X_tr_tfidf, y_tr)
y_pred_tfidf = lr_tfidf.predict(X_te_tfidf)
 
f1_tfidf = f1_score(y_te, y_pred_tfidf, average="weighted")
print(f"\nF1 weighted (TF-IDF + LR): {f1_tfidf:.4f}")
print(classification_report(
    y_te, y_pred_tfidf,
    target_names=["Negativo", "Neutro", "Positivo"],
    zero_division=0,
))

# ========================================
# ETAPA 5 – Baseline ML com Embeddings
#            (Logistic Regression + LinearSVC)
# ========================================
print("\n" + "=" * 55)
print("ETAPA 4 - Baseline ML")
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
        target_names=["Negativo", "Neutro", "Positivo"],
        zero_division=0
    ))

# ========================================
# RESUMO COMPARATIVO DE F1
# ========================================
print("\n" + "=" * 60)
print("RESUMO COMPARATIVO – F1 Weighted")
print("=" * 60)
for nome, f1 in resultados_f1.items():
    print(f"  {nome:<45}: {f1:.4f}")


# ========================================
# ETAPA 6 – ANÁLISE DE FREQUÊNCIA DE PALAVRAS
#            NLTK vs spaCy
# ========================================
print("\n" + "=" * 60)
print("ETAPA 6 – Palavras mais frequentes: NLTK vs spaCy")
print("=" * 60)
 
# --- 6a. Preparação: amostra para a análise lexical ---
# Usa os mesmos textos do df balanceado
corpus = df["texto"].tolist()
 
# ----- NLTK -----
print(f"\n{'─' * 40}")
print("NLTK – tokenização + stopwords")
print(f"{'─' * 40}")
 
stop_nltk = set(stopwords.words("portuguese"))
 
def tokenizar_nltk(texto):
    tokens = word_tokenize(texto.lower(), language="portuguese")
    return [t for t in tokens if t.isalpha() and t not in stop_nltk]
 
print("Tokenizando com NLTK (pode levar alguns instantes)...")
tokens_nltk_todos = []
for texto in corpus:
    tokens_nltk_todos.extend(tokenizar_nltk(texto))
 
freq_nltk = Counter(tokens_nltk_todos)
top_nltk = freq_nltk.most_common(TOP_N_PALAVRAS)
 
print(f"\nTop {TOP_N_PALAVRAS} palavras mais frequentes (NLTK):")
print(f"{'Rank':<6} {'Palavra':<25} {'Frequência':>10}")
print("-" * 45)
for rank, (palavra, freq) in enumerate(top_nltk, 1):
    print(f"{rank:<6} {palavra:<25} {freq:>10,}")
 
# ----- spaCy -----
print(f"\n{'─' * 40}")
print("spaCy – lematização + stopwords")
print(f"{'─' * 40}")
 
try:
    nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
    print("Modelo spaCy 'pt_core_news_sm' carregado.")
except OSError:
    print("Modelo 'pt_core_news_sm' não encontrado.")
    print("Execute: python -m spacy download pt_core_news_sm")
    nlp = None
 
if nlp is not None:
    print("Lematizando com spaCy (pode levar alguns instantes)...")
    tokens_spacy_todos = []
    # processa em lotes para eficiência
    for doc in nlp.pipe(corpus, batch_size=256):
        for token in doc:
            if (
                token.is_alpha
                and not token.is_stop
                and len(token.lemma_) > 1
            ):
                tokens_spacy_todos.append(token.lemma_.lower())
 
    freq_spacy = Counter(tokens_spacy_todos)
    top_spacy = freq_spacy.most_common(TOP_N_PALAVRAS)
 
    print(f"\nTop {TOP_N_PALAVRAS} palavras mais frequentes (spaCy – lemas):")
    print(f"{'Rank':<6} {'Lema':<25} {'Frequência':>10}")
    print("-" * 45)
    for rank, (lema, freq) in enumerate(top_spacy, 1):
        print(f"{rank:<6} {lema:<25} {freq:>10,}")
 
    # ----- Comparação lado a lado -----
    print(f"\n{'=' * 60}")
    print(f"COMPARAÇÃO LADO A LADO – Top {TOP_N_PALAVRAS}")
    print(f"{'=' * 60}")
    print(f"{'Rank':<6} {'NLTK (token)':<28} {'spaCy (lema)':<28}")
    print("-" * 64)
    for i in range(TOP_N_PALAVRAS):
        nltk_item  = f"{top_nltk[i][0]} ({top_nltk[i][1]:,})"   if i < len(top_nltk)  else "–"
        spacy_item = f"{top_spacy[i][0]} ({top_spacy[i][1]:,})" if i < len(top_spacy) else "–"
        print(f"{i+1:<6} {nltk_item:<28} {spacy_item:<28}")
 
    # ----- Palavras exclusivas de cada método -----
    palavras_nltk  = {p for p, _ in top_nltk}
    palavras_spacy = {p for p, _ in top_spacy}
    so_nltk  = palavras_nltk  - palavras_spacy
    so_spacy = palavras_spacy - palavras_nltk
 
    print(f"\nPalavras no Top {TOP_N_PALAVRAS} APENAS no NLTK  : {sorted(so_nltk)}")
    print(f"Palavras no Top {TOP_N_PALAVRAS} APENAS no spaCy : {sorted(so_spacy)}")
 
# ========================================
# ETAPA 7 – TF-IDF: termos mais relevantes
#            por classe
# ========================================
print("\n" + "=" * 60)
print("ETAPA 7 – TF-IDF: termos mais discriminativos por classe")
print("=" * 60)
 
TOP_TERMOS = 15
classes_nomes = ["Negativo", "Neutro", "Positivo"]
feature_names = np.array(tfidf.get_feature_names_out())
 
# Coeficientes da regressão logística (um vetor por classe)
coef = lr_tfidf.coef_   # shape: (n_classes, n_features)
 
for i, nome_classe in enumerate(classes_nomes):
    top_idx = np.argsort(coef[i])[-TOP_TERMOS:][::-1]
    termos_top = feature_names[top_idx]
    pesos_top  = coef[i][top_idx]
    print(f"\nClasse: {nome_classe} – termos com maior coeficiente positivo")
    print(f"{'Rank':<6} {'Termo':<35} {'Coef':>8}")
    print("-" * 51)
    for rank, (termo, peso) in enumerate(zip(termos_top, pesos_top), 1):
        print(f"{rank:<6} {termo:<35} {peso:>8.4f}")