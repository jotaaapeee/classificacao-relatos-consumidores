import json
import re
import numpy as np
import pandas as pd
import faiss
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

CAMINHO_JSON = "dados2025.json"
EMB_PATH = "embeddings_3class_balanced.npy"
MODEL_NAME = "paraphrase-multilingual-mpnet-base-v2"
N_POR_CLASSE = 5_000
SEED = 42
K_SIMILARES = 5

LABEL_NAMES = {0: "😡 Negativo", 1: "😐 Neutro", 2: "😊 Positivo"}
LABEL_COLORS = {0: "#ff4b4b", 1: "#ffa500", 2: "#21c354"}

# CARREGAMENTO (cacheado)
@st.cache_resource(show_spinner="Carregando modelo de embedding...")
def carregar_modelo():
    return SentenceTransformer(MODEL_NAME)

@st.cache_resource(show_spinner="Carregando dataset e embeddings...")
def carregar_dados():
    with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    df_full = pd.DataFrame(data)
    df_full["nota_num"] = pd.to_numeric(df_full["nota"], errors="coerce")
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
        n = min(N_POR_CLASSE, len(grupo))
        grupos.append(grupo.sample(n=n, random_state=SEED))

    df = pd.concat(grupos).sample(frac=1, random_state=SEED).reset_index(drop=True)

    def limpar_texto(texto):
        if not isinstance(texto, str):
            return ""
        texto = re.sub(r"http\S+", "", texto)
        return re.sub(r"\s+", " ", texto).strip()

    df["texto"] = df["relato"].apply(limpar_texto)
    embeddings = np.load(EMB_PATH).astype("float32")

    return df, embeddings

@st.cache_resource(show_spinner="Treinando classificador...")
def treinar_classificador(_embeddings, labels):
    clf = LinearSVC(max_iter=2000, random_state=SEED)
    clf.fit(_embeddings, labels)
    return clf

@st.cache_resource(show_spinner="Construindo índice FAISS...")
def construir_faiss(_embeddings):
    dim = _embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(_embeddings)
    return index

# INTERFACE
st.set_page_config(
    page_title="Analisador de Reclamações",
    layout="wide"
)

st.title("Analisador de Reclamações - consumidores.gov.br")
st.caption("Trabalho Final - Mineração de Textos - Dataset balanceado: 15k relatos (5k/classe)")

# carrega tudo
modelo = carregar_modelo()
df, embeddings = carregar_dados()
clf = treinar_classificador(embeddings, df["label"].values)
index = construir_faiss(embeddings)

# sidebar com stats
with st.sidebar:
    st.header("Dataset")
    
    # relatos
    col_label, col_val = st.columns([3, 3])
    col_label.markdown("**Total de relatos:**")
    col_val.write(f"{len(df):,}")
    
    # classes
    col_label, col_val = st.columns([1, 3])
    col_label.markdown("**Classes:**")
    col_val.write("Negativo - Neutro - Positivo")

    # embeddings
    col_label, col_val = st.columns([3, 3])
    col_label.markdown("**Embeddings:**")
    col_val.write(f"{embeddings.shape[1]} dims")

    st.divider()
    
    st.header("Configurações")
    k = st.slider("Relatos similares (K)", min_value=1, max_value=10, value=K_SIMILARES)

# input
st.subheader("Digite uma reclamação")
relato_input = st.text_area(
    label="Relato",
    placeholder="Ex: Comprei um produto e não recebi, já se passaram 30 dias e a empresa não responde...",
    height=150,
    label_visibility="collapsed"
)

if st.button("Analisar", type="primary", disabled=not relato_input.strip()):

    with st.spinner("Analisando..."):
        # vetoriza a query
        def limpar(t):
            return re.sub(r"\s+", " ", re.sub(r"http\S+", "", t)).strip()

        query_limpa = limpar(relato_input)
        query_emb = modelo.encode([query_limpa]).astype("float32")

        # classifica
        pred_label = clf.predict(query_emb)[0]

        # busca similares
        distancias, indices = index.search(query_emb, k)

    # resultado da classificação
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Classificação")
        cor = LABEL_COLORS[pred_label]
        st.markdown(
            f"<div style='background-color:{cor}22; border-left:4px solid {cor};"
            f"padding:16px;border-radius:8px;font-size:1.4em;font-weight:bold;color:{cor}'>"
            f"{LABEL_NAMES[pred_label]}</div>",
            unsafe_allow_html=True
        )
        st.caption("Predição via LinearSVC + embeddings semânticos")
 
    with col2:
        st.subheader(f"{k} Relatos mais similares")
        for rank, (idx, dist) in enumerate(zip(indices[0], distancias[0]), 1):
            row = df.iloc[idx]
            similaridade = 1 / (1 + dist)
            cor_s = LABEL_COLORS[row["label"]]
            with st.expander(f"#{rank} - {LABEL_NAMES[row['label']]} - {similaridade:.2%} - {row.get('empresa','N/A')}"):
                st.markdown(
                    f"<div style='border-left:3px solid {cor_s};padding-left:12px'>{row['texto']}</div>",
                    unsafe_allow_html=True
                )
                ca, cb = st.columns(2)
                ca.metric("Nota", row.get("nota", "N/A"))
                cb.metric("Status", row.get("status", "N/A"))

else:
    st.info("Digite uma reclamação acima e clique em **Analisar** para ver a classificação e relatos similares.")
    
    st.divider()
    st.subheader("Top 10 empresas mais reclamadas no corpus")
    top_empresas = df["empresa"].value_counts().head(10).sort_values(ascending=True)
    st.bar_chart(top_empresas, horizontal=True)