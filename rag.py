"""
RAG com LlamaIndex — Reclamações do Consumidor
Trabalho Final — Mineração de Textos

Compara 3 estratégias de chunking sobre os relatos do consumidores.gov.br:
  1. Chunking fixo (SentenceSplitter, sem overlap)
  2. Chunking fixo com overlap
  3. Chunking hierárquico

Instalação:
  pip install llama-index llama-index-embeddings-huggingface sentence-transformers pandas
"""

import json
import re
import time
import pandas as pd
from llama_index.core import Document, Settings, VectorStoreIndex, StorageContext
from llama_index.core.node_parser import (
    SentenceSplitter,
    HierarchicalNodeParser,
    get_leaf_nodes,
)
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# ============================================================
# CONFIGURAÇÃO
# ============================================================
CAMINHO_JSON     = "dados2025.json"
N_DOCS           = 500       # relatos indexados (aumentar no PC)
SEED             = 42
TOP_K            = 5
EMBEDDING_MODEL  = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ============================================================
# CARREGAMENTO E PRÉ-PROCESSAMENTO
# ============================================================
print("=" * 55)
print("Carregando dataset...")
print("=" * 55)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

def limpar_texto(texto):
    if not isinstance(texto, str): return ""
    return re.sub(r"\s+", " ", re.sub(r"http\S+", "", texto)).strip()

# Amostra balanceada por empresa (diversidade no corpus)
df = pd.DataFrame(data).dropna(subset=["relato", "empresa"])
df["texto"] = df["relato"].apply(limpar_texto)
df = df[df["texto"].str.len() > 50]  # remove relatos muito curtos
df_sample = df.sample(n=min(N_DOCS, len(df)), random_state=SEED).reset_index(drop=True)

print(f"Relatos indexados: {len(df_sample):,}")
print(f"Empresas únicas: {df_sample['empresa'].nunique()}")

# Converte para Documents do LlamaIndex
documents = [
    Document(
        text=row["texto"],
        metadata={
            "empresa":  row.get("empresa", ""),
            "status":   row.get("status", ""),
            "nota":     str(row.get("nota", "")),
            "doc_id":   str(row.name),
        }
    )
    for _, row in df_sample.iterrows()
]

# ============================================================
# CONFIGURAÇÃO DO EMBEDDING
# ============================================================
print("\nCarregando modelo de embedding...")
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL)
Settings.llm = None  # foco em recuperação, sem LLM

# ============================================================
# ESTRATÉGIAS DE CHUNKING
# ============================================================
SPLITTERS = {
    "fixo":      SentenceSplitter(chunk_size=256, chunk_overlap=0),
    "overlap":   SentenceSplitter(chunk_size=256, chunk_overlap=50),
    "hierarquico": HierarchicalNodeParser.from_defaults(chunk_sizes=[512, 256, 128]),
}

def build_simple_retriever(name, splitter, docs, top_k):
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes)
    return {
        "name": name,
        "retriever": index.as_retriever(similarity_top_k=top_k),
        "n_nodes": len(nodes),
    }

def build_hierarchical_retriever(splitter, docs, top_k):
    all_nodes  = splitter.get_nodes_from_documents(docs)
    leaf_nodes = get_leaf_nodes(all_nodes)
    docstore   = SimpleDocumentStore()
    docstore.add_documents(all_nodes)
    storage    = StorageContext.from_defaults(docstore=docstore)
    index      = VectorStoreIndex(leaf_nodes, storage_context=storage)
    base_ret   = index.as_retriever(similarity_top_k=top_k)
    retriever  = AutoMergingRetriever(base_ret, storage, verbose=False)
    return {
        "name": "hierarquico",
        "retriever": retriever,
        "n_nodes": len(leaf_nodes),
    }

print("\n" + "=" * 55)
print("Montando índices...")
print("=" * 55)

rag_systems = {}
summary_rows = []

for name, splitter in SPLITTERS.items():
    print(f"  → {name}")
    t0 = time.time()
    if name == "hierarquico":
        sys_info = build_hierarchical_retriever(splitter, documents, TOP_K)
    else:
        sys_info = build_simple_retriever(name, splitter, documents, TOP_K)
    sys_info["tempo"] = round(time.time() - t0, 1)
    rag_systems[name] = sys_info
    summary_rows.append({
        "estratégia":    name,
        "nodes_indexados": sys_info["n_nodes"],
        "tempo_s":       sys_info["tempo"],
    })

print(pd.DataFrame(summary_rows).to_string(index=False))

# ============================================================
# PERGUNTAS DE AVALIAÇÃO
# ============================================================
# Perguntas sobre o domínio do consumidor.gov.br
GOLDEN_QUESTIONS = [
    {
        "qid": "q01",
        # "question": "Qual empresa tem mais defeitos registrados?",
        "question": "Qual empresa tem mais reclamaçoes registradas?",
        "keywords": ["defeito", "troca", "devolução", "produto", "garantia"],
    },
    # {
    #     "qid": "q02",
    #     "question": "Fui cobrado indevidamente no cartão de crédito. Como solicitar estorno?",
    #     "keywords": ["cobrança indevida", "cartão", "estorno", "reembolso", "cancelamento"],
    # },
    # {
    #     "qid": "q03",
    #     "question": "Meu pedido atrasou e a empresa não dá informações sobre a entrega.",
    #     "keywords": ["entrega", "atraso", "prazo", "rastreio", "logística"],
    # },
    # {
    #     "qid": "q04",
    #     "question": "Quero cancelar meu contrato mas a empresa não permite.",
    #     "keywords": ["cancelamento", "contrato", "rescisão", "multa", "fidelidade"],
    # },
    # {
    #     "qid": "q05",
    #     "question": "O atendimento da empresa foi péssimo e não resolvi meu problema.",
    #     "keywords": ["atendimento", "SAC", "ouvidoria", "reclamação", "resolução"],
    # },
]

# ============================================================
# AVALIAÇÃO
# ============================================================
def retrieve_rows(question, strategy, top_k=TOP_K):
    results = rag_systems[strategy]["retriever"].retrieve(question)
    rows = []
    for rank, r in enumerate(results[:top_k], 1):
        text = getattr(r.node, "text", "") or r.node.get_content()
        rows.append({
            "rank":    rank,
            "score":   float(r.score or 0),
            "empresa": r.node.metadata.get("empresa", ""),
            "status":  r.node.metadata.get("status", ""),
            "nota":    r.node.metadata.get("nota", ""),
            "texto":   text,
        })
    return rows

def evaluate_one(question_item, strategy):
    results  = retrieve_rows(question_item["question"], strategy)
    keywords = question_item["keywords"]
    context  = " ".join(r["texto"] for r in results).lower()
    kw_hits  = sum(1 for kw in keywords if kw.lower() in context)
    return {
        "qid":               question_item["qid"],
        "estratégia":        strategy,
        "keyword_recall@k":  round(kw_hits / len(keywords), 3),
        "avg_score":         round(sum(r["score"] for r in results) / len(results), 3) if results else 0,
        "context_words":     len(context.split()),
    }

print("\n" + "=" * 55)
print("Avaliando estratégias...")
print("=" * 55)

eval_rows = []
for q in GOLDEN_QUESTIONS:
    for strategy in rag_systems:
        eval_rows.append(evaluate_one(q, strategy))

eval_df = pd.DataFrame(eval_rows)
print(eval_df.to_string(index=False))

# ============================================================
# RESUMO POR ESTRATÉGIA
# ============================================================
print("\n" + "=" * 55)
print("RESUMO — média por estratégia")
print("=" * 55)

summary = (
    eval_df
    .groupby("estratégia", as_index=False)
    .agg(
        keyword_recall_medio=("keyword_recall@k", "mean"),
        avg_score_medio=("avg_score", "mean"),
        context_words_medio=("context_words", "mean"),
    )
    .sort_values("keyword_recall_medio", ascending=False)
)
print(summary.to_string(index=False))

# ============================================================
# INSPECIONAR UMA PERGUNTA
# ============================================================
print("\n" + "=" * 55)
print("INSPEÇÃO — q01 em todas as estratégias")
print("=" * 55)

q01 = next(q for q in GOLDEN_QUESTIONS if q["qid"] == "q01")
print(f"Pergunta: {q01['question']}\n")

for strategy in rag_systems:
    print(f"\n--- {strategy} ---")
    results = retrieve_rows(q01["question"], strategy)
    for r in results:
        preview = r["texto"][:200].replace("\n", " ")
        print(f"  rank={r['rank']} score={r['score']:.3f} empresa={r['empresa']}")
        print(f"  {preview}...")