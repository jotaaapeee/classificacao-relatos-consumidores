# DECISOES.md — Classificação de Relatos do Consumidor

## Visão Geral

Pipeline de classificação supervisionada (Caminho B) aplicado ao dataset público do consumidores.gov.br, com embeddings semânticos, baseline ML clássico e extração de features via LLM local.

---

## 1. Escolha do Dataset

**Decisão:** consumidores.gov.br (204k reclamações em PT-BR)

**Motivo:**
- Corpus público, em português, com rótulos naturais (coluna `nota` e `status`)
- Volume muito acima do mínimo exigido (500 documentos)
- Domínio rico para extração de features semânticas via LLM

---

## 2. Definição do Rótulo (3 classes)

**Decisão:** usar a coluna `nota` (1–5) agrupada em 3 classes:
- `0 = Negativo` (nota 1–2)
- `1 = Neutro` (nota 3)
- `2 = Positivo` (nota 4–5)

**Motivo:** o enunciado exige mínimo de 3 rótulos. A coluna `status` (Resolvido / Não Resolvido) foi usada para validação do pipeline binário auxiliar.

**Trade-off:** a classe Neutro é minoritária (~13k de 204k registros), o que gera F1 baixo nessa classe. Mitigado com `class_weight='balanced'`.

---

## 3. Modelo de Embedding

**Decisão:** `paraphrase-multilingual-mpnet-base-v2`

**Motivo:**
- Suporte nativo a PT-BR
- Roda em CPU (viável no notebook de desenvolvimento)
- Embeddings de 768 dimensões com boa capacidade semântica
- Gratuito e sem dependência de API externa

**Trade-off:** mais lento que modelos menores (ex: MiniLM), mas com representação semântica superior.

---

## 4. Estratégia de Amostragem

**Decisão:** amostragem estratificada por label

**Notebook (desenvolvimento):** 10–20k amostras
**PC (produção):** dataset completo ~200k amostras

**Motivo:** o notebook (Ryzen 7 5825u, 16GB RAM, sem GPU) não suporta vetorização de 200k registros em tempo razoável. Os embeddings são gerados no PC (RTX 3060 12GB) e salvos em `.npy` para uso no notebook.

**Implementação:** substituiu `groupby().apply()` por loop manual — incompatibilidade com pandas no Python 3.12/3.14.

---

## 5. Classificadores Baseline

**Decisão:** Logistic Regression e LinearSVC com `class_weight='balanced'`

**Motivo:**
- Modelos lineares funcionam bem sobre embeddings densos
- `class_weight='balanced'` compensa o desbalanceamento da classe Neutro
- Permitem comparação direta antes e depois da adição de features do LLM

**Resultados (200k, 3 classes):**
| Modelo | F1 weighted |
|---|---|
| Logistic Regression | ~0.69 |
| LinearSVC | ~0.69 |

**Resultados (binário — Resolvido / Não Resolvido, referência):**
| Modelo | 20k F1 | 200k F1 |
|---|---|---|
| Logistic Regression | 0.6812 | 0.6909 |
| LinearSVC | 0.6809 | 0.6935 |

---

## 6. LLM Local (Ollama / Llama 3.2)

**Decisão:** Llama 3.2 via Ollama, rodando localmente no PC com GPU

**Motivo:**
- Gratuito, sem limite de tokens
- RTX 3060 12GB suporta o modelo (3B parâmetros, ~2GB em VRAM)
- Privacidade dos dados (sem envio para APIs externas)

**Trade-off:** Ollama 0.24.0 não detecta GPU automaticamente — atualizado para 0.9.x para habilitar CUDA.

**Limitação:** extração aplicada em subamostra de 500 relatos por custo computacional. Taxa de erro de JSON malformado: 5.6% (28/500).

---

## 7. Schema Pydantic para Extração de Features

**Campos extraídos:**
- `categoria_problema`: 7 categorias (cobrança indevida, produto com defeito, atraso na entrega, atendimento ruim, cancelamento, fraude, outro)
- `tom`: neutro / frustrado / furioso / satisfeito
- `menciona_valor_financeiro`: bool
- `menciona_prazo`: bool
- `complexidade`: baixa / média / alta

**Motivo da escolha dos campos:** cobrem as dimensões semânticas mais relevantes para prever resolução — tipo do problema, urgência emocional e complexidade operacional.

**Resultado:** adição das features LLM melhorou F1 de 0.4874 → 0.5361 na subamostra de 3 classes (+0.049).

---

## 8. Salvamento de Artefatos

**Decisão:** embeddings e features LLM salvos em disco antes da apresentação

| Artefato | Arquivo |
|---|---|
| Embeddings 20k (binário) | `embeddings_20k.npy` |
| Embeddings 3 classes 20k | `embeddings_3class_20k.npy` |
| Embeddings 200k | `embeddings_200k.npy` |
| Features LLM | `features_llm.csv` |

**Motivo:** evita recomputação durante a demo ao vivo no notebook.

---

## 9. O que ficou de fora (e por quê)

| Item | Motivo |
|---|---|
| RAG (FAISS/ChromaDB) | Identificado como bônus, não implementado por escopo |
| Interface Streamlit/Gradio | Bônus, não implementado por escopo |
| DeepL / Google Translate | Descartado — dataset já está em PT-BR |
| Dataset Reddit (150k relatos em inglês) | Substituído pelo consumidores.gov.br para evitar tradução e usar rótulos naturais |