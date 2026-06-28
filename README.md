# Classificação de Relatos de Consumidores com TF-IDF e LLM

Projeto desenvolvido para a disciplina de **Mineração de Textos**, com o objetivo de realizar a classificação automática de relatos de consumidores utilizando técnicas de Processamento de Linguagem Natural (PLN), aprendizado de máquina, modelos de linguagem (LLMs) e recuperação semântica (RAG).

---

## Objetivo

Construir um pipeline de classificação supervisionada capaz de identificar o sentimento de relatos de consumidores utilizando:

* Vetorização TF-IDF
* Modelos clássicos de Machine Learning
* Extração de informações estruturadas via LLM com validação Pydantic
* Classificação zero-shot via LLM
* Recuperação semântica com RAG (LlamaIndex)

---

## Dataset

Dataset público de reclamações do **consumidor.gov.br**, disponível no Kaggle.
Link: https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br

Características:
* Aproximadamente **204 mil relatos**
* Textos em português
* Avaliação do consumidor (nota de 1 a 5)

Classificação em três classes:

| Nota | Classe   |
|------|----------|
| 1–2  | Negativo |
| 3    | Neutro   |
| 4–5  | Positivo |

**Undersampling balanceado:** 5.000 registros por classe, totalizando **15.000 documentos**.

---

## Tecnologias Utilizadas

* Python
* Pandas / NumPy
* Scikit-Learn
* LlamaIndex
* Groq API (Llama 3.1 8B)
* Pydantic
* python-dotenv

---

## Estrutura do Projeto

```
.
├── pipeline-3class-llm_TFIDF.py           # Pipeline principal: TF-IDF + LLM features + zero-shot
├── pipeline-3class-NLTK_spaCy-TFIDF.py    # Pipeline com TF-IDF + NLTK e spaCy (análise exploratória)
├── rag.py                                 # RAG com LlamaIndex — comparação de chunking
├── custo.py                               # Estimativa de custo com Gemini API
├── inspectDF.py                           # Script para entendimento do dataset
├── prompts/
│   └── extract_feats.txt                  # Prompt para extração de features
├── DECISIONS.md                           # Decisões técnicas do projeto
├── ERRORS_N_INSIGHTS.md                   # Análise de erros e insights acionáveis
├── LOGS.md                                # Resultados completos dos experimentos
├── PYDANTIC.md                            # Esquema das features extraídas pelo LLM
└── README.md
```

---

## Pipeline Desenvolvido

1. Carregamento do dataset e undersampling balanceado (5k/classe)
2. Pré-processamento (remoção de URLs, normalização de espaços)
3. Vetorização com TF-IDF (10k features, bigrams, sublinear_tf)
4. Treinamento dos classificadores baseline (Logistic Regression e LinearSVC)
5. Extração combinada de features + classificação zero-shot via Groq API (Llama 3.1 8B)
6. Validação das saídas do LLM com schema Pydantic
7. Concatenação das features ao TF-IDF e novo treinamento
8. Comparação de resultados: baseline vs. features LLM vs. zero-shot
9. RAG com LlamaIndex: indexação de relatos e comparação de estratégias de chunking

---

## Features Extraídas pelo LLM

Seis atributos estruturados em uma única chamada à API:

* Classificação zero-shot (Negativo / Neutro / Positivo)
* Categoria do problema (7 opções)
* Tom emocional (4 opções)
* Menção a valor financeiro (bool)
* Menção a prazo (bool)
* Complexidade da reclamação (3 níveis)

Validação automática com schema Pydantic.

---

## Resultados

### Classificação

| Configuração | F1 Weighted | Relatos |
|---|---|---|
| Baseline (TF-IDF + LR) | **0.4818** | 15k |
| TF-IDF + LLM features | 0.4253 | 501 |
| Zero-shot LLM | 0.1709 | 501 |

### RAG com LlamaIndex (500 relatos)

| Estratégia | Nodes | keyword_recall | avg_score |
|---|---|---|---|
| hierárquico | 1446 | **0.200** | **0.564** |
| overlap | 734 | 0.000 | 0.538 |
| fixo | 715 | 0.000 | 0.533 |

---

## Principais Decisões

* **TF-IDF > embeddings:** F1 0.4818 vs 0.4619
* **Undersampling balanceado:** 5k/classe para tratar desbalanceamento do Neutro
* **LLM via Groq API:** portabilidade, sem dependência de hardware local
* **Extraçãocombinada:** features + zero-shot em uma chamada para economizar tokens
* **Chunking hierárquico:** melhor recall no RAG com `chunk_sizes=[512, 256, 128]`

Justificativas completas em **DECISIONS.md**.

---

## Principais Desafios

* **Zero-shot com viés forte para Negativo** — F1 0.1709, recall 1.00 para Negativo, 0.00 para Neutro
* **Classe Neutro semanticamente difusa** — F1 consistentemente mais baixo (~0.39–0.42)
* **Colapso no campo `tom`** — 91% dos relatos classificados como "furioso"
* **Corpus semanticamente homogêneo** — dificulta diferenciação no RAG

Análises detalhadas em **ERRORS_N_INSIGHTS.md**.

---

## Como Executar

### Criar ambiente virtual

```bash
python -m venv .venv
```

### Ativar ambiente

Linux / Mac:
```bash
source .venv/bin/activate
```

Windows:
```bash
.venv\Scripts\activate
```

### Instalar dependências

```bash
pip install groq pydantic scikit-learn pandas numpy python-dotenv llama-index llama-index-embeddings-huggingface
```

### Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:
```
GROQ_API_KEY=sua_chave_aqui
```

### Executar

Pipeline principal (TF-IDF + LLM):
```bash
python pipeline-3class-llm_TFIDF.py
```

RAG com LlamaIndex:
```bash
python rag.py
```

Pipeline com NLTK e spaCy (análise exploratória):
```bash
python pipeline-3class-NLTK_spaCy-TFIDF.py
```

---

## Arquivos de Apoio

| Arquivo | Descrição |
|---|---|
| DECISIONS.md | Justificativas técnicas das decisões adotadas |
| ERRORS_N_INSIGHTS.md | Análise de erros e insights acionáveis |
| LOGS.md | Resultados completos dos experimentos |
| PYDANTIC.md | Estrutura das features extraídas pelo LLM |

---

## Conclusão

O baseline TF-IDF + Logistic Regression (F1 0.4818) superou as abordagens com LLM em dados rotulados suficientes. O LLM agregou valor como extrator de features estruturadas (F1 0.4253 com 501 amostras), mas o zero-shot sem calibração mostrou-se inviável (F1 0.1709). No RAG, o chunking hierárquico foi o único com recall > 0, confirmando a importância de preservar contexto semântico na recuperação.

---

## Autores

* Letícia Ferreira Silva
* João Pedro Guervich Varrichio
* Eduardo Yuji Yamagata

Projeto desenvolvido para a disciplina de **Mineração de Textos**.