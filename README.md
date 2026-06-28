# Classificação de Relatos de Consumidores com Embeddings e LLM

Projeto desenvolvido para a disciplina de **Mineração de Textos**, com o objetivo de realizar a classificação automática de relatos de consumidores utilizando técnicas de Processamento de Linguagem Natural (PLN), aprendizado de máquina, modelos de linguagem (LLMs) e recuperação semântica (RAG).

---

## Objetivo

Construir um pipeline de classificação supervisionada capaz de identificar o sentimento de relatos de consumidores utilizando:

* Vetorização TF-IDF e embeddings semânticos
* Modelos clássicos de Machine Learning
* Extração de informações estruturadas via LLM com validação Pydantic
* Classificação zero-shot via LLM
* Recuperação semântica com RAG (LlamaIndex)

---

## Dataset

Foi utilizado o dataset público de reclamações do **consumidor.gov.br**, disponível no Kaggle.
Link: https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br

Características do conjunto de dados:

* Aproximadamente **204 mil relatos**
* Textos em português
* Avaliação do consumidor (nota de 1 a 5)

A classificação foi organizada em três classes:

| Nota | Classe   |
|------|----------|
| 1–2  | Negativo |
| 3    | Neutro   |
| 4–5  | Positivo |

Como a classe **Neutro** possui quantidade significativamente menor de exemplos, foi realizado **undersampling balanceado**, utilizando **5.000 registros por classe**, totalizando **15.000 documentos**.

---

## Tecnologias Utilizadas

* Python
* Pandas / NumPy
* Scikit-Learn
* Sentence Transformers
* LlamaIndex
* Groq API (Llama 3.1 8B)
* Pydantic
* python-dotenv

---

## Estrutura do Projeto

```
.
├── pipeline-3class.py           # Pipeline com TF-IDF + classificadores baseline
├── pipeline-3class-llm.py       # Pipeline unificado: TF-IDF + LLM features + zero-shot
├── rag_llamaindex.py            # RAG com LlamaIndex — comparação de estratégias de chunking
├── custo.py                     # Estimativa de custo com Gemini API
├── prompts/
│   └── extract_feats.txt        # Prompt utilizado na extração de features
├── DECISIONS.md                 # Decisões técnicas de projeto
├── ERRORS_N_INSIGHTS.md         # Análise de erros e insights acionáveis
├── LOGS.md                      # Resultados completos dos experimentos
├── PYDANTIC.md                  # Esquema das features extraídas pelo LLM
└── README.md
```

---

## Pipeline Desenvolvido

O fluxo do projeto é composto pelas seguintes etapas:

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

Cada relato foi convertido em seis atributos estruturados em uma única chamada à API:

* Classificação zero-shot (Negativo / Neutro / Positivo)
* Categoria do problema
* Tom emocional
* Menção a valor financeiro
* Menção a prazo
* Complexidade da reclamação

Essas informações foram validadas automaticamente utilizando um schema Pydantic.

---

## Resultados

### Pipeline de Classificação

| Configuração | F1 Weighted | Relatos |
|---|---|---|
| Baseline (TF-IDF + LR) | **0.4818** | 15k |
| TF-IDF + LLM features | 0.3684 | 414 |
| Zero-shot LLM | 0.2371 | 414 |

### RAG com LlamaIndex (500 relatos, 5 perguntas)

| Estratégia | Nodes indexados | keyword_recall médio | avg_score médio |
|---|---|---|---|
| hierárquico | 1446 | **0.200** | **0.564** |
| overlap | 734 | 0.000 | 0.538 |
| fixo | 715 | 0.000 | 0.533 |

---

## Principais Decisões

* TF-IDF superou embeddings densos no baseline (F1 0.4818 vs 0.4619)
* Undersampling balanceado (5k/classe) para tratar desbalanceamento do Neutro
* LLM via Groq API para portabilidade — sem dependência de hardware local
* Extração combinada (features + zero-shot) em uma única chamada para economizar tokens
* Chunking hierárquico com `chunk_sizes=[512, 256, 128]` obteve melhor recall no RAG

As justificativas completas encontram-se no arquivo **DECISIONS.md**.

---

## Principais Desafios

* Rate limit do Groq (6k tokens/min) causou 17.4% de erros em 500 chamadas consecutivas
* Zero-shot com viés forte para classe Negativo — modelo não calibrado para dataset balanceado
* Classe Neutro semanticamente difusa — F1 consistentemente mais baixo (~0.40–0.42)
* Corpus de relatos semanticamente homogêneo dificulta diferenciação entre estratégias de chunking no RAG

Essas análises encontram-se detalhadas em **ERRORS_N_INSIGHTS.md**.

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
pip install groq pydantic sentence-transformers scikit-learn pandas numpy tiktoken python-dotenv llama-index llama-index-embeddings-huggingface
```

### Configurar variáveis de ambiente

Crie um arquivo `.env` na raiz do projeto:
```
GROQ_API_KEY=sua_chave_aqui
```

### Executar

Pipeline baseline (TF-IDF) + LLM:
```bash
python pipeline-3class-llm_TFIDF
```

RAG com LlamaIndex:
```bash
python rag.py
```

TF-IDF com NLTK e spaCy para testes:
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

Os experimentos demonstraram que o baseline TF-IDF + Logistic Regression supera as abordagens LLM com dados rotulados suficientes (15k relatos, F1 0.4818). O LLM agrega valor como ferramenta de extração de features estruturadas, mas requer volume maior de amostras para superar o baseline. O RAG com chunking hierárquico obteve o melhor desempenho de recuperação entre as estratégias testadas.

---

## Autores

* Letícia Ferreira Silva
* João Pedro Guervich Varrichio
* Eduardo Yuji Yamagata

Projeto desenvolvido para a disciplina de **Mineração de Textos**.