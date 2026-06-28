# Classificação de Relatos de Consumidores com Embeddings e LLM

Projeto desenvolvido para a disciplina de **Mineração de Textos**, com o objetivo de realizar a classificação automática de relatos de consumidores utilizando técnicas de Processamento de Linguagem Natural (PLN), aprendizado de máquina e modelos de linguagem (LLMs).

---

## Objetivo

Construir um pipeline de classificação supervisionada capaz de identificar o sentimento de relatos de consumidores utilizando:

* Embeddings semânticos;
* Modelos clássicos de Machine Learning;
* Extração de informações estruturadas por meio de um LLM executado localmente.

Além da classificação tradicional, o projeto avalia se informações extraídas por um modelo de linguagem podem melhorar o desempenho do classificador.

---

## Dataset

Foi utilizado o dataset público de reclamações do **Consumidor.gov.br**, disponível no Kaggle.

Características do conjunto de dados:

* Aproximadamente **204 mil relatos**
* Textos em português
* Avaliação do consumidor (nota de 1 a 5)

A classificação foi organizada em três classes:

| Nota | Classe   |
| ---- | -------- |
| 1–2  | Negativo |
| 3    | Neutro   |
| 4–5  | Positivo |

Como a classe **Neutro** possui quantidade significativamente menor de exemplos, foi realizado **undersampling balanceado**, utilizando **5.000 registros por classe**, totalizando **15.000 documentos**.

---

## Tecnologias Utilizadas

* Python
* Pandas
* NumPy
* Scikit-Learn
* Sentence Transformers
* GROQ
* .env
* Pydantic

---

## Estrutura do Projeto

```
.
├── pipeline-3class.py           # Pipeline utilizando apenas embeddings
├── pipeline-3class-llm.py       # Pipeline com embeddings + features do LLM
├── custo.py                     # Utilitário de cálculo
├── extract_feats.txt            # Prompt utilizado na extração
├── DECISIONS.md                 # Decisões de projeto
├── ERRORS_N_INSIGHTS.md         # Análise de erros e insights
├── LOGS.md                      # Resultados experimentais
├── PYDANTIC.md                  # Esquema das features extraídas
└── README.md
```

---

## Pipeline Desenvolvido

O fluxo do projeto é composto pelas seguintes etapas:

1. Carregamento do dataset
2. Balanceamento das classes
3. Geração dos embeddings utilizando o modelo:

```
paraphrase-multilingual-mpnet-base-v2
```

4. Treinamento dos classificadores baseline:

   * Logistic Regression
   * LinearSVC

5. Extração de features semânticas utilizando o Llama 3.2 via Ollama.

6. Conversão das informações extraídas para variáveis estruturadas utilizando Pydantic.

7. Concatenação das novas features aos embeddings.

8. Novo treinamento dos classificadores.

9. Comparação entre os resultados obtidos.

---

## Features Extraídas pelo LLM

Cada relato foi convertido em cinco atributos estruturados:

* Categoria do problema
* Tom emocional
* Menção a valor financeiro
* Menção a prazo
* Complexidade da reclamação

Essas informações foram validadas automaticamente utilizando um schema Pydantic.

---

## Resultados

### Baseline

| Modelo              | F1 Weighted |
| ------------------- | ----------: |
| Logistic Regression |      0.4619 |
| LinearSVC           |      0.4637 |

### Embeddings + LLM

| Configuração                | F1 Weighted |
| --------------------------- | ----------: |
| Embedding puro              |      0.3189 |
| Embedding + Features do LLM |  **0.4058** |

A utilização das informações estruturadas extraídas pelo modelo de linguagem proporcionou um ganho aproximado de **0,087 pontos de F1 Weighted**.

---

## Principais Decisões

Durante o desenvolvimento foram tomadas algumas decisões importantes:

* utilização de três classes (Negativo, Neutro e Positivo);
* balanceamento por undersampling;
* uso de embeddings multilíngues;
* utilização do Llama 3.2 executado localmente;
* validação das respostas do LLM utilizando Pydantic.

As justificativas completas encontram-se no arquivo **DECISIONS.md**.

---

## Principais Desafios

Durante os experimentos foram observados alguns problemas relevantes:

* respostas do LLM ocasionalmente retornavam JSON inválido;
* dificuldade do modelo em classificar corretamente relatos da classe Neutro;
* divergência entre o sentimento expresso no texto e a nota atribuída pelo consumidor.

Essas análises encontram-se detalhadas em **ERRORS_N_INSIGHTS.md**.

---

## Como Executar

### Criar ambiente virtual

```bash
python -m venv .venv
```

### Ativar ambiente

Windows

```bash
.venv\Scripts\activate
```

Linux / Mac

```bash
source .venv/bin/activate
```

### Instalar dependências

```bash
pip install ollama pydantic sentence-transformers scikit-learn pandas numpy tiktoken
```

### Executar o pipeline

Sem LLM

```bash
python pipeline-3class.py
```

Com LLM

```bash
python pipeline-3class-llm.py
```

---

## Arquivos de Apoio

| Arquivo              | Descrição                                     |
| -------------------- | --------------------------------------------- |
| DECISIONS.md         | Justificativas técnicas das decisões adotadas |
| ERRORS_N_INSIGHTS.md | Análise de erros e oportunidades de melhoria  |
| LOGS.md              | Resultados completos dos experimentos         |
| PYDANTIC.md          | Estrutura das features extraídas pelo LLM     |

---

## Conclusão

Os experimentos demonstraram que a combinação entre embeddings semânticos e informações estruturadas extraídas por um modelo de linguagem pode melhorar o desempenho de classificadores tradicionais em tarefas de mineração de textos.

Apesar das limitações observadas — especialmente na classificação da classe Neutro e na geração ocasional de respostas inválidas pelo LLM — os resultados indicam que a utilização de modelos de linguagem como etapa complementar de engenharia de atributos representa uma abordagem promissora para aplicações de Processamento de Linguagem Natural em português.

---

## Autores
* Letícia Ferreira Silva
* João Pedro Guervich Varrichio
* Eduardo

Projeto desenvolvido para a disciplina de **Mineração de Textos**.
