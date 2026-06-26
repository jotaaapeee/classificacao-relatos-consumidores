# DECISOES.md - Classificação de Relatos do Consumidor

## Visão Geral

Pipeline de classificação supervisionada (Caminho B) aplicado ao dataset público do consumidores.gov.br, com embeddings semânticos, baseline ML clássico e extração de features via LLM local.

---

## 1. Escolha do Dataset

**Decisão:** consumidores.gov.br (204k reclamações em PT-BR) https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br

**Motivo:**
- Corpus público, em português, com rótulos naturais (coluna `nota`)
- Volume muito acima do mínimo exigido (500 documentos), cada linha do dataset é um documento (um relato = um texto)
- Domínio rico para extração de features semânticas via LLM

---

## 2. Definição do Rótulo (3 classes)

**Decisão:** usar a coluna `nota` (1–5) agrupada em 3 classes:
- `0 = Negativo` (nota 1–2)
- `1 = Neutro` (nota 3)
- `2 = Positivo` (nota 4–5)

**Motivo:** o enunciado exige mínimo de 3 rótulos.

**Trade-off:** a classe Neutro é minoritária (~13k de 204k registros). Solução adotada: undersampling balanceado (ver seção 4).

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

## 4. Estratégia de Amostragem, Undersampling Balanceado

**Decisão:** undersampling balanceado com 5.000 registros por classe (15.000 total)

**Motivo:** o dataset original é desbalanceado (~105k Positivos, ~84k Negativos, ~13k Neutros). Com amostragem proporcional, o modelo ignorava completamente a classe Neutro (F1 = 0.00). O undersampling nivelar pelo Neutro resolve o desbalanceamento sem precisar de `class_weight='balanced'`, tornando as métricas mais confiáveis e comparáveis entre classes.

**Comparação:**
| Estratégia | F1 Neutro | F1 weighted |
|---|---|---|
| Proporcional (desbalanceado) | 0.00 | ~0.69 (enganoso) |
| Undersampling 5k/classe | ~0.40 | ~0.46 (honesto) |

O F1 geral cai, mas passa a refletir o desempenho real nas 3 classes.

**Implementação:** substituiu `groupby().apply()` por loop manual, incompatibilidade com pandas no Python 3.12/3.14.

---

## 5. Classificadores Baseline

**Decisão:** Logistic Regression e LinearSVC

**Motivo:** modelos lineares funcionam bem sobre embeddings densos e permitem comparação direta antes e depois da adição de features do LLM.

**Resultados, baseline 3 classes (balanceado, 15k):**
| Modelo | F1 weighted |
|---|---|
| Logistic Regression | 0.4619 |
| LinearSVC | 0.4637 |

---

## 6. LLM Local (Ollama / Llama 3.2)

**Decisão:** Llama 3.2 via Ollama, rodando localmente no PC com GPU

**Motivo:**
- Gratuito, sem limite de tokens
- Privacidade dos dados (sem envio para APIs externas)

**Trade-off:** No notebook (CPU), velocidade lenta inviabilizou 500 relatos, usado N=50 na demo ao vivo, com resultados do PC (N=500) carregados de arquivo.

**Taxa de erro (JSON malformado):**
- 500 relatos no PC (GPU): 36/500 = 7.2%
- 50 relatos no notebook (CPU): 2/50 = 4.0%

---

## 7. Schema Pydantic para Extração de Features

**Campos extraídos:**
- `categoria_problema`: 7 categorias (cobrança indevida, produto com defeito, atraso na entrega, atendimento ruim, cancelamento, fraude, outro)
- `tom`: neutro / frustrado / furioso / satisfeito
- `menciona_valor_financeiro`: bool
- `menciona_prazo`: bool
- `complexidade`: baixa / média / alta

**Motivo da escolha dos campos:** cobrem as dimensões semânticas mais relevantes para prever satisfação, tipo do problema, urgência emocional e complexidade operacional.

**Resultado (500 relatos, dataset balanceado 3 classes):**
| Configuração | F1 weighted |
|---|---|
| Embedding puro | 0.3189 |
| Embedding + LLM features | 0.4058 |

Ganho de +0.087 com a adição das features do LLM.

---

## 8. Tokenização e Representação Semântica

**Tokenização:** o modelo `paraphrase-multilingual-mpnet-base-v2` usa tokenização WordPiece (mesmo estilo do BERT), que quebra o texto em subpalavras. Exemplo: "cancelamento" → `["cancel", "##amento"]`. Isso permite que o modelo lide com palavras desconhecidas e variações morfológicas do português.

**Captura de contexto:** o modelo usa atenção bidirecional, o vetor de cada token é influenciado por todos os outros tokens da frase. O embedding final de 768 dimensões representa o significado do texto inteiro, não palavras isoladas. Isso é o diferencial em relação a abordagens como TF-IDF, onde cada termo é tratado de forma independente.

**Exemplo prático:** "produto não chegou" e "entrega atrasada" geram embeddings próximos no espaço vetorial mesmo sem compartilhar palavras, porque o modelo captura a equivalência semântica entre as duas frases.

---

## 9. Limpeza de Texto

**Decisão:** limpeza mínima, remoção de URLs e normalização de espaços

**Motivo:** o dataset do consumidores.gov.br já vem bem estruturado e limpo. Os relatos são textos escritos por consumidores em um formulário web, o que naturalmente reduz ruído (sem gírias extremas, sem emojis em massa, sem HTML). Não foi necessário aplicar stemming, remoção de stopwords ou normalização agressiva, técnicas que poderiam remover informação semântica relevante para o embedding.

**Trade-off:** uma limpeza mais agressiva poderia beneficiar abordagens baseadas em frequência (TF-IDF, BoW), mas prejudicaria modelos de embedding contextual que dependem da estrutura linguística do texto.

---

## 10. RAG — Recuperação por Similaridade Semântica

**Decisão:** RAG com FAISS (`IndexFlatL2`) sobre os 15k embeddings do corpus balanceado, integrado ao Streamlit

**Motivo:**
- Permite busca semântica sobre o corpus em tempo real
- Complementa a classificação mostrando relatos reais similares ao input do usuário
- FAISS é o padrão para RAG em produção, leve, rápido e sem dependência de servidor externo

**Fluxo:**
1. Usuário digita uma reclamação
2. Texto é vetorizado com o mesmo `paraphrase-multilingual-mpnet-base-v2`
3. FAISS busca os K vetores mais próximos por distância euclidiana (L2)
4. Relatos similares são exibidos com nota, status e empresa

**Por que sem LLM no RAG:** o Llama 3.2 foi testado localmente e mostrou ganho de F1, mas a latência no notebook inviabiliza uso interativo na demo. O FAISS sozinho entrega a experiência RAG em milissegundos.

---

## 11. Salvamento de Artefatos

**Decisão:** embeddings salvos em disco antes da apresentação

| Artefato | Arquivo |
|---|---|
| Embeddings 3 classes balanceado | `embeddings_3class_balanced.npy` |

**Motivo:** evita recomputação durante a demo ao vivo no notebook.