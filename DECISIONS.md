# DECISOES.md — Classificação de Relatos do Consumidor

## Visão Geral

Pipeline de classificação supervisionada (Caminho B) aplicado ao dataset público do consumidores.gov.br, com embeddings semânticos, baseline ML clássico, extração de features via LLM e RAG com LlamaIndex.

---

## 1. Escolha do Dataset

**Decisão:** consumidores.gov.br (204k reclamações em PT-BR)
https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br

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

**Decisão:** `paraphrase-multilingual-mpnet-base-v2` (pipeline de classificação) e `paraphrase-multilingual-MiniLM-L12-v2` (RAG/LlamaIndex)

**Motivo:**
- Suporte nativo a PT-BR
- Roda em CPU (viável no notebook de desenvolvimento)
- Gratuito e sem dependência de API externa

**Trade-off:** mpnet (768 dims) tem representação semântica superior ao MiniLM (384 dims), mas é mais lento. O MiniLM foi escolhido para o RAG por ser mais leve no contexto de indexação de múltiplos chunks.

---

## 4. Estratégia de Amostragem — Undersampling Balanceado

**Decisão:** undersampling balanceado com 5.000 registros por classe (15.000 total)

**Motivo:** o dataset original é desbalanceado (~105k Positivos, ~84k Negativos, ~13k Neutros). Com amostragem proporcional, o modelo ignorava completamente a classe Neutro (F1 = 0.00). O undersampling nivelar pelo Neutro resolve o desbalanceamento sem precisar de `class_weight='balanced'`, tornando as métricas mais confiáveis e comparáveis entre classes.

**Comparação:**
| Estratégia | F1 Neutro | F1 weighted |
|---|---|---|
| Proporcional (desbalanceado) | 0.00 | ~0.69 (enganoso) |
| Undersampling 5k/classe | ~0.40 | ~0.46 (honesto) |

O F1 geral cai, mas passa a refletir o desempenho real nas 3 classes.

**Implementação:** substituiu `groupby().apply()` por loop manual — incompatibilidade com pandas no Python 3.12/3.14.

---

## 5. Vetorização — TF-IDF

**Decisão:** TF-IDF com `max_features=10.000`, `sublinear_tf=True`, `min_df=2`, `ngram_range=(1, 2)`

**Motivo:** após testes comparativos, o TF-IDF com Logistic Regression superou embeddings densos na tarefa de 3 classes (F1 0.4818 vs 0.4619). Textos de reclamação com ~673 chars têm vocabulário suficiente para o TF-IDF capturar bem as categorias.

**Comparação com embeddings:**
| Vetorização | LR F1 | LinearSVC F1 |
|---|---|---|
| Embedding (mpnet) | 0.4619 | 0.4637 |
| TF-IDF | **0.4818** | 0.4468 |

**Trade-off:** TF-IDF não captura contexto semântico (ex: "produto não chegou" ≠ "entrega atrasada"), mas é mais eficiente computacionalmente e interpretável.

---

## 6. Classificadores Baseline

**Decisão:** Logistic Regression e LinearSVC

**Motivo:** modelos lineares funcionam bem sobre matrizes TF-IDF esparsas e permitem comparação direta antes e depois da adição de features do LLM.

**Resultados — baseline 3 classes (balanceado, 15k):**
| Modelo | F1 weighted |
|---|---|
| Logistic Regression | 0.4818 |
| LinearSVC | 0.4468 |

---

## 7. LLM via API (Groq / Llama 3.1 8B)

**Decisão:** Llama 3.1 8B via Groq API (plano gratuito)

**Motivo:**
- Portável — não depende de hardware local
- Gratuito até 500k tokens/dia
- Velocidade adequada (~0.5–5s/req dependendo do rate limit)

**Trade-off:** rate limit de 6k tokens/minuto no plano gratuito causou 87 erros em 500 chamadas a partir de ~420 requisições consecutivas. Recomendado `time.sleep(6)` entre chamadas para respeitar o limite.

**Nota:** experimentos anteriores foram realizados com Llama 3.2 via Ollama local (GPU RTX 3060), com velocidade ~0.8s/req e taxa de erro de 7.2%. A migração para Groq API torna o projeto reproduzível sem hardware especializado.

---

## 8. Schema Pydantic — Extração Combinada

**Decisão:** schema Pydantic com 6 campos, incluindo classificação zero-shot na mesma chamada

**Campos extraídos:**
- `classificacao`: Negativo / Neutro / Positivo (zero-shot)
- `categoria_problema`: 7 categorias
- `tom`: neutro / frustrado / furioso / satisfeito
- `menciona_valor_financeiro`: bool
- `menciona_prazo`: bool
- `complexidade`: baixa / média / alta

**Motivo da unificação:** extrair features e classificação zero-shot na mesma chamada reduz o consumo de tokens e o tempo de execução à metade em relação a duas chamadas separadas.

**Resultado (414 relatos válidos, dataset balanceado 3 classes):**
| Configuração | F1 weighted |
|---|---|
| TF-IDF puro | 0.4818 (15k) |
| TF-IDF + LLM features | 0.3684 (414) |
| Zero-shot LLM | 0.2371 (414) |

---

## 9. Tokenização e Representação Semântica

**Tokenização:** o modelo `paraphrase-multilingual-mpnet-base-v2` usa tokenização WordPiece (mesmo estilo do BERT), que quebra o texto em subpalavras. Exemplo: "cancelamento" → `["cancel", "##amento"]`. Isso permite que o modelo lide com palavras desconhecidas e variações morfológicas do português.

**Captura de contexto:** o modelo usa atenção bidirecional — o vetor de cada token é influenciado por todos os outros tokens da frase. O embedding final representa o significado do texto inteiro, não palavras isoladas. Isso é o diferencial em relação ao TF-IDF, onde cada termo é tratado de forma independente.

**Exemplo prático:** "produto não chegou" e "entrega atrasada" geram embeddings próximos no espaço vetorial mesmo sem compartilhar palavras — porque o modelo captura a equivalência semântica entre as duas frases.

---

## 10. Limpeza de Texto

**Decisão:** limpeza mínima — remoção de URLs e normalização de espaços

**Motivo:** o dataset do consumidores.gov.br já vem bem estruturado e limpo. Os relatos são textos escritos por consumidores em um formulário web, o que naturalmente reduz ruído. Não foi necessário aplicar stemming, remoção de stopwords ou normalização agressiva — técnicas que poderiam remover informação semântica relevante.

**Trade-off:** limpeza mais agressiva poderia beneficiar o TF-IDF, mas prejudicaria modelos de embedding contextual que dependem da estrutura linguística do texto.

---

## 11. RAG com LlamaIndex

**Decisão:** RAG com LlamaIndex comparando 3 estratégias de chunking sobre 500 relatos

**Estratégias:**
- `fixo`: SentenceSplitter com chunk_size=256, overlap=0
- `overlap`: SentenceSplitter com chunk_size=256, overlap=50
- `hierárquico`: HierarchicalNodeParser com chunk_sizes=[512, 256, 128]

**Motivo do ajuste de 64→128 no hierárquico:** o LlamaIndex emitia aviso de chunks abaixo do tamanho mínimo recomendado com chunk_size=64. O ajuste para 128 eliminou o aviso sem impactar a qualidade de recuperação.

**Embedding usado:** `paraphrase-multilingual-MiniLM-L12-v2` — modelo mais leve que o mpnet, adequado para indexação de múltiplos chunks em CPU.

**Resultados:**
| Estratégia | Nodes indexados | keyword_recall médio | avg_score médio |
|---|---|---|---|
| hierárquico | 1446 | 0.200 | 0.564 |
| fixo | 715 | 0.000 | 0.533 |
| overlap | 734 | 0.000 | 0.538 |

**Observação:** scores de similaridade próximos entre estratégias (~0.53–0.56) indicam que o corpus de relatos é semanticamente homogêneo — reclamações compartilham vocabulário similar independente do tema específico.

---

## 12. Salvamento de Artefatos

**Decisão:** embeddings salvos em disco antes da apresentação

| Artefato | Arquivo |
|---|---|
| Embeddings 3 classes balanceado | `embeddings_3class_balanced.npy` |
| Features LLM extraídas | `features_llm.csv` |

**Motivo:** evita recomputação durante a demo ao vivo no notebook.

---

## 13. Lista de palavras mais frequentes - NLTK e spaCy

**Decisão:** listagem das palavras mais frequentes tanto no NLTK quanto no spaCy

**NLTK – tokenização + stopwords**

***Top 30 palavras mais frequentes (NLTK):***
| Rank | Palavra | Frequência |
|---|---|---|
| 1 | dia | 11,725 |
| 2 | valor | 8,412 |
| 3 | empresa | 6,516 |
| 4 | contato | 6,109 |
| 5 | compra | 5,328 |
| 6 | pois | 5,074 |
| 7 | r | 4,978 |
| 8 | conta | 4,766 |
| 9 | dias | 4,129 |
| 10 | pedido | 3,866 |
| 11 | cartão | 3,829 |
| 12 | data | 3,744 |
| 13 | banco | 3,689 |
| 14 | fiz | 3,679 |
| 15 | cancelamento | 3,604 |
| 16 | pagamento | 3,500 |
| 17 | produto | 3,388 |
| 18 | site | 3,365 |
| 19 | porém | 3,299 |
| 20 | sendo | 3,212 |
| 21 | recebi | 2,963 |
| 22 | problema | 2,881 |
| 23 | reembolso | 2,870 |
| 24 | após | 2,714 |
| 25 | fazer | 2,683 |
| 26 | nome | 2,678 |
| 27 | entrei | 2,673 |
| 28 | prazo | 2,665 |
| 29 | voo | 2,653 |
| 30 | ainda | 2,586 |

**spaCy – lematização + stopwords**

***Classe: Negativo – termos com maior coeficiente positivo***
| Rank | Termo | Coef |
|---|---|---|
| 1 | hurb | 2.5370 |
| 2 | pacote | 2.3009 |
| 3 | dinheiro | 1.5759 |
| 4 | aparelho | 1.5624 |
| 5 | samsung | 1.4991 |
| 6 | datas | 1.4933 |
| 7 | hotel | 1.4104 |
| 8 | placa | 1.3404 |
| 9 | fato | 1.2961 |
| 10 | 60 dias | 1.2954 |
| 11 | solução | 1.2722 |
| 12 | original | 1.2302 |
| 13 | pacote viagem | 1.2246 |
| 14 | resposta | 1.2200 |
| 15 | defeito | 1.1632 |

***Classe: Neutro – termos com maior coeficiente positivo***
| Rank | Termo | Coef |
|---|---|---|
| 1 | pix | 1.0927 |
| 2 | entregou | 1.0675 |
| 3 | pagina | 1.0609 |
| 4 | dívidas | 0.9749 |
| 5 | constar | 0.9443 |
| 6 | prazo entrega | 0.9385 |
| 7 | informação | 0.9356 |
| 8 | espera | 0.8865 |
| 9 | descaso | 0.8852 |
| 10 | inicial | 0.8850 |
| 11 | determinado | 0.8839 |
| 12 | cartao | 0.8546 |
| 13 | várias vezes | 0.8452 |
| 14 | item | 0.8437 |
| 15 | riachuelo | 0.8365 |

***Classe: Positivo – termos com maior coeficiente positivo***
Rank   Termo                                   Coef |
|---|---|---|
| 1 | passagem | 2.2484 |
| 2 | internet | 2.2134 |
| 3 | plano | 2.1495 |
| 4 | azul | 1.9181 |
| 5 | tim | 1.9045 |
| 6 | voo | 1.7204 |
| 7 | fatura | 1.6996 |
| 8 | remarcação | 1.6645 |
| 9 | instalação | 1.3201 |
| 10 | alteração | 1.2745 |
| 11 | titularidade | 1.2453 |
| 12 | mudança | 1.2193 |
| 13 | olá | 1.1679 |
| 14 | latam | 1.1584 |
| 15 | pontos | 1.1441 |

**Motivo:** verificar como a tokenização funciona e como pode auxiliar na análise de sentimentos.
