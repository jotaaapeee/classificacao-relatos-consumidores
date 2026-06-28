# DECISOES.md — Decisões Técnicas do Projeto

Documento com as principais decisões de arquitetura e suas justificativas. Para visão geral e resultados, consulte README.md.

---

## 1. TF-IDF vs Sentence-Transformer

**Decisão:** TF-IDF com max_features=10.000, sublinear_tf=True, min_df=2, ngram_range=(1, 2)

**Motivo:** TF-IDF + Logistic Regression superou embeddings densos (sentence-transformer) na tarefa de 3 classes:

| Vetorização | F1 (LR) |
|---|---|
| Sentence-Transformer (mpnet) | 0.4619 |
| TF-IDF | **0.4818** |

**Vantagens do TF-IDF:**
- Coeficientes interpretáveis (permite identificar termos mais relevantes por classe)
- Matriz esparsa → mais eficiente
- Independente de modelo pré-treinado externo

**Trade-off:** não captura contexto semântico, mas compensado pelas features do LLM.

---

## 2. Undersampling Balanceado (5k/classe)

**Decisão:** undersampling com 5.000 registros por classe (15.000 total)

**Motivo:** classe Neutro é minoritária (~13k vs ~105k Positivos). Com amostragem proporcional, o modelo ignorava Neutro (F1 = 0.00). O undersampling nivelou as classes e tornou as métricas comparáveis.

| Estratégia | F1 Neutro | F1 weighted |
|---|---|---|
| Proporcional | 0.00 | ~0.69 (enganoso) |
| Undersampling | ~0.40 | ~0.46 (honesto) |

---

## 3. Extração Combinada (LLM)

**Decisão:** extrair features + classificação zero-shot em UMA chamada à API

**Motivo:** reduz consumo de tokens e tempo de execução pela metade vs duas chamadas separadas.

**Schema Pydantic (6 campos):**
- classificacao (zero-shot)
- categoria_problema (7 opções)
- tom (4 opções)
- menciona_valor_financeiro (bool)
- menciona_prazo (bool)
- complexidade (3 níveis)

---

## 4. RAG — Chunking Hierárquico

**Decisão:** HierarchicalNodeParser com chunk_sizes=[512, 256, 128]

**Motivo:** foi a única estratégia que recuperou contexto relevante (keyword_recall = 0.200). Estratégias fixas (com e sem overlap) tiveram recall = 0.000.

| Estratégia | Nodes | keyword_recall |
|---|---|---|
| hierárquico | 1446 | **0.200** |
| fixo | 715 | 0.000 |
| overlap | 734 | 0.000 |

**Trade-off:** hierárquico indexa ~2x mais nodes, mas o ganho em recall justifica.

---

## 5. Limpeza de Texto

**Decisão:** limpeza mínima — remoção de URLs + normalização de espaços

**Motivo:** dataset já vem bem estruturado. Não foi necessário stemming, stopwords ou normalização agressiva — técnicas que poderiam remover informação relevante.

---

## 6. NLTK vs spaCy (Análise Exploratória)

**Decisão:** utilizar NLTK para análises rápidas; spaCy para tarefas que exigem lematização

**Diferenças observadas:**
- NLTK: mais rápido, mas inclui ruído ("r" como token) e mantém plurais separados
- spaCy: lematização consolida variações ("dias" → "dia"), representação mais limpa

**Conclusão:** para classificação com TF-IDF, a diferença é marginal. spaCy seria mais útil em tarefas de análise semântica mais profunda.

---

## 7. Principais Lições Aprendidas

| Desafio | Solução |
|---|---|
| Rate limit Groq (6k tokens/min) → 17.4% erros | time.sleep(6) entre chamadas |
| Zero-shot enviesado para Negativo | Informar no prompt que classes são balanceadas |
| Classe Neutro semanticamente difusa | Explorar features como `menciona_resolucao_parcial` |
| Tom `furioso` colapsou 93% dos casos | Redefinir critérios no prompt ou simplificar schema |

---

## 8. Por que o LLM não superou o baseline?

| Método | F1 | Motivo |
|---|---|---|
| TF-IDF + LR | **0.4818** | 15k exemplos rotulados → dados suficientes |
| TF-IDF + LLM features | 0.3684 | Apenas 414 amostras válidas (rate limit prejudicou) |
| Zero-shot | 0.2371 | Modelo 8B não calibrado para classes balanceadas |

**Conclusão:** com dados rotulados suficientes, ML clássico supera LLM. O LLM agrega valor como extrator de features estruturadas, mas requer volume maior de amostras ou melhor calibração.