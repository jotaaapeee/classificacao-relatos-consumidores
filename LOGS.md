# Resultados — Extração de Features com LLM

Dataset: consumidores.gov.br · 15k registros balanceados (5k/classe)
https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br

Schema: Pydantic com 6 campos estruturados (incluindo classificação zero-shot)

---

## Pipeline Unificado — Groq API (llama-3.1-8b-instant)

**N_SUBAMOSTRA:** 500 relatos  
**Extraídos com sucesso:** 414 | **Erros:** 87 (17.4% — rate limit a partir de ~420 chamadas)  
**Vetorização:** TF-IDF (10k features, bigrams, sublinear_tf)

### Resumo Comparativo

| Configuração | F1 weighted | Relatos |
|---|---|---|
| Baseline (TF-IDF + LR) | 0.4818 | 15k |
| TF-IDF + LLM features | 0.3684 | 414 |
| Zero-shot LLM | 0.2371 | 414 |

### Classification Report — Baseline (TF-IDF + LR)

```
              precision    recall  f1-score   support
    Negativo       0.52      0.50      0.51      1000
      Neutro       0.42      0.42      0.42      1000
    Positivo       0.51      0.52      0.52      1000
    accuracy                           0.48      3000
   macro avg       0.48      0.48      0.48      3000
weighted avg       0.48      0.48      0.48      3000
```

### Classification Report — TF-IDF + LLM features

```
              precision    recall  f1-score   support
    Negativo       0.44      0.47      0.46        34
      Neutro       0.39      0.55      0.46        33
    Positivo       0.00      0.00      0.00        16
    accuracy                           0.41        83
   macro avg       0.28      0.34      0.30        83
weighted avg       0.34      0.41      0.37        83
```

### Classification Report — Zero-shot LLM

```
              precision    recall  f1-score   support
    Negativo       0.40      1.00      0.58       167
      Neutro       0.00      0.00      0.00       167
    Positivo       1.00      0.01      0.02        80
    accuracy                           0.41       414
   macro avg       0.47      0.34      0.20       414
weighted avg       0.36      0.41      0.24       414
```

### Distribuição das features extraídas

**categoria_problema**
| Categoria | Count |
|---|---|
| atendimento ruim | 123 |
| cobrança indevida | 122 |
| cancelamento | 76 |
| atraso na entrega | 45 |
| produto com defeito | 31 |
| fraude | 11 |
| outro | 6 |

**tom**
| Tom | Count |
|---|---|
| furioso | 386 |
| frustrado | 25 |
| satisfeito | 2 |
| neutro | 1 |

**complexidade**
| Complexidade | Count |
|---|---|
| alta | 380 |
| baixa | 26 |
| média | 8 |

---

## Observações

- **Baseline TF-IDF + LR supera ambas as abordagens LLM** — com 15k exemplos rotulados, o classificador clássico é mais robusto
- **TF-IDF + LLM features (0.3684)** ficou abaixo do baseline por limitação de subamostra (414 relatos vs 12k de treino) e erros de rate limit que prejudicaram a distribuição das classes
- **Zero-shot (0.2371)** apresentou viés forte para a classe Negativo (recall 1.00), ignorando Neutro e Positivo — comportamento esperado em modelos menores (8B) sem exemplos de calibração
- **Rate limit Groq** (6k tokens/min no plano gratuito) causou 87 erros a partir de ~420 chamadas — recomendado `time.sleep(6)` para execuções futuras
- **Campo `tom`** apresentou colapso semântico no Groq: 386 "furioso" vs 25 "frustrado", sugerindo que o modelo de 8B não distingue bem os dois tons em português
- `atendimento ruim` e `cobrança indevida` dominam as categorias (~59% dos casos), consistente com execuções anteriores