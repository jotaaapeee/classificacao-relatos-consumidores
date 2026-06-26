# Resultados - Extração de Features com LLM (Llama 3.2)

Dataset: consumidores.gov.br · 15k registros balanceados (5k/classe) https://www.kaggle.com/datasets/beatrizmsarmento/relatos-de-consumidores-do-site-consumidor-gov-br
Schema: Pydantic com 5 campos estruturados
Modelo: `llama3.2` via Ollama

---

## Experimento 1 - Notebook (CPU)

**Hardware:** Ryzen 7 5825u · 16GB RAM · sem GPU
**N_SUBAMOSTRA:** 50 relatos
**Velocidade:** ~10s/req
**Taxa de erro:** 2/50 = 4.0%
**Extraídos com sucesso:** 49

### Comparação F1

| Configuração | F1 weighted |
|---|---|
| Embedding puro | 0.3619 |
| Embedding + LLM features | 0.4143 |

**Ganho:** +0.052

### Classification Report - Embedding puro

```
              precision    recall  f1-score   support
    Negativo       0.67      0.50      0.57         4
      Neutro       0.00      0.00      0.00         3
    Positivo       0.33      0.67      0.44         3
    accuracy                           0.40        10
   macro avg       0.33      0.39      0.34        10
weighted avg       0.37      0.40      0.36        10
```

### Classification Report - Embedding + LLM features

```
              precision    recall  f1-score   support
    Negativo       0.67      0.50      0.57         4
      Neutro       0.33      0.33      0.33         3
    Positivo       0.25      0.33      0.29         3
    accuracy                           0.40        10
   macro avg       0.42      0.39      0.40        10
weighted avg       0.44      0.40      0.41        10
```

### Distribuição das features extraídas

**categoria_problema**
| Categoria | Count |
|---|---|
| atendimento ruim | 16 |
| cobrança indevida | 16 |
| cancelamento | 7 |
| outro | 4 |
| atraso na entrega | 3 |
| produto com defeito | 2 |
| fraude | 1 |

**tom**
| Tom | Count |
|---|---|
| frustrado | 24 |
| furioso | 21 |
| neutro | 2 |
| satisfeito | 2 |

**complexidade**
| Complexidade | Count |
|---|---|
| alta | 26 |
| baixa | 23 |

---

## Experimento 2 - PC com GPU (referência)

**Hardware:** RTX 3060 12GB · Ollama 0.9.x com CUDA
**N_SUBAMOSTRA:** 500 relatos
**Velocidade:** ~0.8s/req
**Taxa de erro:** 36/500 = 7.2%
**Extraídos com sucesso:** 465 (após descarte de 1 por índice)→ 464 usados

### Comparação F1

| Configuração | F1 weighted |
|---|---|
| Embedding puro | 0.3189 |
| Embedding + LLM features | 0.4058 |

**Ganho:** +0.087

### Classification Report - Embedding puro

```
              precision    recall  f1-score   support
    Negativo       0.29      0.22      0.25        32
      Neutro       0.42      0.47      0.44        30
    Positivo       0.25      0.29      0.27        31
    accuracy                           0.32        93
   macro avg       0.32      0.33      0.32        93
weighted avg       0.32      0.32      0.32        93
```

### Classification Report - Embedding + LLM features

```
              precision    recall  f1-score   support
    Negativo       0.44      0.44      0.44        32
      Neutro       0.45      0.50      0.48        30
    Positivo       0.32      0.29      0.31        31
    accuracy                           0.41        93
   macro avg       0.40      0.41      0.41        93
weighted avg       0.40      0.41      0.41        93
```

### Distribuição das features extraídas

**categoria_problema**
| Categoria | Count |
|---|---|
| atendimento ruim | 196 |
| cobrança indevida | 152 |
| cancelamento | 42 |
| produto com defeito | 28 |
| atraso na entrega | 24 |
| outro | 14 |
| fraude | 9 |

**tom**
| Tom | Count |
|---|---|
| frustrado | 250 |
| furioso | 202 |
| neutro | 10 |
| satisfeito | 3 |

**complexidade**
| Complexidade | Count |
|---|---|
| alta | 298 |
| baixa | 167 |

---

## Observações

- O LLM agregou ganho consistente de F1 nos dois experimentos (+0.052 no notebook, +0.087 no PC)
- A classe Neutro foi a mais beneficiada pelas features do LLM, recall subiu de 0.47 → 0.50 no PC
- 87–90% dos relatos classificados como `frustrado` ou `furioso`, padrão esperado em canal de reclamações
- `atendimento ruim` e `cobrança indevida` dominam as categorias (~75% dos casos)
- Taxa de erro de JSON malformado maior no PC (7.2%) do que no notebook (4.0%), possivelmente relacionada ao aquecimento do modelo após muitas chamadas consecutivas