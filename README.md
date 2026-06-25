# classificacao-relatos-consumidores
Pipeline - Mineracaoo de textos

python3 -m venv .venv
pip install ollama pydantic sentence-transformers scikit-learn pandas numpy

Testes com 3 classes
Teste full

=======================================================
ETAPA 1 — Carregamento e amostragem
=======================================================
Dataset completo (após filtro): 203,810 registros
label
Positivo (2)    105732
Negativo (0)     84660
Neutro (1)       13418

Amostra: 203,810 registros
label
Positivo (2)    105732
Negativo (0)     84660
Neutro (1)       13418

=======================================================
ETAPA 2 — Pré-processamento
=======================================================
Textos vazios após limpeza: 4
Tamanho médio do relato: 675 chars

Distribuição de labels:
label
Positivo (2)    105732
Negativo (0)     84660
Neutro (1)       13418

=======================================================
ETAPA 3 — Embeddings
=======================================================
Modelo: paraphrase-multilingual-mpnet-base-v2
Vetorizando... (pode demorar alguns minutos no notebook)
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 8643.66it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3185/3185 [11:15<00:00,  4.71it/s]
Embeddings salvos em 'embeddings_3class_full.npy'
Shape dos embeddings: (203810, 768)

=======================================================
ETAPA 4 — Baseline ML
=======================================================
Treino: 163,048 | Teste: 40,762

--- Logistic Regression ---
              precision    recall  f1-score   support

    Negativo       0.63      0.52      0.57     16932
      Neutro       0.09      0.38      0.15      2684
    Positivo       0.72      0.54      0.61     21146

    accuracy                           0.52     40762
   macro avg       0.48      0.48      0.45     40762
weighted avg       0.64      0.52      0.57     40762


--- LinearSVC ---
              precision    recall  f1-score   support

    Negativo       0.61      0.62      0.61     16932
      Neutro       0.12      0.04      0.06      2684
    Positivo       0.67      0.72      0.69     21146

    accuracy                           0.63     40762
   macro avg       0.47      0.46      0.46     40762
weighted avg       0.61      0.63      0.62     40762


=======================================================
RESUMO
=======================================================
  Logistic Regression: 0.5669
  LinearSVC: 0.619

Teste no llm

=======================================================
Carregando dataset e embeddings...
=======================================================
label
Positivo (2)    259
Negativo (0)    208
Neutro (1)       33
Dataset: 500 linhas | Embeddings: (203810, 768)

=======================================================
Extraindo features com LLM (500 relatos)...
Modelo: llama3.2
=======================================================
  50/500 | erros: 1
  100/500 | erros: 1
  150/500 | erros: 6
  200/500 | erros: 7
  250/500 | erros: 12
  300/500 | erros: 13
  350/500 | erros: 15
  400/500 | erros: 21
  450/500 | erros: 27
  500/500 | erros: 28

Extraídos: 472 | Erros/ignorados: 28

=======================================================
Comparação — baseline vs. enriquecido com LLM
=======================================================

--- Embedding puro (F1 weighted: 0.4874) ---
              precision    recall  f1-score   support

    Negativo       0.49      0.47      0.48        40
      Neutro       0.00      0.00      0.00         6
    Positivo       0.52      0.59      0.55        49

    accuracy                           0.51        95
   macro avg       0.34      0.36      0.34        95
weighted avg       0.47      0.51      0.49        95


--- Embedding + LLM features (F1 weighted: 0.5361) ---
              precision    recall  f1-score   support

    Negativo       0.53      0.45      0.49        40
      Neutro       0.00      0.00      0.00         6
    Positivo       0.58      0.71      0.64        49

    accuracy                           0.56        95
   macro avg       0.37      0.39      0.38        95
weighted avg       0.52      0.56      0.54        95

=======================================================
Distribuição das features extraídas pelo LLM
=======================================================

categoria_problema:
categoria_problema
atendimento ruim       173
cobrança indevida      169
produto com defeito     40
cancelamento            38
atraso na entrega       25
outro                   18
fraude                   9

tom:
tom
frustrado     257
furioso       194
neutro         11
satisfeito     10

complexidade:
complexidade
alta     300
baixa    172




Testes com 2 classes

Teste de 20k

Treino: 16,000 | Teste: 4,000
--- Logistic Regression ---
               precision    recall  f1-score   support
Não Resolvido       0.67      0.49      0.57      1636
    Resolvido       0.70      0.83      0.76      2364
     accuracy                           0.69      4000
    macro avg       0.68      0.66      0.66      4000
 weighted avg       0.69      0.69      0.68      4000
--- LinearSVC ---
               precision    recall  f1-score   support
Não Resolvido       0.66      0.50      0.57      1636
    Resolvido       0.70      0.82      0.76      2364
     accuracy                           0.69      4000
    macro avg       0.68      0.66      0.66      4000
 weighted avg       0.69      0.69      0.68      4000
=======================================================
RESUMO
=======================================================
  Logistic Regression: 0.6812
  LinearSVC: 0.6809



Teste de 200k

=======================================================
ETAPA 1 — Carregamento e amostragem
=======================================================
Dataset completo: 204,031 registros
status
Resolvido        120599
Não Resolvido     83432

Amostra: 204,031 registros
status
Resolvido        120599
Não Resolvido     83432

=======================================================
ETAPA 2 — Pré-processamento
=======================================================
Textos vazios após limpeza: 4
Tamanho médio do relato: 675 chars

Distribuição de labels:
label
Resolvido (1)        120599
Não Resolvido (0)     83432

=======================================================
ETAPA 3 — Embeddings
=======================================================
Modelo: paraphrase-multilingual-mpnet-base-v2
Vetorizando...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 199/199 [00:00<00:00, 9334.33it/s]
Batches: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3188/3188 [11:07<00:00,  4.78it/s]
Embeddings salvos em 'embeddings_2class_full.npy'
Shape dos embeddings: (204031, 768)

=======================================================
ETAPA 4 — Baseline ML
=======================================================
Treino: 163,224 | Teste: 40,807

--- Logistic Regression ---
               precision    recall  f1-score   support

Não Resolvido       0.67      0.52      0.59     16687
    Resolvido       0.71      0.82      0.76     24120

     accuracy                           0.70     40807
    macro avg       0.69      0.67      0.67     40807
 weighted avg       0.70      0.70      0.69     40807


--- LinearSVC ---
               precision    recall  f1-score   support

Não Resolvido       0.68      0.52      0.59     16687
    Resolvido       0.71      0.83      0.77     24120

     accuracy                           0.70     40807
    macro avg       0.70      0.67      0.68     40807
 weighted avg       0.70      0.70      0.69     40807


=======================================================
RESUMO
=======================================================
  Logistic Regression: 0.6909
  LinearSVC: 0.6935