# classificacao-relatos-consumidores
Pipeline - Mineracaoo de textos

python3 -m venv .venv
pip install ollama pydantic sentence-transformers scikit-learn pandas numpy

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



Teste de 40k

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