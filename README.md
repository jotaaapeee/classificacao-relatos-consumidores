# classificacao-relatos-consumidores
Pipeline - Mineracaoo de textos

python3 -m venv .venv
pip install sentence-transformers scikit-learn pandas numpy


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