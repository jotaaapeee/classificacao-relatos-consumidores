# LOGS.md — Resultados dos Experimentos

## pipeline-3class-llm_TFIDF.py

=======================================================
ETAPA 1 — Carregamento e undersampling balanceado
=======================================================
Dataset balanceado: 15,000 registros
label
Positivo (2)    5000
Neutro (1)      5000
Negativo (0)    5000

=======================================================
ETAPA 2 — Vetorização TF-IDF
=======================================================
Shape TF-IDF: (15000, 10000)

=======================================================
ETAPA 3 — Baseline: TF-IDF + Logistic Regression
=======================================================
Treino: 12,000 | Teste: 3,000
              precision    recall  f1-score   support

    Negativo       0.52      0.50      0.51      1000
      Neutro       0.42      0.42      0.42      1000
    Positivo       0.51      0.52      0.52      1000

    accuracy                           0.48      3000
   macro avg       0.48      0.48      0.48      3000
weighted avg       0.48      0.48      0.48      3000

F1 weighted baseline: 0.4818

=======================================================
ETAPA 4 — Chamada LLM combinada (500 relatos)
Modelo: llama-3.1-8b-instant
=======================================================
  10/501 | erros: 0 | 0.6s/req | ~4.8min restantes
  50/501 | erros: 0 | 4.0s/req | ~29.7min restantes
 100/501 | erros: 0 | 4.6s/req | ~31.0min restantes
 200/501 | erros: 0 | 5.0s/req | ~25.0min restantes
 300/501 | erros: 0 | 5.1s/req | ~17.0min restantes
 400/501 | erros: 0 | 5.1s/req | ~8.6min restantes
 500/501 | erros: 0 | 5.1s/req | ~0.1min restantes

Extraídos: 501 | Erros: 0

=======================================================
ETAPA 5 — TF-IDF + LLM features
=======================================================
              precision    recall  f1-score   support

    Negativo       0.38      0.39      0.39        33
      Neutro       0.41      0.38      0.39        34
    Positivo       0.49      0.50      0.49        34

    accuracy                           0.43       101
   macro avg       0.42      0.43      0.42       101
weighted avg       0.43      0.43      0.43       101

F1 weighted: 0.4253
Features salvas em 'features_llm.csv'

=======================================================
ETAPA 6 — Zero-shot LLM (sem treinamento)
=======================================================
              precision    recall  f1-score   support

    Negativo       0.33      1.00      0.50       167
      Neutro       0.00      0.00      0.00       167
    Positivo       1.00      0.01      0.01       167

    accuracy                           0.34       501
   macro avg       0.44      0.34      0.17       501
weighted avg       0.44      0.34      0.17       501

F1 weighted: 0.1709

=======================================================
RESUMO COMPARATIVO
=======================================================
  Baseline (TF-IDF + LR):       0.4818  (15k relatos)
  TF-IDF + LLM features:        0.4253  (501 relatos)
  Zero-shot LLM:                0.1709  (501 relatos)

  Ganho features LLM vs base:   -0.0566
  Ganho zero-shot vs base:      -0.3110

=======================================================
Distribuição das features extraídas
=======================================================

categoria_problema:
categoria_problema
atendimento ruim       155
cobrança indevida      143
cancelamento            97
atraso na entrega       50
produto com defeito     34
fraude                  12
outro                   10

tom:
tom
furioso       456
frustrado      42
satisfeito      2
neutro          1

complexidade:
complexidade
alta     455
baixa     30
média     16


---

## rag.py


=======================================================
Carregando dataset...
=======================================================
Relatos indexados: 500
Empresas únicas: 172

Carregando modelo de embedding...
Loading weights: 100%|████████████| 199/199 [00:00<00:00, 1549.52it/s]

=======================================================
Montando índices...
=======================================================
  → fixo
  → overlap
  → hierarquico

 estratégia  nodes_indexados  tempo_s
       fixo              715     19.9
    overlap              734     19.4
hierarquico             1446     34.0

=======================================================
Avaliando estratégias...
=======================================================
qid  estratégia  keyword_recall@k  avg_score  context_words
q01        fixo               0.0      0.533            150
q01     overlap               0.0      0.538            199
q01 hierarquico               0.2      0.564            167

=======================================================
RESUMO — média por estratégia
=======================================================
 estratégia  keyword_recall_medio  avg_score_medio  context_words_medio
hierarquico                   0.2            0.564                167.0
       fixo                   0.0            0.533                150.0
    overlap                   0.0            0.538                199.0

=======================================================
INSPEÇÃO — q01 em todas as estratégias
=======================================================
Pergunta: Qual empresa tem mais reclamaçoes registradas?


--- fixo ---
  rank=1 score=0.544 empresa=Giga+ (antiga Sumicity)
  Já abri uma reclamação aqui no Procon e não foi resolvido pela empresa que me deixou sem respostas, porém minha avaliação expirou e acabei perdendo o prazo....
  rank=2 score=0.538 empresa=Credicard
  AGORA SE QUISEREM ACORDO VAI TER QUE SER DO MEU JEITO E DENTRO DAS MINHAS REAIS CONDIÇÕES... EU ESTAVA DISPOSTO A NEGOCIAR E CUMPRIR COM O ACORDO... MAS ME VIRARAM TOTALMENTE AS COSTAS. SÓ LAMENTO......
  rank=3 score=0.536 empresa=Crefisa
  Consumidor reclamar que perdeu 59 horas de relógio para contratação do empréstimo consignado bolsa família e informar que enviou todas as documentações solicitada e não houve o cumprimento da oferta c...
  rank=4 score=0.529 empresa=C&A
  GOSTARIA DE FAZER UMA RENEGOCIAÇÃO DE VALORES EM ABERTO...
  rank=5 score=0.520 empresa=Universidade Estácio de Sá
  Recebimento de cobrança constante devido a erro de cadastro...

--- overlap ---
  rank=1 score=0.549 empresa=Copasa - Companhia de Saneamento de Minas Gerais
  Assim, estou numa situação onde foi gerada uma conta para ser paga 15 dias antes do previsto e eu não tenho meios de pagá-la ou solicitar que o vencimento seja corrigido. Houve uma sequência de falhas...
  rank=2 score=0.544 empresa=Giga+ (antiga Sumicity)
  Já abri uma reclamação aqui no Procon e não foi resolvido pela empresa que me deixou sem respostas, porém minha avaliação expirou e acabei perdendo o prazo....
  rank=3 score=0.536 empresa=Crefisa
  Consumidor reclamar que perdeu 59 horas de relógio para contratação do empréstimo consignado bolsa família e informar que enviou todas as documentações solicitada e não houve o cumprimento da oferta c...
  rank=4 score=0.534 empresa=Oi Fixo
  Ao realizar o mesmo pedido com o CPF da minha companheira para o mesmo endereço, conseguimos. Isso demonstra que a empresa mais vez está me gerando transtornos injustificados....
  rank=5 score=0.529 empresa=C&A
  GOSTARIA DE FAZER UMA RENEGOCIAÇÃO DE VALORES EM ABERTO...

--- hierarquico ---
  rank=1 score=0.576 empresa=Copasa - Companhia de Saneamento de Minas Gerais
  Houve uma sequência de falhas, por parte da empresa reclamada que eu pude identificar e entrar em contato com eles solicitando alguma possibilidade de resolver, porém eles não ofereceram nenhum caminh...
  rank=2 score=0.573 empresa=Crefisa
  Tendo em visto propaganda enganosa que tem lesado o consumidor em si. Dessa forma solicitar providências Sobre pena de indenização por perdas e danos morais ocasionado por falhas e erros cometido pela...
  rank=3 score=0.565 empresa=Enjoei
  Não compreendo essa cobrança, pois tinha outros produtos à venda, sendo assim minha conta não estava inativa, e também não recebi nenhum tipo de aviso de que essa taxa seria cobrada. Tentei entrar em ...
  rank=4 score=0.556 empresa=Porto Seguro Cia de Seguros Gerais
  Preciso realizar a renovação nos próximos dias, e caso a empresa não retorne, procederei com reclamação em PROCON, ASF e outras associações envolvidas. Obrigado....
  rank=5 score=0.550 empresa=Smiles
  000 pontos na hora.....documentos em anexo, tanto da adesão quanto a promoção. Fiz a reclamação por tel e chat e ainda não está solucionado....


---

## Resumo

### Classificação

| Método | F1 weighted | Dataset |
|---|---|---|
| Baseline (TF-IDF + LR) | 0.4818 | 15k |
| TF-IDF + LLM features | 0.4253 | 501 |
| Zero-shot LLM | 0.1709 | 501 |

### RAG

| Estratégia | keyword_recall | avg_score | Nodes | Tempo (s) |
|---|---|---|---|---|
| hierarquico | 0.2 | 0.564 | 1446 | 34.0 |
| fixo | 0.0 | 0.533 | 715 | 19.9 |
| overlap | 0.0 | 0.538 | 734 | 19.4 |