# Análise de Erros e Insights Acionáveis

---

## Análise de Erro Estruturada

### Categoria 1 — Erro de Classificação — Classe Neutro

**Observações:** a classe Neutro (nota 3) apresenta F1 consistentemente mais baixo (~0.39–0.42) mesmo após undersampling balanceado.

**Exemplos observados:**
- Relatos com tom neutro e problema resolvido parcialmente classificados como Negativo
- Reclamações com linguagem formal e sem carga emocional confundidas com Positivo

**Hipótese de causa:** a nota 3 reflete ambiguidade real do consumidor — nem satisfeito nem insatisfeito. O texto do relato frequentemente não sinaliza isso de forma explícita.

**Ação proposta:** explorar features adicionais como `menciona_resolucao_parcial` e `tom_ambiguo` no schema Pydantic para capturar essa nuance.

---

### Categoria 2 — Ambiguidade do Dado

**Observações:** relatos com linguagem intensa e nota alta (Positivo) ou relatos curtos e vagos com nota baixa (Negativo).

**Exemplo observado:**
- "Péssimo atendimento mas resolveram no final" → nota 4 (Positivo), mas embedding captura tom negativo → classificado como Negativo

**Hipótese de causa:** a nota reflete a satisfação final do consumidor, mas o texto descreve o processo — que pode ter sido negativo mesmo com desfecho positivo. Há desalinhamento semântico entre texto e rótulo.

**Ação proposta:** usar a coluna `comentario` (avaliação pós-resolução) como feature complementar.

---

### Categoria 3 — Colapso Semântico no Campo `tom`

**Observações:** o Llama 3.1 8B classificou 456/501 (91%) relatos como `furioso` e apenas 42 como `frustrado`, colapsando praticamente toda a escala emocional.

**Hipótese de causa:** o modelo 8B não distingue bem as nuances entre `frustrado` e `furioso` em português. A distinção é sutil e requer capacidade de raciocínio contextual que modelos menores tendem a simplificar.

**Ação proposta:** redefinir critérios no prompt com exemplos concretos ou simplificar o schema para dois níveis (`negativo` / `positivo`).

---

### Categoria 4 — Viés de Classe no Zero-shot

**Observações:** zero-shot atribuiu `Negativo` a todos os relatos (recall 1.00 em Negativo, 0.00 em Neutro e Positivo), F1 = 0.1709.

| Classe | Precision | Recall | F1 |
|---|---|---|---|
| Negativo | 0.33 | 1.00 | 0.50 |
| Neutro | 0.00 | 0.00 | 0.00 |
| Positivo | 1.00 | 0.01 | 0.01 |

**Hipótese de causa:** o LLM não sabe que as classes são balanceadas. Como reclamações são por natureza negativas, o modelo assume que quase tudo é Negativo.

**Ação proposta:** informar no prompt que as classes são igualmente distribuídas e fornecer critérios claros ancorados na nota (1–2 = Negativo, 3 = Neutro, 4–5 = Positivo).

---

## Observações sobre o RAG com LlamaIndex

**Avaliação:** 1 pergunta ("Qual empresa tem mais reclamações registradas?") com keywords: ["defeito", "troca", "devolução", "produto", "garantia"]

| Estratégia | keyword_recall | avg_score | Nodes |
|---|---|---|---|
| hierárquico | **0.2** | 0.564 | 1446 |
| fixo | 0.0 | 0.533 | 715 |
| overlap | 0.0 | 0.538 | 734 |

**Scores de similaridade homogêneos:** as três estratégias apresentaram avg_score próximo (0.533–0.564), indicando que o corpus de relatos é semanticamente homogêneo — reclamações compartilham vocabulário similar independente do tema.

**keyword_recall:** apenas o hierárquico recuperou contexto relevante. Estratégias fixas perderam contexto semântico.

**Escala:** corpus de 500 relatos limita a diversidade. Escalar para 5k+ provavelmente diferenciaria melhor as estratégias.

---

## Três Insights Acionáveis

### Insight 1 — Priorizar cobrança indevida e atendimento

**atendimento ruim** (155 casos) e **cobrança indevida** (143 casos) somam ~60% das reclamações. Empresas que automatizarem a triagem dessas duas categorias podem resolver a maioria dos casos mais rápido.

**Ação:** implementar classificador automático de categoria na entrada da reclamação e criar filas prioritárias para cobrança indevida (maior impacto financeiro).

---

### Insight 2 — Tom como indicador de urgência

91% dos relatos têm tom `furioso`. Relatos com este tom têm maior probabilidade de escalada para órgãos reguladores (Procon, Anatel, etc.).

**Ação:** usar o campo `tom` extraído pelo LLM como flag de urgência no CRM — relatos `furiosos` entram em fila prioritária com SLA reduzido.

---

### Insight 3 — Nota 3 como oportunidade de reversão

Consumidores com nota 3 (Neutro) estão indecisos — não abandonaram a empresa mas não estão satisfeitos. Este grupo representa a maior oportunidade de reversão com baixo custo.

**Ação:** identificar automaticamente reclamações com nota 3 e acionar fluxo de pós-atendimento (pesquisa de satisfação, oferta de compensação simbólica) dentro de 48h após o fechamento do caso.