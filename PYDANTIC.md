# Schema Pydantic — Extração de Features de Relatos

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