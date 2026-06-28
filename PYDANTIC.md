# Schema Pydantic — Extração de Features de Relatos

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