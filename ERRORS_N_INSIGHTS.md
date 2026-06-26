# Análise de Erros e Insights Acionáveis

---

## Análise de Erro Estruturada

### Categoria 1 - Alucinação do LLM (JSON malformado)

**Observações:** 36 de 500 chamadas retornaram JSON inválido ou com campos fora do schema Pydantic.

**Exemplos observados:**
- LLM retornou texto explicativo antes do JSON, quebrando o parse
- Campo `complexidade` com valor `"media"` em vez de `"média"` (sem acento)
- Campo `tom` com valor `"irritado"` em vez de `"furioso"`

**Hipótese de causa:** o Llama 3.2 (3B parâmetros) tem capacidade limitada de seguir instruções estritas de formato em textos longos. Relatos acima de 800 chars aumentam a chance de desvio do schema.

**Ação proposta:** adicionar few-shot examples no prompt (1-2 exemplos de input/output corretos) e implementar retry automático com temperatura 0 antes de descartar o registro (sugerido pelo Claude).

---

### Categoria 2 - Erro de Classificação na classe neutro

**Observações:** a classe Neutro (nota 3) apresenta F1 consistentemente mais baixo mesmo após undersampling balanceado.

**Exemplos observados:**
- Relatos com tom neutro e problema resolvido parcialmente classificados como Negativo
- Reclamações com linguagem formal e sem carga emocional confundidas com Positivo

**Hipótese de causa:** a nota 3 reflete ambiguidade real do consumidor, nem satisfeito nem insatisfeito. O texto do relato frequentemente não sinaliza isso de forma explícita, tornando a classe semanticamente difusa para o embedding.

**Ação proposta:** explorar features adicionais como `menciona_resolucao_parcial` e `tom_ambiguo` no schema Pydantic para tentar capturar essa nuance.

---

### Categoria 3 - Ambiguidade do Dado

**Observações:** relatos com linguagem intensa e nota alta (Positivo) ou relatos curtos e vagos com nota baixa (Negativo).

**Exemplo observado:**
- "Péssimo atendimento mas resolveram no final" -> nota 4 (Positivo), mas embedding captura tom negativo -> classificado como Negativo

**Hipótese de causa:** a nota reflete a satisfação final do consumidor, mas o texto do relato descreve o processo, que pode ter sido negativo mesmo com desfecho positivo. Há desalinhamento semântico entre texto e rótulo.

**Ação proposta:** usar a coluna `comentario` (avaliação pós-resolução) como feature complementar, já que tende a refletir melhor a satisfação final do que o `relato`.

---

## Três Insights Acionáveis para o Cliente/Gestor

### Insight 1 - Priorizar canais de resolução de cobrança e atendimento

Das reclamações analisadas, **atendimento ruim** (42%) e **cobrança indevida** (33%) somam aproximadamente 75% dos casos. Empresas que investirem em automação de triagem para essas duas categorias, identificando e redirecionando automaticamente, têm potencial de resolver a maioria dos casos de forma mais rápida e eficiente.

**Ação sugerida:** implementar classificador automático de categoria na entrada da reclamação e criar filas de atendimento prioritário para cobrança indevida, que tende a ter maior impacto financeiro para o consumidor.

---

### Insight 2 - Tom como indicador de urgência e risco de escalada

Quase 90% dos relatos apresentam tom **frustrado** ou **furioso**. Relatos com tom `furioso` têm maior probabilidade de escalada para órgãos reguladores (Procon, Anatel, etc.). Um sistema de triagem por tom permitiria priorizar atendimento humano para esses casos antes que escalem.

**Ação sugerida:** usar o campo `tom` extraído pelo LLM como flag de urgência no sistema de CRM, relatos `furiosos` entram em fila prioritária com SLA reduzido.

---

### Insight 3 - Nota 3 como oportunidade de reversão

Consumidores com nota 3 (Neutro) estão indecisos, não abandonaram a empresa mas não estão satisfeitos. Este grupo (nem 10% do corpus) representa a maior oportunidade de reversão com baixo custo: uma ação proativa de acompanhamento pós-resolução pode converter esse consumidor em promotor.

**Ação sugerida:** identificar automaticamente reclamações com nota 3 e acionar fluxo de pós-atendimento (pesquisa de satisfação, oferta de compensação simbólica) dentro de 48h após o fechamento do caso.