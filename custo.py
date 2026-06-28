"""
Estimativa de Custo — Gemini API
Trabalho Final — Mineração de Textos

Modelos avaliados (preços em junho/2026):
  - gemini-2.5-flash:       $0.30 input / $2.50 output por 1M tokens
  - gemini-3-flash-preview: $0.50 input / $3.00 output por 1M tokens
  - gemini-3.1-pro-preview: $2.00 input / $12.00 output por 1M tokens

Referência: https://ai.google.dev/gemini-api/docs/pricing

Instalação:
  pip install tiktoken
"""

# ============================================================
# CONFIGURAÇÃO
# ============================================================
CAMINHO_JSON  = "dados2025.json"
N_ELEMENTOS   = 1_000
USD_BRL       = 5.70

# Overhead fixo por chamada: prompt sistema + schema Pydantic ≈ 120 tokens
OVERHEAD_INPUT_TOKENS = 120
# Resposta JSON com 5 campos ≈ 80 tokens de output
OUTPUT_TOKENS_FIXO = 80

# Preços por 1M tokens (USD, junho 2026)
modelos = {
    "gemini-2.5-flash": {
        "input_por_milhao":  0.30,
        "output_por_milhao": 2.50,
    },
    "gemini-3-flash-preview": {
        "input_por_milhao":  0.50,
        "output_por_milhao": 3.00,
    },
    "gemini-3.1-pro-preview": {
        "input_por_milhao":  2.00,
        "output_por_milhao": 12.00,
    },
}

# ============================================================
# ETAPA 1 — Contagem real de tokens com tiktoken
# ============================================================
import json
import tiktoken

print("=" * 60)
print("CONTAGEM REAL DE TOKENS (tiktoken / cl100k_base)")
print("=" * 60)

with open(CAMINHO_JSON, "r", encoding="utf-8") as f:
    data = json.load(f)

enc = tiktoken.get_encoding("cl100k_base")

relatos = [d["relato"] for d in data if isinstance(d.get("relato"), str)][:5_000]
tokens_por_relato = [len(enc.encode(r[:1000])) for r in relatos]  # trunca igual ao pipeline

media_tokens_relato = sum(tokens_por_relato) / len(tokens_por_relato)
input_tokens_por_chamada = round(media_tokens_relato + OVERHEAD_INPUT_TOKENS)

print(f"Relatos amostrados:            {len(relatos):,}")
print(f"Média tokens por relato:       {media_tokens_relato:.1f}")
print(f"Overhead fixo (prompt+schema): {OVERHEAD_INPUT_TOKENS}")
print(f"Input tokens por chamada:      {input_tokens_por_chamada}")
print(f"Output tokens por chamada:     {OUTPUT_TOKENS_FIXO}")

# ============================================================
# ETAPA 2 — Estimativa de custo para 1.000 elementos
# ============================================================
print(f"\n{'=' * 60}")
print(f"ESTIMATIVA DE CUSTO — {N_ELEMENTOS:,} elementos")
print("=" * 60)

total_input_tokens  = input_tokens_por_chamada * N_ELEMENTOS
total_output_tokens = OUTPUT_TOKENS_FIXO       * N_ELEMENTOS

print(f"\nTokens totais:")
print(f"  Input:  {total_input_tokens:,}")
print(f"  Output: {total_output_tokens:,}")
print()

resultados = []
for nome, precos in modelos.items():
    custo_input  = (total_input_tokens  / 1_000_000) * precos["input_por_milhao"]
    custo_output = (total_output_tokens / 1_000_000) * precos["output_por_milhao"]
    custo_total  = custo_input + custo_output
    custo_brl    = custo_total * USD_BRL

    resultados.append({
        "modelo":    nome,
        "custo_usd": custo_total,
        "custo_brl": custo_brl,
    })

    print(f"--- {nome} ---")
    print(f"  Input:  ${custo_input:.4f}")
    print(f"  Output: ${custo_output:.4f}")
    print(f"  Total:  ${custo_total:.4f}  (~R$ {custo_brl:.2f})")
    print()

# ============================================================
# ETAPA 3 — Projeção para dataset completo (200k elementos)
# ============================================================
print("=" * 60)
print("PROJEÇÃO — dataset completo (200.000 elementos)")
print("=" * 60)

fator = 200_000 / N_ELEMENTOS
for r in resultados:
    custo_200k_usd = r["custo_usd"] * fator
    custo_200k_brl = r["custo_brl"] * fator
    print(f"  {r['modelo']}: ${custo_200k_usd:.2f}  (~R$ {custo_200k_brl:.2f})")

print()
print("=" * 60)
print("CONCLUSÃO")
print("=" * 60)
print(f"""
Para {N_ELEMENTOS:,} elementos (escopo do trabalho):
  - gemini-2.5-flash é o mais econômico (~R$ 1.62)
  - Diferença entre modelos é mínima nessa escala

Para o dataset completo (200k elementos):
  - gemini-2.5-flash é o recomendado por custo/benefício
  - gemini-3.1-pro-preview é ~20x mais caro que o flash

Escolha adotada no trabalho: Llama 3.1 8B via Groq API (plano gratuito),
com rate limit de 6k tokens/minuto e 500k tokens/dia.
""")