from __future__ import annotations

SYSTEM_PROMPT = """Você é especialista em análise e documentação de software legado.
Documente somente comportamentos sustentados pelo código fornecido.
Não invente regras de negócio. Quando não houver evidência suficiente, registre o ponto em
'Pontos não determinados'. Responda em português e use Markdown."""

OUTPUT_TEMPLATE = """## {artifact_heading}

{summary}

### Objetivo

{objective}

### Parâmetros

{parameters}

### Retorno

{returns}

### Funcionamento

{flow}

### Regras de negócio identificadas

{rules}

### Pontos não determinados

{unknowns}"""


def user_prompt(language: str, code: str) -> str:
    artifact_heading = "Consulta SQL" if language.lower() == "sql" else "Método ou função"
    return f"""Analise a função abaixo e produza documentação técnica estruturada.

Linguagem: {language}

```{language}
{code.strip()}
```

Use obrigatoriamente estas seções:
- ## {artifact_heading}
- ### Objetivo
- ### Parâmetros
- ### Retorno
- ### Funcionamento
- ### Regras de negócio identificadas
- ### Pontos não determinados
"""


def training_messages(language: str, code: str, reference: str) -> list[dict[str, str]]:
    """Monta uma conversa SFT; a referência original não é tratada como regra inferida."""
    answer = OUTPUT_TEMPLATE.format(
        artifact_heading="Consulta SQL" if language.lower() == "sql" else "Método ou função",
        summary=reference.strip(),
        objective=reference.strip(),
        parameters="Não descritos na referência original.",
        returns="Não descrito na referência original.",
        flow="- O comportamento detalhado deve ser confirmado pela análise do código.",
        rules="- Nenhuma regra de negócio explícita foi fornecida pela docstring.",
        unknowns="- Parâmetros, retorno, validações e exceções exigem análise adicional.",
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(language, code)},
        {"role": "assistant", "content": answer},
    ]
