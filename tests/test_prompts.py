from legacy_doc.prompts import SYSTEM_PROMPT, training_messages, user_prompt


REQUIRED_SECTIONS = (
    "### Objetivo",
    "### Parâmetros",
    "### Retorno",
    "### Funcionamento",
    "### Regras de negócio identificadas",
    "### Pontos não determinados",
)


def test_user_prompt_contains_language_code_and_sections() -> None:
    prompt = user_prompt("python", "def soma(a, b): return a + b")

    assert "Linguagem: python" in prompt
    assert "def soma" in prompt
    assert "## Método ou função" in prompt
    for section in REQUIRED_SECTIONS:
        assert section in prompt


def test_training_messages_form_valid_conversation() -> None:
    messages = training_messages("php", "function dobro($n) { return $n * 2; }", "Dobra n.")

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "## Método ou função" in messages[-1]["content"]
    for section in REQUIRED_SECTIONS:
        assert section in messages[-1]["content"]


def test_system_prompt_forbids_inventing_rules() -> None:
    assert "Não invente regras de negócio" in SYSTEM_PROMPT


def test_sql_uses_specific_artifact_heading() -> None:
    prompt = user_prompt("sql", "SELECT id FROM pedidos")
    messages = training_messages("sql", "SELECT id FROM pedidos", "Lista pedidos.")

    assert "## Consulta SQL" in prompt
    assert "## Consulta SQL" in messages[-1]["content"]
