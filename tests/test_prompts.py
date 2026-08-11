from legacy_doc.prompts import (
    SYSTEM_PROMPT,
    required_sections,
    system_prompt,
    training_messages,
    user_prompt,
)


REQUIRED_SECTIONS_PT = required_sections("pt-BR")
REQUIRED_SECTIONS_EN = required_sections("en")


def test_user_prompt_contains_language_code_and_sections() -> None:
    prompt = user_prompt("python", "def soma(a, b): return a + b")

    assert "Linguagem: python" in prompt
    assert "def soma" in prompt
    assert "## Método ou função" in prompt
    assert "Idioma da documentação: pt-BR" in prompt
    for section in REQUIRED_SECTIONS_PT:
        assert section in prompt


def test_user_prompt_english() -> None:
    prompt = user_prompt("php", "function x() {}", doc_language="en")
    assert "Documentation language: en" in prompt
    assert "## Method or function" in prompt
    for section in REQUIRED_SECTIONS_EN:
        assert section in prompt


def test_training_messages_form_valid_conversation() -> None:
    messages = training_messages("php", "function dobro($n) { return $n * 2; }", "Dobra n.")

    assert [message["role"] for message in messages] == ["system", "user", "assistant"]
    assert messages[0]["content"] == system_prompt("pt-BR")
    assert "## Método ou função" in messages[-1]["content"]
    for section in REQUIRED_SECTIONS_PT:
        assert section in messages[-1]["content"]


def test_training_messages_english_uses_en_sections() -> None:
    messages = training_messages(
        "php",
        "function dobro($n) { return $n * 2; }",
        "Doubles n.",
        doc_language="en",
    )
    assert "Reply in English" in messages[0]["content"]
    assert "## Method or function" in messages[-1]["content"]
    assert "### Objective" in messages[-1]["content"]
    assert "Doubles n." in messages[-1]["content"]


def test_system_prompt_forbids_inventing_rules() -> None:
    assert "Não invente regras de negócio" in SYSTEM_PROMPT
    assert "Do not invent business rules" in system_prompt("en")


def test_sql_uses_specific_artifact_heading() -> None:
    prompt = user_prompt("sql", "SELECT id FROM pedidos")
    messages = training_messages("sql", "SELECT id FROM pedidos", "Lista pedidos.")

    assert "## Consulta SQL" in prompt
    assert "## Consulta SQL" in messages[-1]["content"]
    assert "## SQL query" in user_prompt("sql", "SELECT 1", doc_language="en")
