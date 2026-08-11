from legacy_doc.thinking import extract_thinking, strip_thinking


def test_strip_thinking_removes_block() -> None:
    raw = "<think>analise o codigo</think>\n\n## Método ou função\n\nOk."
    assert strip_thinking(raw).startswith("## Método ou função")
    assert "analise" not in strip_thinking(raw)


def test_extract_thinking() -> None:
    raw = "<think>passo 1</think>\n### Objetivo\nX"
    assert extract_thinking(raw) == "passo 1"
