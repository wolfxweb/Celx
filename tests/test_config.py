from legacy_doc.config import load_config


def test_default_config_has_required_sections() -> None:
    config = load_config("configs/default.yaml")

    assert {"project", "data", "model", "training", "generation"} <= config.keys()
    assert config["data"]["codexglue_languages"] == ["python", "php", "javascript"]
    assert config["data"]["target_languages"] == ["python", "php", "javascript", "sql"]
    assert config["model"]["max_length"] > 0


def test_qlora_configuration_is_coherent() -> None:
    config = load_config("configs/default.yaml")

    assert config["model"]["load_in_4bit"] is True
    assert config["training"]["lora_r"] > 0
    assert config["training"]["gradient_accumulation_steps"] >= 1
