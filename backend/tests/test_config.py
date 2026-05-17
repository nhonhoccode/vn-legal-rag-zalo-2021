"""Smoke test cho config loading."""

from src.config import PROJECT_ROOT, Settings, settings


def test_project_root_exists():
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "backend").is_dir()
    assert (PROJECT_ROOT / "docs").is_dir()


def test_settings_loads_with_defaults():
    s = Settings()
    assert s.app_env in ("dev", "prod")
    assert s.llm_provider in ("openrouter", "gemini", "deepseek", "custom")
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.top_k_final > 0


def test_settings_secrets_hidden_in_repr():
    """SecretStr → repr KHÔNG hiện giá trị thật (security)."""
    s = Settings(
        gemini_api_key="AIzaSyXXXSecretXXX",
        openrouter_api_key="sk-or-v1-XXXSecretXXX",
        jwt_secret="real-secret-value-32-chars-min-padding",
    )
    repr_str = repr(s)
    # Secret values KHÔNG được hiện trong repr.
    assert "AIzaSyXXXSecretXXX" not in repr_str
    assert "sk-or-v1-XXXSecretXXX" not in repr_str
    assert "real-secret-value-32-chars" not in repr_str
    # Vẫn access được qua .get_secret_value().
    assert s.gemini_api_key.get_secret_value() == "AIzaSyXXXSecretXXX"


def test_settings_singleton_loaded():
    assert settings.embedding_model == "BAAI/bge-m3"


def test_csv_validator_for_api_keys():
    s = Settings(allowed_api_keys="key1, key2,key3")
    assert s.allowed_api_keys == ["key1", "key2", "key3"]


def test_csv_validator_for_cors():
    s = Settings(cors_origins="http://localhost:3000,https://example.com")
    assert s.cors_origins == ["http://localhost:3000", "https://example.com"]


def test_is_prod_property():
    s_dev = Settings(app_env="dev")
    s_prod = Settings(app_env="prod")
    assert s_dev.is_prod is False
    assert s_prod.is_prod is True
