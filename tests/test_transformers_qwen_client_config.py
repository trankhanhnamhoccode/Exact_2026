from xai_physics.llm.transformers_qwen_client import TransformersQwenSchemaClient, qwen_client_from_env


def test_qwen_client_from_env(monkeypatch):
    monkeypatch.setenv("XAI_QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    monkeypatch.setenv("XAI_QWEN_MAX_NEW_TOKENS", "512")
    monkeypatch.setenv("XAI_QWEN_TEMPERATURE", "0.2")
    monkeypatch.setenv("XAI_QWEN_TOP_P", "0.8")
    monkeypatch.setenv("XAI_QWEN_LOAD_IN_4BIT", "0")
    monkeypatch.setenv("XAI_QWEN_TRUST_REMOTE_CODE", "0")
    monkeypatch.setenv("XAI_QWEN_DEVICE_MAP", "cuda:0")

    client = qwen_client_from_env()

    assert client.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert client.max_new_tokens == 512
    assert client.temperature == 0.2
    assert client.top_p == 0.8
    assert client.load_in_4bit is False
    assert client.trust_remote_code is False
    assert client.device_map == "cuda:0"


def test_qwen_client_is_lazy_loaded():
    client = TransformersQwenSchemaClient()

    assert client._tokenizer is None
    assert client._model is None
    assert client.load_in_4bit is False
