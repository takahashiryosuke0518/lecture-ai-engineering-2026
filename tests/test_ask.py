"""素のLLM呼び出しのテスト（LLMもAPIキーも使わない）。"""

from types import SimpleNamespace

from app import ask


def test_ask_prints_token_usage(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
    )
    client = SimpleNamespace(
        models=SimpleNamespace(generate_content=lambda **kwargs: response)
    )
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question") == "answer"
    assert "[usage] input=12 output=7" in capsys.readouterr().out