"""素のLLM呼び出しのテスト（LLMもAPIキーも使わない）。"""

import csv
from types import SimpleNamespace

from google.genai import errors

from app import ask


def test_ask_prints_token_usage(monkeypatch, capsys):
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
    )
    chat = SimpleNamespace(send_message=lambda question: response)
    client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")

    assert ask.ask("question") == "answer"
    assert "[usage] input=12 output=7" in capsys.readouterr().out


def test_main_keeps_chat_history_for_follow_up_questions(monkeypatch):
    sent_questions = []
    response = SimpleNamespace(
        text="answer",
        usage_metadata=SimpleNamespace(prompt_token_count=12, candidates_token_count=7),
    )
    chat = SimpleNamespace(
        send_message=lambda question: (sent_questions.append(question), response)[1]
    )
    client = SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))
    questions = iter(["follow-up", ""])
    monkeypatch.setattr(ask.genai, "Client", lambda **kwargs: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("sys.argv", ["app.ask", "first question"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask.main()

    assert sent_questions == ["first question", "follow-up"]


def test_experiment_appends_distinct_runs_and_averages_successes_only(
    monkeypatch, tmp_path, capsys
):
    api_key = "test-api-key-secret"
    client_count = 0

    def create_client(api_key):
        nonlocal client_count
        client_count += 1
        call_number = client_count

        def send_message(question):
            if call_number == 2:
                raise errors.APIError(
                    503,
                    {"error": {"message": f"service unavailable {api_key}"}},
                )
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12, candidates_token_count=4
                )
            )

        chat = SimpleNamespace(send_message=send_message)
        return SimpleNamespace(chats=SimpleNamespace(create=lambda **kwargs: chat))

    clock_values = iter(
        value
        for trial in range(24)
        for value in (0.0, 100.0 if trial == 1 else 2.0)
    )
    monkeypatch.setattr(ask.genai, "Client", create_client)
    monkeypatch.setattr(ask, "get_api_key", lambda: api_key)
    monkeypatch.setattr(ask.time, "perf_counter", lambda: next(clock_values))
    output_path = tmp_path / "results.csv"
    models = ["model-a", "model-b"]
    questions = [f"question-{api_key}", "question-b"]

    ask.run_experiment(models, questions, 3, output_path)
    first_output = capsys.readouterr().out
    ask.run_experiment(models, questions, 3, output_path)
    second_output = capsys.readouterr().out

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    run_ids = {row["run_id"] for row in rows}

    assert len(rows) == 24
    assert len(run_ids) == 2
    assert client_count == 24
    assert sum(row["status"] == "error" for row in rows) == 1
    assert rows[0]["question"] == "question-[REDACTED]"
    assert "平均 2.00 秒" in first_output
    assert "平均 2.00 秒" in second_output
    assert api_key not in output_path.read_text(encoding="utf-8")
    assert api_key not in first_output + second_output


def test_experiment_stops_after_rate_limit_and_records_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr(
        ask,
        "_run_trial",
        lambda *args: {
            "status": "error",
            "elapsed_seconds": 0.5,
            "input_tokens": None,
            "output_tokens": None,
            "error_code": "429",
            "error_message": "quota exceeded",
        },
    )
    output_path = tmp_path / "results.csv"

    ask.run_experiment(["model-a", "model-b"], ["question"], 3, output_path)

    with output_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["status"] == "error"
    assert rows[0]["error_code"] == "429"
    assert rows[0]["error_message"] == "quota exceeded"