"""第1回: 素のLLMに会話履歴を保って質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 最初に講座に固有の質問をする
    uv run python -m app.ask --experiment --question "LoRAとは？" --question "QLoRAとは？"

ツールは持たないが、同じチャット内の会話履歴を保つ「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import csv
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
DEFAULT_EXPERIMENT_MODELS = ["gemini-3.5-flash-lite", "gemini-3.5-flash"]
DEFAULT_EXPERIMENT_QUESTIONS = [
    DEFAULT_QUESTION,
    "LoRAとQLoRAの違いは何ですか？",
]
EXPERIMENT_FIELDS = [
    "run_id",
    "started_at",
    "model",
    "question",
    "trial",
    "status",
    "elapsed_seconds",
    "input_tokens",
    "output_tokens",
    "error_code",
    "error_message",
]


def ask(question: str, chat=None) -> str:
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
    response = chat.send_message(question)

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[usage] input={usage.prompt_token_count} "
            f"output={usage.candidates_token_count}"
        )

    return response.text


def _redact_sensitive_text(message: str, api_key: str) -> str:
    message = str(message or "").replace(api_key, "[REDACTED]") if api_key else str(message or "")
    return re.sub(r"AIza[0-9A-Za-z_-]{20,}", "[REDACTED]", message)[:500]


def _run_trial(model: str, question: str, api_key: str) -> dict:
    started = None
    try:
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model=model)
        started = time.perf_counter()
        response = chat.send_message(question)
        elapsed = time.perf_counter() - started
        usage = response.usage_metadata
        return {
            "status": "success",
            "elapsed_seconds": elapsed,
            "input_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": getattr(usage, "candidates_token_count", None),
            "error_code": "",
            "error_message": "",
        }
    except Exception as error:
        elapsed = time.perf_counter() - started if started is not None else None
        code = getattr(error, "code", "")
        return {
            "status": "error",
            "elapsed_seconds": elapsed,
            "input_tokens": None,
            "output_tokens": None,
            "error_code": str(code),
            "error_message": _redact_sensitive_text(
                getattr(error, "message", None) or str(error), api_key
            ),
        }


def _write_experiment_results(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with output_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _print_experiment_summary(rows: list[dict], models: list[str], questions: list[str]) -> None:
    print("\n--- 実験結果（成功試行のみ）---")
    for model in models:
        for question in questions:
            group = [
                row for row in rows
                if row["model"] == model and row["question"] == question
            ]
            successful = [row for row in group if row["status"] == "success"]
            failed_count = len(group) - len(successful)
            label = f"[model={model}] [質問={question}]"
            if not successful:
                print(f"{label}: 成功試行なし（失敗 {failed_count}）")
                continue
            elapsed = fmean(row["elapsed_seconds"] for row in successful)
            input_tokens = [row["input_tokens"] for row in successful if row["input_tokens"] is not None]
            output_tokens = [row["output_tokens"] for row in successful if row["output_tokens"] is not None]
            input_mean = fmean(input_tokens) if input_tokens else "N/A"
            output_mean = fmean(output_tokens) if output_tokens else "N/A"
            print(
                f"{label}: 成功 {len(successful)}, 失敗 {failed_count}, "
                f"平均 {elapsed:.2f} 秒, 入力 {input_mean} tokens, 出力 {output_mean} tokens"
            )


def run_experiment(models: list[str], questions: list[str], repeat: int, output_path: Path) -> None:
    run_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    api_key = get_api_key()
    recorded_questions = [
        _redact_sensitive_text(question, api_key) for question in questions
    ]
    rows = []
    stop_after_rate_limit = False

    for model in models:
        for question in questions:
            for trial in range(1, repeat + 1):
                result = _run_trial(model, question, api_key)
                row = {
                    "run_id": run_id,
                    "started_at": started_at,
                    "model": model,
                    "question": _redact_sensitive_text(question, api_key),
                    "trial": trial,
                    **result,
                }
                rows.append(row)
                if result["status"] == "success":
                    print(
                        f"[success] {model} trial={trial}/{repeat} "
                        f"{result['elapsed_seconds']:.2f}s "
                        f"input={result['input_tokens']} output={result['output_tokens']}"
                    )
                else:
                    print(
                        f"[error] {model} trial={trial}/{repeat} "
                        f"code={result['error_code']} {result['error_message']}"
                    )
                    if result["error_code"] == "429":
                        print("レート制限のため、残りの試行を中止します。")
                        stop_after_rate_limit = True
                        break
            if stop_after_rate_limit:
                break
        if stop_after_rate_limit:
            break

    _write_experiment_results(output_path, rows)
    _print_experiment_summary(rows, models, recorded_questions)
    print(f"結果を保存しました: {output_path} (run_id={run_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="会話履歴を保って素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--experiment", action="store_true", help="モデル・質問ごとの比較実験を実行する")
    parser.add_argument("--model", action="append", help="実験対象モデル（複数回指定可）")
    parser.add_argument("--question", dest="experiment_questions", action="append", help="実験用の質問（複数回指定可）")
    parser.add_argument("--repeat", type=int, default=3, help="質問ごとの試行回数（既定: 3）")
    parser.add_argument("--output", type=Path, default=Path("data/experiment-results.csv"), help="結果CSVの出力先")
    args = parser.parse_args()

    if args.experiment:
        if args.repeat < 1:
            parser.error("--repeat は1以上を指定してください")
        models = args.model or DEFAULT_EXPERIMENT_MODELS
        questions = args.experiment_questions or DEFAULT_EXPERIMENT_QUESTIONS
        run_experiment(models, questions, args.repeat, args.output)
        return

    print(f"[model] {MODEL}")
    print(f"[質問] {args.question}\n")
    try:
        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        print(ask(args.question, chat))
        while True:
            try:
                question = input("\n[続けて質問] (空行で終了): ")
            except EOFError:
                break
            if not question.strip():
                break
            print(f"\n[質問] {question}\n")
            print(ask(question, chat))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
