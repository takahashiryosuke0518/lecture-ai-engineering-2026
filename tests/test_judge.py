"""第4回: 採点（app/judge.py）の配線テスト。API キーは使わない。

--model test と同じ偽モデル（規則で判定する FunctionModel）で、評価セット 10 件の
「採点 → 人のラベルとの比較 → 要確認・不一致の集計」が通ることを確かめる。
"""

from app import judge
from app.judge import (
    ALL_CRITERIA,
    DEFAULT_CASES,
    DEFAULT_RUBRIC,
    build_judge,
    build_test_model,
    check_lecture_mark,
    judge_cases,
    load_labels,
    overall,
    read_jsonl,
    summarize,
)

INSTRUCTOR_LABELS = judge.ROOT / "lectures" / "04" / "reference" / "labels_instructor.jsonl"


def test_generate_answers_saves_each_answer_and_skips_completed_cases(tmp_path, monkeypatch):
    import app.agent as app_agent

    cases = [
        {"id": "1", "question": "answered", "answer": "既存回答", "reference": {"expected": "保持"}},
        {"id": "2", "question": "generate then fail", "answer": "", "reference": {"expected": "保持"}},
        {"id": "3", "question": "retry later", "answer": "", "reference": {"expected": "保持"}},
    ]
    output = tmp_path / "cases.jsonl"
    calls = []
    waits = []

    class FakeAgent:
        def __init__(self, responses):
            self.responses = iter(responses)

        def run_sync(self, question, model):
            calls.append(question)
            response = next(self.responses)
            if isinstance(response, Exception):
                raise response
            return type("Result", (), {"output": response})()

    fake_agent = FakeAgent(["生成済み回答", RuntimeError("API error")])
    monkeypatch.setattr(app_agent, "agent", fake_agent)
    monkeypatch.setattr(app_agent, "build_model", lambda: object())
    monkeypatch.setattr(judge.time, "sleep", waits.append)

    try:
        judge.generate_answers(cases, wait=2.5, save_path=output)
    except RuntimeError as error:
        assert str(error) == "API error"
    else:
        raise AssertionError("API error should propagate")

    saved = read_jsonl(output)
    assert [case["answer"] for case in saved] == ["既存回答", "生成済み回答", ""]
    assert [case["reference"] for case in saved] == [{"expected": "保持"}] * 3
    assert calls == ["generate then fail", "retry later"]
    assert waits == [2.5]

    calls.clear()
    fake_agent.responses = iter(["再開回答"])
    judge.generate_answers(saved, wait=2.5, save_path=output)

    assert calls == ["retry later"]
    assert [case["answer"] for case in read_jsonl(output)] == ["既存回答", "生成済み回答", "再開回答"]


def test_lecture_mark_is_checked_by_code():
    assert check_lecture_mark("LoRA は…\n\n関係する回：第2回") == "pass"
    assert check_lecture_mark("第2回で扱う LoRA は…\n\n以上です。") == "fail"  # 末尾に無ければ fail
    assert check_lecture_mark("") == "fail"


def test_overall_requires_all_required_criteria():
    base = {"grounding": "pass", "scope": "pass", "hint": "pass", "lecture_mark": "pass", "clarity": "fail"}
    assert overall(base) == "pass"  # 分かりやすさは総合に入れない
    assert overall({**base, "hint": "fail"}) == "fail"
    assert overall({**base, "grounding": "unsure"}) == "unsure"


def test_cases_and_rubric_are_consistent():
    cases = read_jsonl(DEFAULT_CASES)
    assert len(cases) == 10
    assert len({c["id"] for c in cases}) == 10
    for c in cases:
        assert c["question"] and c["answer"] and c["reference"]["expected"]
    rubric = DEFAULT_RUBRIC.read_text(encoding="utf-8")
    for key, name in ALL_CRITERIA[:-1]:  # 回の表示はコードで判定するので rubric には書かない
        assert f"{name}（{key}）" in rubric
    labels = load_labels(INSTRUCTOR_LABELS)
    assert set(labels) == {c["id"] for c in cases}


def test_judge_pipeline_with_fake_model():
    cases = read_jsonl(DEFAULT_CASES)
    labels = load_labels(INSTRUCTOR_LABELS)
    agent = build_judge(build_test_model(), DEFAULT_RUBRIC.read_text(encoding="utf-8"))

    verdicts = judge_cases(agent, cases)
    assert set(verdicts) == {c["id"] for c in cases}
    for v in verdicts.values():
        for key, _ in ALL_CRITERIA:
            assert v[key]["verdict"] in ("pass", "fail", "unsure")
            assert v[key]["reason"]

    summary = summarize(cases, verdicts, labels)
    assert summary["n"] == 10
    # unsure にした項目は一致率の分母から外れ、要確認に入る
    excluded = sum(p["total"] - p["compared"] for p in summary["per_criterion"].values())
    assert excluded == len(summary["review"])
    assert summary["coverage"]["covered"] + len({r["id"] for r in summary["review"]}) == 10
    for m in summary["mismatches"]:
        assert m["kind"] in ("見逃し", "過検出")
