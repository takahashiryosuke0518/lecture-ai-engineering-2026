"""実SQLiteと固定ベクトルで、索引作成から検索までを確認する。APIは呼ばない。"""

import json
from unittest.mock import Mock

import pytest

from app import retrieval
from scripts.build_rag_index import build_index


@pytest.fixture
def materials(tmp_path):
    source = tmp_path / "lectures.json"
    source.write_text(json.dumps([
        {"id": f"lec{n:02d}-main", "title": f"講義{n}", "content": [
            {"section": "章", "content": [
                {"id": f"lec{n:02d}-main-p001", "title": f"ページ{n}",
                 "page": 1, "content": f"本文{n}"}]}]}
        for n in (1, 2, 5)
    ], ensure_ascii=False), encoding="utf-8")
    return source, tmp_path / "index.sqlite3"


@pytest.fixture
def multi_page_materials(tmp_path):
    source = tmp_path / "lectures.json"
    documents = [
        {
            "id": "lec01-main",
            "title": "講義1",
            "content": [{
                "section": "章1",
                "content": [
                    {
                        "id": f"lec01-main-p{page:03d}",
                        "title": f"ページ{page}",
                        "page": page,
                        "content": f"本文 lec01-main-p{page:03d}",
                    }
                    for page in range(1, 6)
                ],
            }],
        },
        {
            "id": "lec02-main",
            "title": "講義2",
            "content": [{
                "section": "章2",
                "content": [{
                    "id": "lec02-main-p003",
                    "title": "別資料のページ3",
                    "page": 3,
                    "content": "本文 lec02-main-p003",
                }],
            }],
        },
    ]
    source.write_text(json.dumps(documents, ensure_ascii=False), encoding="utf-8")
    return source, tmp_path / "index.sqlite3"


def test_built_index_can_be_searched(materials, monkeypatch):
    source, database = materials
    texts = []

    def embed(text, task):
        assert task == "RETRIEVAL_DOCUMENT"
        texts.append(text)
        return [1, 0] if "本文1" in text else [0, 1]

    result = build_index(source, database, embedder=embed)
    assert result["pages"] == 2  # 第5回の資料は検索対象に含めない
    assert all("章" in text and "ページ" in text for text in texts)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])
    hits = retrieval.search("質問")
    assert [h["page_id"] for h in hits] == ["lec01-main-p001", "lec02-main-p001"]
    assert hits[0]["content"] == "本文1"
    assert hits[0]["distance"] == pytest.approx(0)


@pytest.mark.parametrize(
    ("nearest_pages", "expected_hits", "expected_contexts"),
    [
        (
            ("lec01-main-p002", "lec01-main-p004"),
            ["lec01-main-p002", "lec01-main-p004"],
            [
                [("lec01-main-p001", "previous"), ("lec01-main-p003", "next")],
                [("lec01-main-p005", "next")],
            ],
        ),
        (
            ("lec01-main-p002", "lec01-main-p003"),
            ["lec01-main-p002", "lec01-main-p003"],
            [
                [("lec01-main-p001", "previous")],
                [("lec01-main-p004", "next")],
            ],
        ),
        (
            ("lec01-main-p001", "lec01-main-p005"),
            ["lec01-main-p001", "lec01-main-p005"],
            [
                [("lec01-main-p002", "next")],
                [("lec01-main-p004", "previous")],
            ],
        ),
    ],
)
def test_search_adds_adjacent_pages_without_changing_hits(
    multi_page_materials, monkeypatch, nearest_pages, expected_hits, expected_contexts
):
    source, database = multi_page_materials

    def embed(text, task):
        assert task == "RETRIEVAL_DOCUMENT"
        page_id = next(
            page_id for page_id in [f"lec01-main-p{page:03d}" for page in range(1, 6)]
            + ["lec02-main-p003"]
            if f"本文 {page_id}" in text
        )
        if page_id == nearest_pages[0]:
            return [1, 0]
        if page_id == nearest_pages[1]:
            return [0.8, 0.6]
        return [0, 1]

    build_index(source, database, embedder=embed)
    monkeypatch.setattr(retrieval, "RAG_DB", database)
    monkeypatch.setattr(retrieval, "RAG_SOURCE", source)
    monkeypatch.setattr(retrieval, "RAG_TOP_K", 2)
    monkeypatch.setattr(retrieval, "embed_text", lambda *_: [1, 0])

    hits = retrieval.search("質問")

    assert [hit["page_id"] for hit in hits] == expected_hits
    assert [hit["distance"] for hit in hits] == pytest.approx([0, 0.2])
    assert [
        [(page["page_id"], page["relation"]) for page in hit["context_pages"]]
        for hit in hits
    ] == expected_contexts

    all_page_ids = [hit["page_id"] for hit in hits] + [
        page["page_id"] for hit in hits for page in hit["context_pages"]
    ]
    assert len(all_page_ids) == len(set(all_page_ids))
    assert all(
        page["document_id"] == "lec01-main"
        for hit in hits for page in hit["context_pages"]
    )
    if expected_hits[0] == "lec01-main-p002":
        previous = hits[0]["context_pages"][0]
        assert {
            key: previous[key]
            for key in ("lecture", "document_title", "section", "title", "page", "content")
        } == {
            "lecture": 1,
            "document_title": "講義1",
            "section": "章1",
            "title": "ページ1",
            "page": 1,
            "content": "本文 lec01-main-p001",
        }


def test_existing_index_survives_failed_rebuild(materials):
    source, database = materials
    build_index(source, database, embedder=lambda *_: [1, 0])
    before = database.read_bytes()
    embed = Mock(side_effect=RuntimeError("埋め込みに失敗"))
    with pytest.raises(FileExistsError):
        build_index(source, database, embedder=embed)
    embed.assert_not_called()
    with pytest.raises(RuntimeError, match="埋め込みに失敗"):
        build_index(source, database, rebuild=True, embedder=embed)
    assert database.read_bytes() == before
    assert not list(database.parent.glob(".rag-build-*"))
