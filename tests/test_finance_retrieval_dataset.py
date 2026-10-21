"""Keep relevance labels anchored to the actual local handbook."""

import hashlib
import json
from pathlib import Path

from finassist.agents.finance_qa.config import FinanceAgentSettings
from finassist.agents.finance_qa.corpus_index import load_finance_corpus

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "evals/finance/retrieval"


def test_relevance_labels_resolve_to_frozen_handbook_passages():
    manifest = json.loads((DATASET / "corpus.json").read_text())
    pdf = ROOT / manifest["source"]
    assert hashlib.sha256(pdf.read_bytes()).hexdigest() == manifest["sha256"]
    settings = FinanceAgentSettings(
        chunk_size=manifest["chunk_size"], chunk_overlap=manifest["chunk_overlap"]
    )
    chunks = {
        d.metadata["chunk_id"]: d for d in load_finance_corpus(pdf_path=pdf, settings=settings)
    }
    assert len(chunks) == manifest["chunk_count"]
    rows = [json.loads(line) for line in (DATASET / "cases.jsonl").read_text().splitlines()]
    assert rows
    assert len(rows) == manifest["case_count"]
    assert len({r["id"] for r in rows}) == len(rows)
    assert len({r["query"] for r in rows}) == len(rows)
    assert any(not r["answerable"] for r in rows)
    for row in rows:
        assert len({g["concept"] for g in row["relevance_groups"]}) == len(row["relevance_groups"])
        assert row["query"].strip() and row["tags"]
        assert isinstance(row["critical"], bool)
        assert isinstance(row["answerable"], bool)
        assert row["dense_queries"] and row["sparse_keywords"]
        assert all(q.strip() for q in row["dense_queries"] + row["sparse_keywords"])
        assert bool(row["relevance_groups"]) == row["answerable"]
        for group in row["relevance_groups"]:
            assert group["concept"] and group["rationale"] and group["passages"]
            for passage in group["passages"]:
                document = chunks[passage["chunk_id"]]
                assert document.metadata["page_number"] == passage["page_number"]
                assert (
                    hashlib.sha256(document.page_content.encode()).hexdigest()
                    == passage["content_sha256"]
                )


def test_coverage_map_tracks_every_case_and_required_scenario():
    rows = [json.loads(line) for line in (DATASET / "cases.jsonl").read_text().splitlines()]
    coverage = json.loads((DATASET / "coverage.json").read_text())
    mapped = [case_id for ids in coverage["topics"].values() for case_id in ids]
    assert len(mapped) == len(set(mapped))
    assert set(mapped) == {r["id"] for r in rows}
    for row in rows:
        assert row["id"] in coverage["topics"][row["tags"][0]]
    tags = {tag for row in rows for tag in row["tags"]}
    assert set(coverage["scenario_tags"]) <= tags
    # Keep hard negatives and genuine cross-passage questions in the set.
    assert sum(not r["answerable"] for r in rows) >= 3
    assert any(len(r["relevance_groups"]) > 1 for r in rows)
