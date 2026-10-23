"""Before/after eval for adding collections, with the deployed retriever.

Retrieval: every question with a "relevant" list is searched with the app's
pgvector_bm25 retriever (collection names and generic verbs dropped from
the BM25 query, 0.30 cutoff). Routed queries and their embeddings are cached
in eval/retrieval_inputs.json (shared with eval_retrieval.py), so a run
before ingesting and one after search exactly the same inputs. Sources are
scored two ways:
  strict  against "relevant" only (the patterns written before the new
          collections existed): a drop shows older sources being displaced
  full    against "relevant" + "relevant_new" (the new collections' pages
          that answer the question): the bar, no question may get worse

Answers (--answers): every question also goes through the real pipeline
(prepare_answer, then the chat model; nothing is written to the database).
Each answer is checked for the scripture references it should cite ("cites"
in questions.json), and, when Institute manual passages were used, for
citing at least one reference those passages discuss.

Usage:
    uv run python scripts/eval_collections.py eval/phase_d_before.json
    uv run python scripts/eval_collections.py eval/phase_d_after.json --answers --compare eval/phase_d_before.json

Costs a few cents: a routing call and an embedding for new questions, and
with --answers one or two chat calls per question.
"""

import argparse
import asyncio
import fnmatch
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from langchain_openai import ChatOpenAI  # noqa: E402

import src.backend.main as backend  # noqa: E402
from src.backend.database import get_session_factory  # noqa: E402
from src.backend.main import (  # noqa: E402
    CHAT_MODEL,
    CHUNKS_PER_RETRIEVER,
    PGVECTOR_BM25_MIN_SIMILARITY,
    Message,
    describe_source,
    keyword_drop_phrases,
)
from src.backend.retrieval.keyword import GENERIC_VERBS  # noqa: E402
from src.backend.retrieval.pgvector import PgvectorBm25Retriever, SqlTransport  # noqa: E402
from src.backend.source_links import source_link_for  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from eval_retrieval import CachedEmbeddings, prepare_inputs  # noqa: E402

QUESTIONS = PROJECT_ROOT / "tests" / "eval" / "questions.json"
# A scripture reference in an answer: "Alma 32:21", "D&C 121", "1 Nephi 4:10-18".
REFERENCE = re.compile(r"\b((?:[1-4] )?[A-Z][a-z]+(?: of [A-Z][a-z]+)?|D&C) (\d+)(?::\d+)?")


def score(sources: list[str], patterns: list[str]) -> tuple[int, int]:
    relevant = sum(any(fnmatch.fnmatch(s, p) for p in patterns) for s in sources)
    return relevant, len(sources) - relevant


def chapters(text: str) -> set[str]:
    """{"Alma 32", "D&C 121"} for every reference in the text."""
    text = text.replace("Doctrine and Covenants", "D&C")
    return {f"{book} {chapter}" for book, chapter in REFERENCE.findall(text)}


class Capturing:
    """The retriever, remembering the passages of the last search."""

    def __init__(self, inner):
        self.inner = inner
        self.last = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def ainvoke(self, query):
        self.last = await self.inner.ainvoke(query)
        return self.last


async def answer(item: dict, retriever: Capturing, chat) -> dict:
    backend._INTENT_CACHE.clear()
    retriever.last = []
    history = [Message(**m) for m in item.get("history", [])]
    prepared = await backend.prepare_answer(item["question"], history, retriever, chat)
    if prepared.fixed_answer is not None:
        text = prepared.fixed_answer
    else:
        text = (await chat.ainvoke(prepared.llm_messages)).content
    manual_refs = set()
    for doc in retriever.last if prepared.sources else []:
        if doc.metadata.get("source", "").startswith("institute_"):
            first = doc.page_content.split("\n\n", 2)
            manual_refs |= chapters(" ".join(first[:2]))  # the section line and its "Scriptures:" line
    cited = chapters(text)
    result = {
        "intent": prepared.intent,
        "answer": text,
        "labels": [link.label for link in prepared.links or []],
        "cited_chapters": sorted(cited),
        "manual_chapters": sorted(manual_refs),
        "cites_manual_ref": bool(cited & manual_refs) if manual_refs else None,
    }
    if item.get("cites"):
        result["cites_expected"] = any(c in text.replace("Doctrine and Covenants", "D&C") for c in item["cites"])
    return result


def compare(before: dict, after: dict, items: list[dict]) -> list[dict]:
    rows = []
    for item in items:
        qid = item["id"]
        if qid not in after["retrieval"]:
            continue
        a = after["retrieval"][qid]
        b = before["retrieval"].get(qid)
        row = {"id": qid, "after_full": a["full"], "after_strict": a["strict"]}
        if b:
            row["before_full"], row["before_strict"] = b["full"], b["strict"]
            row["worse"] = a["full"][0] < b["full"][0] or a["full"][1] > b["full"][1]
            row["displaced"] = b["strict"][0] - a["strict"][0]
        rows.append(row)
    return rows


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--answers", action="store_true", help="also generate and check full answers")
    parser.add_argument("--compare", type=Path, help="an earlier run to compare retrieval with")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    searched = [i for i in items if i.get("relevant")]
    inputs = await prepare_inputs(searched, refresh=False)
    embeddings = CachedEmbeddings({v["query"]: v["embedding"] for v in inputs.values()})
    retriever = PgvectorBm25Retriever(
        SqlTransport(get_session_factory()), embeddings, k=CHUNKS_PER_RETRIEVER,
        min_similarity=PGVECTOR_BM25_MIN_SIMILARITY, drop_phrases=keyword_drop_phrases(), drop_words=GENERIC_VERBS,
    )
    print(f"passages: {retriever.chunk_counts()['vector']:,}")

    retrieval = {}
    for item in searched:
        docs = await retriever.ainvoke(inputs[item["id"]]["query"])
        sources = list(dict.fromkeys(describe_source(d) for d in docs))
        labels = {}
        for d in docs:
            labels.setdefault(describe_source(d), source_link_for(describe_source(d), d.metadata.get("title"),
                                                                   d.metadata.get("url")).label)
        retrieval[item["id"]] = {
            "query": inputs[item["id"]]["query"],
            "sources": sources,
            "labels": [labels[s] for s in sources],
            "strict": score(sources, item["relevant"]),
            "full": score(sources, item["relevant"] + item.get("relevant_new", [])),
        }
    result = {"passages": retriever.chunk_counts()["vector"], "retrieval": retrieval}

    if args.answers:
        chat = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        live = Capturing(retriever)
        result["answers"] = {}
        for item in items:
            result["answers"][item["id"]] = await answer(item, live, chat)
            a = result["answers"][item["id"]]
            print(f"  answered {item['id']:<36} cites expected: {a.get('cites_expected', '-')!s:<5} "
                  f"cites a manual ref: {a['cites_manual_ref']!s}", flush=True)

    if args.compare:
        before = json.loads(args.compare.read_text(encoding="utf-8"))
        rows = compare(before, result, items)
        result["comparison"] = rows
        print(f"\n{'question':<36}{'before full':>12}{'after full':>12}{'strict b→a':>12}  verdict")
        for r in rows:
            if "before_full" in r:
                verdict = "WORSE" if r["worse"] else ("better" if r["after_full"][0] > r["before_full"][0]
                                                       or r["after_full"][1] < r["before_full"][1] else "same")
                print(f"{r['id']:<36}{'%d rel/%d off' % tuple(r['before_full']):>12}"
                      f"{'%d rel/%d off' % tuple(r['after_full']):>12}"
                      f"{'%d→%d' % (r['before_strict'][0], r['after_strict'][0]):>12}  {verdict}")
        worse = [r["id"] for r in rows if r.get("worse")]
        print(f"\nworse: {len(worse)} ({', '.join(worse) or '-'})")

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
