"""Score retrieval backends against the relevance lists in tests/eval/questions.json.

Every question with a "relevant" list (page patterns matched against
"collection/slug", e.g. "*/tith*") is routed once with the real router and
embedded once; both are cached in eval/retrieval_inputs.json, so all
variants see identical inputs and re-runs cost nothing. For each variant and
question it counts relevant and off-topic sources, then compares with the
FAISS backend, the bar: a question is WORSE if it has fewer relevant or more
off-topic sources than FAISS.

Variants:
  faiss            today's backend (FAISS + BM25 files)
  pgvector         hybrid_search() as deployed (pgvector + Postgres full text)
  pg_ft:w,c        pgvector + Postgres full text, vector weight w, cutoff c
  pg_bm25:w,c      pgvector + in-memory BM25 over rag_chunks, vector weight w, cutoff c
  pg_bm25f:w,c     the same with collection names and generic verbs dropped from
                   the BM25 query (what the app deploys: PgvectorBm25Retriever with
                   keyword_drop_phrases() and GENERIC_VERBS)
where cutoff c drops keyword-only hits whose similarity to the question is
below c ("-" = no cutoff). Every fused passage goes to the model, so the
weights only change the order of sources, not which ones are used.

pg_bm25f is also compared with pg_bm25 at the same settings (the backend
before the filter): it may not be worse on any question there either.

The three regressions found in the first pgvector eval count as fixed when
the question is no worse than FAISS and none of the off-topic sources that
version returned (MUST_FIX) are back.

Usage:
    uv run python scripts/eval_retrieval.py                 # all variants, summary + table
    uv run python scripts/eval_retrieval.py --refresh       # re-route and re-embed the questions
"""

import argparse
import asyncio
import fnmatch
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

from langchain_core.embeddings import Embeddings  # noqa: E402
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # noqa: E402

from src.backend.main import (  # noqa: E402
    CHAT_MODEL,
    keyword_drop_phrases,
    CHUNKS_PER_RETRIEVER,
    EMBEDDING_MODEL,
    INDEX_DIR,
    Message,
    describe_source,
    route_question,
    trim_history,
)
from src.backend.retrieval import load_local_retriever  # noqa: E402
from src.backend.retrieval.keyword import GENERIC_VERBS, full_text_query  # noqa: E402
from src.backend.retrieval.pgvector import PgvectorBm25Retriever, RestTransport, _document, fuse_with_cutoff  # noqa: E402

QUESTIONS = PROJECT_ROOT / "tests" / "eval" / "questions.json"
INPUTS = PROJECT_ROOT / "eval" / "retrieval_inputs.json"
OUTPUT = PROJECT_ROOT / "eval" / "retrieval_comparison.json"
WEIGHTS = [0.5, 0.7]
CUTOFFS = [None, 0.30, 0.35, 0.40, 0.45]
MUST_FIX = {
    "followup-tithing": ["bible_dictionary/gospels", "topics_and_questions/family-finances"],
    "followup-sacrament-scriptures": ["topics_and_questions/sabbath-day*", "topics_and_questions/sacrifice*"],
    "doctrine-moroni": ["bible_dictionary/moses"],
}


class CachedEmbeddings(Embeddings):
    """Serves cached question embeddings; embeds (and caches) anything new."""

    def __init__(self, cache: dict[str, list[float]]):
        self.cache = cache
        self.real = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]

    def embed_query(self, text):
        if text not in self.cache:
            self.cache[text] = self.real.embed_query(text)
        return self.cache[text]

    async def aembed_query(self, text):
        if text not in self.cache:
            self.cache[text] = await self.real.aembed_query(text)
        return self.cache[text]


def sources_of(docs) -> list[str]:
    return list(dict.fromkeys(describe_source(d) for d in docs))


def score(sources: list[str], patterns: list[str]) -> tuple[int, int]:
    relevant = sum(any(fnmatch.fnmatch(s, p) for p in patterns) for s in sources)
    return relevant, len(sources) - relevant


async def prepare_inputs(items, refresh: bool) -> dict:
    cache = json.loads(INPUTS.read_text(encoding="utf-8")) if INPUTS.exists() and not refresh else {}
    chat = ChatOpenAI(model=CHAT_MODEL, temperature=0)
    embedder = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    for item in items:
        if item["id"] in cache:
            continue
        history = trim_history([Message(**m) for m in item.get("history", [])])
        decision = await route_question(item["question"], history, chat)
        cache[item["id"]] = {
            "intent": decision.intent,
            "query": decision.query,
            "embedding": await embedder.aembed_query(decision.query),
        }
    INPUTS.parent.mkdir(exist_ok=True)
    INPUTS.write_text(json.dumps(cache), encoding="utf-8")
    return cache


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-route and re-embed the questions")
    args = parser.parse_args()

    items = [i for i in json.loads(QUESTIONS.read_text(encoding="utf-8")) if i.get("relevant")]
    inputs = await prepare_inputs(items, args.refresh)
    embeddings = CachedEmbeddings({v["query"]: v["embedding"] for v in inputs.values()})

    faiss = load_local_retriever(INDEX_DIR, embeddings, k=CHUNKS_PER_RETRIEVER)
    transport = RestTransport(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    passages = transport.load_passages()
    bm25 = PgvectorBm25Retriever(transport, embeddings, k=CHUNKS_PER_RETRIEVER, passages=passages)
    bm25f = PgvectorBm25Retriever(transport, embeddings, k=CHUNKS_PER_RETRIEVER, passages=passages,
                                  drop_phrases=keyword_drop_phrases(), drop_words=GENERIC_VERBS)

    results: dict[str, dict[str, list[str]]] = {}
    for item in items:
        query = inputs[item["id"]]["query"]
        embedding = inputs[item["id"]]["embedding"]
        per = results[item["id"]] = {}
        per["faiss"] = sources_of(await faiss.ainvoke(query))

        hybrid_rows = await transport.search(embedding, full_text_query(query), CHUNKS_PER_RETRIEVER)
        per["pgvector"] = sources_of([_document(r) for r in hybrid_rows])
        full_text = [_document(r) for r in sorted(
            (r for r in hybrid_rows if r.get("full_text_rank")), key=lambda r: r["full_text_rank"])]

        semantic, keyword, similarity = await bm25.search(query)
        _, keyword_f, similarity_f = await bm25f.search(query)
        similarity |= similarity_f
        extra = [d.metadata["chunk_id"] for d in full_text if d.metadata["chunk_id"] not in similarity]
        if extra:
            similarity |= {r["chunk_id"]: r["similarity"] for r in await transport.match(embedding, 1, extra)}

        for w in WEIGHTS:
            for c in CUTOFFS:
                tag = f"{w},{'-' if c is None else c}"
                per[f"pg_ft:{tag}"] = sources_of(fuse_with_cutoff(semantic, full_text, similarity, (w, 1 - w), c))
                per[f"pg_bm25:{tag}"] = sources_of(fuse_with_cutoff(semantic, keyword, similarity, (w, 1 - w), c))
                per[f"pg_bm25f:{tag}"] = sources_of(fuse_with_cutoff(semantic, keyword_f, similarity, (w, 1 - w), c))
        print(f"  {item['id']:<30} searched {query!r}", flush=True)

    patterns = {i["id"]: i["relevant"] for i in items}
    variants = list(next(iter(results.values())))
    summary = []
    for variant in variants:
        worse, rel_total, off_total = [], 0, 0
        for qid, per in results.items():
            rel, off = score(per[variant], patterns[qid])
            base_rel, base_off = score(per["faiss"], patterns[qid])
            rel_total += rel
            off_total += off
            if rel < base_rel or off > base_off:
                worse.append(qid)
        fixed = [q for q, banned in MUST_FIX.items()
                 if q not in worse
                 and not any(fnmatch.fnmatch(s, b) for s in results[q][variant] for b in banned)]
        row = {"variant": variant, "worse": worse, "relevant": rel_total, "off_topic": off_total, "must_fix_ok": fixed}
        if variant.startswith("pg_bm25f:"):
            before = variant.replace("pg_bm25f:", "pg_bm25:")
            row["worse_than_unfiltered"] = [
                qid for qid, per in results.items()
                if score(per[variant], patterns[qid])[0] < score(per[before], patterns[qid])[0]
                or score(per[variant], patterns[qid])[1] > score(per[before], patterns[qid])[1]
            ]
            row["changed_vs_unfiltered"] = [qid for qid, per in results.items() if per[variant] != per[before]]
        summary.append(row)

    summary.sort(key=lambda r: (len(r["worse"]), -len(r["must_fix_ok"]), r["off_topic"], -r["relevant"]))
    print(f"\n{'variant':<20}{'worse':>6}{'relevant':>10}{'off-topic':>11}  regressions fixed")
    shown = ("faiss", "pgvector", "pg_bm25:0.5,0.3", "pg_bm25f:0.5,0.3")
    for row in summary[:12] + [r for r in summary if r["variant"] in shown]:
        print(f"{row['variant']:<20}{len(row['worse']):>6}{row['relevant']:>10}{row['off_topic']:>11}  "
              f"{len(row['must_fix_ok'])}/3  worse: {', '.join(row['worse']) or '-'}")

    for row in summary:
        if row["variant"] == "pg_bm25f:0.5,0.3":
            print(f"\npg_bm25f:0.5,0.3 vs pg_bm25:0.5,0.3 (live before the filter): "
                  f"worse on {len(row['worse_than_unfiltered'])} ({', '.join(row['worse_than_unfiltered']) or '-'}); "
                  f"sources changed on {len(row['changed_vs_unfiltered'])} ({', '.join(row['changed_vs_unfiltered']) or '-'})")
    OUTPUT.write_text(json.dumps({"inputs": {k: {"intent": v["intent"], "query": v["query"]} for k, v in inputs.items()},
                                  "summary": summary, "results": results}, indent=1), encoding="utf-8")
    print(f"\nwrote {OUTPUT}")


if __name__ == "__main__":
    asyncio.run(main())
