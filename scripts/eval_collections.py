"""Before/after eval for adding collections, with the deployed retriever.

Retrieval: every question with a "relevant" list is searched with the app's
pgvector_bm25 retriever, built by main.build_pgvector_bm25 (grouped when
data/collections.yaml has study collections). Routed queries and their embeddings are cached
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

Comparisons re-score both runs with today's patterns. Questions tagged
"since": "phase-d" are new; the others are the existing ones whose older
sources must be kept.

Answer accuracy (--accuracy-compare, with --answers): the share of shown
source chips that are relevant, and of cited scripture references that the
shown sources' passages support (citations.check_references), against an
earlier --answers run. A question is worse on relevance when a smaller share
of its chips is relevant, or none is where one was.

Usage:
    uv run python scripts/eval_collections.py eval/accuracy_after.json --answers --accuracy-compare eval/accuracy_before.json
    uv run python scripts/eval_collections.py eval/phase_d_before.json
    uv run python scripts/eval_collections.py eval/phase_d_after.json --answers --compare eval/phase_d_before.json
    uv run python scripts/eval_collections.py eval/phase_d_tuning.json --tune --compare eval/phase_d_before.json

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
    Message,
    build_pgvector_bm25,
    describe_source,
    format_context,
)
from src.backend.citations import check_references, split_used  # noqa: E402
from src.backend.retrieval.pgvector import GroupedRetriever, SqlTransport  # noqa: E402
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
    raw, corrections = None, []
    shown = list(prepared.sources or [])
    if prepared.fixed_answer is not None:
        text = prepared.fixed_answer
    else:
        text = raw = (await chat.ainvoke(prepared.llm_messages)).content
        finalize = getattr(backend, "finalize_answer", None)  # absent before the answer-accuracy change
        if finalize:
            final = finalize(raw, prepared)
            text, shown = final.answer, final.sources
            corrections = [vars(c) for c in final.corrections]
    manual_refs = set()
    for doc in retriever.last if prepared.sources else []:
        if doc.metadata.get("source", "").startswith("institute_"):
            first = doc.page_content.split("\n\n", 2)
            manual_refs |= chapters(" ".join(first[:2]))  # the section line and its "Scriptures:" line
    cited = chapters(text)
    passages = [{"source": describe_source(d), "content": d.page_content}
                for d in (retriever.last if prepared.sources else [])]
    result = {
        "intent": prepared.intent,
        "answer": text,
        "raw_answer": raw,  # the model's own output, before the USED line and reference checks
        "corrections": corrections,
        "labels": [link.label for link in prepared.links or []],
        "shown_sources": shown,  # the source chips under the answer
        "passages": passages,  # what the model was given, for checking its references
        "cited_chapters": sorted(cited),
        "manual_chapters": sorted(manual_refs),
        "cites_manual_ref": bool(cited & manual_refs) if manual_refs else None,
    }
    if item.get("cites"):
        result["cites_expected"] = any(c in text.replace("Doctrine and Covenants", "D&C") for c in item["cites"])
    return result


def accuracy(answers: dict, items: list[dict]) -> dict:
    """Per question: shown sources that are relevant, and cited references
    that the shown sources' passages support (check_references)."""
    rows = {}
    for item in items:
        a = answers.get(item["id"])
        if not a or not a["passages"]:
            continue
        shown = a["shown_sources"]
        relevant = score(shown, patterns(item))[0] if item.get("relevant") else None
        shown_passages = [p["content"] for p in a["passages"] if p["source"] in shown]
        check = check_references(a["answer"], shown_passages)
        raw = check_references(split_used(a["raw_answer"])[0], [p["content"] for p in a["passages"]]) \
            if a.get("raw_answer") else check
        rows[item["id"]] = {"shown": len(shown), "relevant": relevant, "references": check.total,
                            "verified": check.verified, "raw_references": raw.total, "raw_verified": raw.verified,
                            "corrections": a.get("corrections", [])}
    return rows


def accuracy_report(before: dict, after: dict, items: list[dict]) -> dict:
    b, a = accuracy(before, items), accuracy(after, items)
    worse = []
    print(f"\n{'question':<36}{'relevant shown b→a':>20}{'refs verified b→a':>20}  corrections")
    for qid in a:
        if qid not in b:
            continue
        rb, ra = b[qid], a[qid]
        if rb["relevant"] is not None:
            precision_b = rb["relevant"] / rb["shown"] if rb["shown"] else 0
            precision_a = ra["relevant"] / ra["shown"] if ra["shown"] else 0
            if precision_a < precision_b or (rb["relevant"] and not ra["relevant"]):
                worse.append(qid)
        print(f"{qid:<36}{'%s/%d → %s/%d' % (rb['relevant'], rb['shown'], ra['relevant'], ra['shown']):>20}"
              f"{'%d/%d → %d/%d' % (rb['verified'], rb['references'], ra['verified'], ra['references']):>20}  "
              + "; ".join(f"{c['original']} → {c['replacement'] or '(removed)'}" for c in ra["corrections"]))

    def total(rows, key, of):
        num = sum(r[key] or 0 for r in rows.values() if r["relevant"] is not None or key != "relevant")
        den = sum(r[of] for r in rows.values() if r["relevant"] is not None or key != "relevant")
        return {"count": f"{num}/{den}", "percent": round(100 * num / den, 1) if den else None}

    summary = {
        "relevant_shown_sources": {"before": total(b, "relevant", "shown"), "after": total(a, "relevant", "shown")},
        "verified_references": {"before": total(b, "verified", "references"), "after": total(a, "verified", "references"),
                                "after_model_output_before_checks": total(a, "raw_verified", "raw_references")},
        "corrections": sum(len(r["corrections"]) for r in a.values()),
        "relevance_worse": worse,
    }
    print(json.dumps(summary, indent=1))
    return summary


def patterns(item: dict) -> list[str]:
    return item["relevant"] + item.get("relevant_new", [])


def compare(before: dict, after: dict, items: list[dict]) -> dict:
    """Per question: relevant/off-topic before and after (both scored with
    today's patterns), the verdict, and whether every older source was kept."""
    rows = {}
    for item in items:
        qid = item["id"]
        if qid not in after or qid not in before:
            continue
        b, a = score(before[qid]["sources"], patterns(item)), score(after[qid]["sources"], patterns(item))
        verdict = "WORSE" if a[0] < b[0] or a[1] > b[1] else ("same" if a == b else "better")
        kept = all(source in after[qid]["sources"] for source in before[qid]["sources"])
        rows[qid] = {"existing": item.get("since") != "phase-d", "before": b, "after": a,
                     "verdict": verdict, "older_sources_kept": kept}
    return rows


def summarise(rows: dict) -> dict:
    existing = [r for r in rows.values() if r["existing"]]
    new = [r for r in rows.values() if not r["existing"]]
    return {
        "existing_worse": [q for q, r in rows.items() if r["existing"] and r["verdict"] == "WORSE"],
        "older_sources_kept": f"{sum(r['older_sources_kept'] for r in existing)}/{len(existing)}",
        "new_relevant": sum(r["after"][0] for r in new),
        "new_off_topic": sum(r["after"][1] for r in new),
        "new_better": sum(r["verdict"] == "better" for r in new),
        "new_worse": [q for q, r in rows.items() if not r["existing"] and r["verdict"] == "WORSE"],
    }


async def search_all(retriever, searched: list[dict], inputs: dict) -> tuple[dict, float]:
    """Sources per question, and the average prompt tokens of the passages."""
    import tiktoken

    encoding = tiktoken.get_encoding("o200k_base")  # gpt-4o-mini's tokenizer
    retrieval, tokens = {}, []
    for item in searched:
        docs = await retriever.ainvoke(inputs[item["id"]]["query"])
        sources = list(dict.fromkeys(describe_source(d) for d in docs))
        labels = {}
        for d in docs:
            labels.setdefault(describe_source(d), source_link_for(describe_source(d), d.metadata.get("title"),
                                                                   d.metadata.get("url")).label)
        tokens.append(len(encoding.encode(format_context(docs))))
        retrieval[item["id"]] = {
            "query": inputs[item["id"]]["query"],
            "sources": sources,
            "labels": [labels[s] for s in sources],
            "strict": score(sources, item["relevant"]),
            "full": score(sources, patterns(item)),
        }
    return retrieval, sum(tokens) / len(tokens)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--answers", action="store_true", help="also generate and check full answers")
    parser.add_argument("--compare", type=Path, help="an earlier run to compare retrieval with")
    parser.add_argument("--accuracy-compare", type=Path,
                        help="an earlier --answers run: compare shown-source relevance and verified references")
    parser.add_argument("--tune", action="store_true",
                        help="score study-group caps 2-4 x thresholds 0.55/0.60 against --compare")
    args = parser.parse_args()

    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    searched = [i for i in items if i.get("relevant")]
    inputs = await prepare_inputs(searched, refresh=False)
    embeddings = CachedEmbeddings({v["query"]: v["embedding"] for v in inputs.values()})
    retriever = build_pgvector_bm25(SqlTransport(get_session_factory()), embeddings)  # exactly the app's
    print(f"retriever: {type(retriever).__name__}, passages: {retriever.chunk_counts()['vector']:,}")
    before = json.loads(args.compare.read_text(encoding="utf-8"))["retrieval"] if args.compare else None

    if args.tune:
        if not isinstance(retriever, GroupedRetriever) or before is None:
            sys.exit("--tune needs study collections and --compare")
        table = []
        for relative in (False, True):
          for cap in (2, 3, 4):
            for threshold in (0.55, 0.60):
                retriever.max_added, retriever.min_similarity, retriever.relative_to_core = cap, threshold, relative
                retrieval, avg_tokens = await search_all(retriever, searched, inputs)
                row = {"cap": cap, "threshold": threshold, "relative_to_core": relative,
                       "avg_prompt_tokens": round(avg_tokens),
                       **summarise(compare(before, retrieval, items)), "retrieval": retrieval}
                table.append(row)
                print(f"{'relative ' if relative else ''}cap {cap} min {threshold:.2f}: existing worse {len(row['existing_worse'])} "
                      f"{row['existing_worse']} | older kept {row['older_sources_kept']} | new rel/off "
                      f"{row['new_relevant']}/{row['new_off_topic']}, better {row['new_better']}, worse "
                      f"{row['new_worse']} | avg prompt tokens {row['avg_prompt_tokens']}", flush=True)
        args.output.write_text(json.dumps(table, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output}")
        return

    retrieval, avg_tokens = await search_all(retriever, searched, inputs)
    result = {"passages": retriever.chunk_counts()["vector"], "avg_prompt_tokens": round(avg_tokens),
              "retrieval": retrieval}

    if args.answers:
        chat = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        live = Capturing(retriever)
        result["answers"] = {}
        for item in items:
            result["answers"][item["id"]] = await answer(item, live, chat)
            a = result["answers"][item["id"]]
            print(f"  answered {item['id']:<36} cites expected: {a.get('cites_expected', '-')!s:<5} "
                  f"cites a manual ref: {a['cites_manual_ref']!s}", flush=True)

    if args.accuracy_compare and "answers" in result:
        earlier = json.loads(args.accuracy_compare.read_text(encoding="utf-8"))["answers"]
        result["accuracy"] = accuracy_report(earlier, result["answers"], items)

    if before is not None:
        rows = compare(before, retrieval, items)
        result["comparison"] = rows
        result["summary"] = summarise(rows)
        print(f"\n{'question':<36}{'before':>9}{'after':>9}  verdict  older sources kept")
        for qid, r in rows.items():
            print(f"{qid:<36}{'%d/%d' % tuple(r['before']):>9}{'%d/%d' % tuple(r['after']):>9}  "
                  f"{r['verdict']:<7}  {r['older_sources_kept']}")
        print(json.dumps(result["summary"], indent=1))

    args.output.parent.mkdir(exist_ok=True)
    args.output.write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
