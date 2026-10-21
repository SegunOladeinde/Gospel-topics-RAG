"""Record how the answer pipeline behaves on a fixed set of questions.

For each question in tests/eval/questions.json this runs the real pipeline
(real indexes, real OpenAI calls) up to the first streamed token and records:
the intent, the source ids, the answer length, and how long each step took
(intent, rewrite, retrieval, time to first token). Used to compare behaviour
before and after a dependency upgrade or a performance change.

Usage:
    uv run python scripts/eval_baseline.py eval/baseline_0.3.json
    uv run python scripts/eval_baseline.py eval/after_1x.json --runs 3

Costs a few cents of OpenAI usage per run (one embedding plus one or two
chat calls per question).
"""

import argparse
import asyncio
import importlib.metadata
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")

import src.backend.main as backend  # noqa: E402
from src.backend.main import Message  # noqa: E402

QUESTIONS = PROJECT_ROOT / "tests" / "eval" / "questions.json"
PACKAGES = ["langchain", "langchain-core", "langchain-openai", "langchain-community", "langchain-text-splitters", "openai"]


def versions() -> dict[str, str | None]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            found[name] = None
    return found


class StepTimer:
    """Wraps the pipeline's steps (whichever exist) to time them and catch the intent."""

    def __init__(self):
        self.ms: dict[str, float] = {}
        self.intent: str | None = None

    def wrap(self, owner, name: str, step: str, capture_intent: bool = False):
        original = getattr(owner, name, None)
        if original is None:
            return

        async def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = await original(*args, **kwargs)
            finally:
                self.ms[step] = self.ms.get(step, 0.0) + (time.perf_counter() - start) * 1000
            if capture_intent:
                self.intent = getattr(result, "intent", result)
            return result

        setattr(owner, name, timed)
        return original


class TimedRetriever:
    """Stands in for the retriever (a pydantic model can't be patched) and times it."""

    def __init__(self, inner, timer: StepTimer):
        self.inner = inner
        self.timer = timer

    def __getattr__(self, name):
        return getattr(self.inner, name)

    async def _timed(self, method, query):
        start = time.perf_counter()
        try:
            return await method(query)
        finally:
            self.timer.ms["retrieval"] = self.timer.ms.get("retrieval", 0.0) + (time.perf_counter() - start) * 1000

    async def ainvoke(self, query, *args, **kwargs):
        return await self._timed(self.inner.ainvoke, query)

    async def aretrieve(self, query):
        return await self._timed(self.inner.aretrieve, query)


async def run_once(item: dict, retriever, chat_model) -> dict:
    question = item["question"]
    history = [Message(**m) for m in item.get("history", [])]
    backend._INTENT_CACHE.clear()  # measure the real calls, not cache hits

    timer = StepTimer()
    originals = {
        "classify_intent": timer.wrap(backend, "classify_intent", "intent", capture_intent=True),
        "rewrite_to_standalone": timer.wrap(backend, "rewrite_to_standalone", "rewrite"),
    }
    timed_retriever = TimedRetriever(retriever, timer)
    try:
        start = time.perf_counter()
        prepared = await backend.prepare_answer(question, history, timed_retriever, chat_model)
        prepare_ms = (time.perf_counter() - start) * 1000

        ttft_ms = None
        answer = ""
        if prepared.fixed_answer is not None:
            answer = prepared.fixed_answer
            ttft_ms = prepare_ms
        else:
            async for chunk in chat_model.astream(prepared.llm_messages):
                text = backend._chunk_text(chunk) if hasattr(backend, "_chunk_text") else str(chunk.content)
                if text and ttft_ms is None:
                    ttft_ms = (time.perf_counter() - start) * 1000
                answer += text
        total_ms = (time.perf_counter() - start) * 1000
    finally:
        for name, original in originals.items():
            if original is not None:
                setattr(backend, name, original)

    # The pipeline may report its own step timings and intent (newer code).
    steps = dict(timer.ms)
    steps.update(getattr(prepared, "timings", None) or {})
    intent = getattr(prepared, "intent", None) or timer.intent or "CHITCHAT/none"
    return {
        "intent": intent,
        "sources": prepared.sources,
        "answer_chars": len(answer),
        "steps_ms": {k: round(v) for k, v in steps.items()},
        "prepare_ms": round(prepare_ms),
        "ttft_ms": round(ttft_ms) if ttft_ms is not None else None,
        "total_ms": round(total_ms),
        "answer_preview": answer[:160],
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", help="where to write the JSON results, e.g. eval/baseline_0.3.json")
    parser.add_argument("--runs", type=int, default=2, help="runs per question; timings use the median")
    args = parser.parse_args()

    retriever = backend.load_retriever()
    chat_model = backend.ChatOpenAI(model=backend.CHAT_MODEL, temperature=0)
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))

    results = []
    for item in items:
        runs = []
        for _ in range(args.runs):
            try:
                runs.append(await run_once(item, retriever, chat_model))
            except Exception as exc:  # noqa: BLE001 - record and carry on
                runs.append({"error": f"{type(exc).__name__}: {exc}"})
        ok = [r for r in runs if "error" not in r]
        first = ok[0] if ok else runs[0]
        median = lambda key: round(statistics.median(r[key] for r in ok if r.get(key) is not None)) if ok else None  # noqa: E731
        results.append({
            "id": item["id"],
            "kind": item["kind"],
            "question": item["question"],
            **{k: first.get(k) for k in ("intent", "sources", "answer_chars", "answer_preview", "error")},
            "intents_all_runs": [r.get("intent") for r in runs],
            "steps_ms_median": {
                step: round(statistics.median(r["steps_ms"].get(step, 0) for r in ok))
                for step in sorted({s for r in ok for s in r["steps_ms"]})
            } if ok else {},
            "prepare_ms_median": median("prepare_ms"),
            "ttft_ms_median": median("ttft_ms"),
            "total_ms_median": median("total_ms"),
        })
        print(f"{item['id']:<32} {results[-1]['intent'] or '-':<14} ttft={results[-1]['ttft_ms_median']}ms "
              f"steps={results[-1]['steps_ms_median']} sources={len(results[-1]['sources'] or [])}", flush=True)

    out = PROJECT_ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"versions": versions(), "runs_per_question": args.runs, "results": results},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    asyncio.run(main())
