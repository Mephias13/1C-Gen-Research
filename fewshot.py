# -*- coding: utf-8 -*-
"""Few-shot RAG: достаём похожие gated-примеры из data/dataset.jsonl и кладём в промпт.
Эмбеддинги — nomic-embed-text через Ollama. Без обучения: даёт основную пользу fine-tune здесь и сейчас.
ponytail: косинус по всем элементам O(n) на запрос — ок для сотен; при десятках тысяч завести нормальный ANN."""
import json
import math
import os
import time
import urllib.error
import urllib.request
from bslgen import DATA  # noqa: E402  (ленивая связка путей)

EMBED_MODEL = os.environ.get("BSLGEN_EMBED", "nomic-embed-text")
EMBED_URL = os.environ.get("OLLAMA_EMBED_URL", "http://localhost:11434/api/embeddings")
DATASET = DATA / "dataset.jsonl"
INDEX = DATA / "fewshot_index.json"


def embed(text: str) -> list[float]:
    # ponytail: REAL root cause of the gen-time 500s — the retrieval query is
    # plan+full-JSON-context (3.8k–5k chars), which overflows nomic's 2048-token
    # window and Ollama answers 500. Truncate to a safe char budget (the plan
    # leads the string, so it always survives; the JSON tail is only a similarity
    # signal). Verified: ≤3500 chars OK, ~3843+ fails — 2000 leaves token-density
    # margin. num_gpu=0 keeps embed off the GPU; retry covers transient blips.
    text = text[:2000]
    body = json.dumps({
        "model": EMBED_MODEL, "prompt": text,
        "keep_alive": -1, "options": {"num_gpu": 0},
    }).encode("utf-8")
    req = urllib.request.Request(EMBED_URL, data=body, headers={"Content-Type": "application/json"})
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))["embedding"]
        except (urllib.error.HTTPError, urllib.error.URLError) as e:
            last = e
            time.sleep(1.5 * (attempt + 1))  # 1.5s, 3s, 4.5s
    raise last


def _cos(a: list[float], b: list[float]) -> float:
    s = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return s / (na * nb) if na and nb else 0.0


def build_index() -> list[dict]:
    items = []
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        msgs = {m["role"]: m["content"] for m in rec["messages"]}
        items.append({"user": msgs.get("user", ""), "code": msgs.get("assistant", ""),
                      "vec": embed(msgs.get("user", ""))})
    INDEX.write_text(json.dumps({"model": EMBED_MODEL, "items": items}, ensure_ascii=False), encoding="utf-8")
    return items


def _load() -> list[dict]:
    if not DATASET.exists():
        return []
    if (not INDEX.exists()) or INDEX.stat().st_mtime < DATASET.stat().st_mtime:
        return build_index()  # датасет обновился -> переиндексируем
    return json.loads(INDEX.read_text(encoding="utf-8")).get("items", [])


def retrieve(query: str, k: int = 3, exclude_exact: bool = True) -> list[dict]:
    """Top-k похожих примеров. exclude_exact отсекает почти-дубль запроса (для честной оценки)."""
    items = _load()
    if not items:
        return []
    qv = embed(query)
    ranked = sorted(items, key=lambda it: _cos(qv, it["vec"]), reverse=True)
    if exclude_exact and ranked and _cos(qv, ranked[0]["vec"]) > 0.985:
        ranked = ranked[1:]
    return ranked[:k]
