#!/usr/bin/env python3
"""Гейтит примеры из seed_data.py через BSL LS, пишет прошедшие в data/dataset.jsonl (chat-формат для QLoRA).
Железное правило: в обучение идёт только код, прошедший гейт (severity=Error отбраковывает)."""
import json
from bslgen import gate, DATA
from seed_data import EXAMPLES


def to_user(ex: dict) -> str:
    parts = [ex["instruction"]]
    if ex.get("context"):
        parts.append("Контекст: " + ex["context"])
    return "\n".join(parts)


def main() -> None:
    DATA.mkdir(exist_ok=True)
    out = DATA / "dataset.jsonl"
    kept, failed, src = 0, [], "-"
    with out.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(EXAMPLES):
            ok, errs, src = gate(ex["code"])
            if ok:
                rec = {"messages": [{"role": "user", "content": to_user(ex)},
                                    {"role": "assistant", "content": ex["code"]}],
                       "category": ex["category"]}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
            else:
                failed.append((i, ex["category"], errs))
    print(f"gate={src}  kept={kept}/{len(EXAMPLES)}  ->  {out}")
    for i, cat, errs in failed:
        print(f"  FAIL #{i} [{cat}]: {errs}")


if __name__ == "__main__":
    main()
