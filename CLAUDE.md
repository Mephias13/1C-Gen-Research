# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A pipeline that generates **1С:Предприятие (BSL)** code for client work, cheaply and reliably. The strategy is a **system, not a smart model**: Claude Code does the planning/hard reasoning, a small local model does the bulk code generation for free, a real compiler-grade gate verifies, and a repair loop fixes errors before escalating back to Claude. Full plan/report: `C:\Users\art63\.claude\plans\nifty-drifting-lecun.md`.

## Hard constraint — NO Claude API

There is **no Anthropic API budget**. "Opus"/"the planner" = **this Claude Code session on the Max subscription**, driven interactively — never a programmatic `anthropic`-SDK call. Do **not** write SDK/API code or suggest API billing. "Token savings" means conserving the **subscription allowance**: offload high-volume BSL generation to the free local model so it doesn't burn the subscription.

## Architecture (the loop)

```
task (a function to implement)
  → Claude Code: compact plan + RAG grounding (real object/field/register names)
  → bslgen.py: local model (Ollama) generates BSL from the plan
  → gate: BSL Language Server (compile-grade static check)
        OK   → return + log (plan→code) pair        (data/pairs.jsonl)
        FAIL → local self-repair from the error text (≤2 tries)
                 still FAIL → escalate: Claude Code writes it
```

The local model is small **by hardware necessity** (RTX 4060, 8 GB). It is **not** a quality engine — its job is cheap offload of the *easy, in-slice* tasks (queries / forms+CRUD / posting+registers). Hard or novel tasks always go back to Claude Code. External benchmark reality: even flagship models score <52% on 1C; bare ≤35B models are unsuitable alone — so the scaffolding (plan + few-shot + gate + repair) matters more than the raw model.

## Key files

| File | Role |
|---|---|
| `bslgen.py` | Orchestrator + CLI + the gate. `run()` = few-shot → Ollama gen → gate → repair → log. `gate()` shells out to BSL LS. |
| `fewshot.py` | Few-shot RAG: embeds `data/dataset.jsonl` via `nomic-embed-text` (Ollama), cosine top-k, injected into the prompt. Index cached at `data/fewshot_index.json`. |
| `seed_data.py` | Hand-authored gold examples (the distillation seed). |
| `gen_synthetic.py` | Templated synthetic examples × varying object names, **idempotent** (dedups against existing), **batch-gated** in one BSL LS pass. Scale the dataset by adding templates/params here and re-running. |
| `build_dataset.py` | Gates `seed_data.EXAMPLES` → writes `data/dataset.jsonl` (chat format). `gen_synthetic.py` appends to the same file. |
| `train.py` | QLoRA fine-tune (unsloth) on `data/dataset.jsonl`. **Not run yet** — deferred (needs WSL2; Windows-Store Python is hostile to unsloth/bitsandbytes/triton). |
| `tools/` | Bundled `bsl-ls.jar` (BSL LS v1.0.1) + `jre21/` (BSL LS needs Java 21; `jre17/` is leftover/unused). Do not edit. |
| `data/dataset.jsonl` | The gated training corpus (currently ~101 pairs). |

## Commands

```bash
# Gate self-check (offline + drives real BSL LS on good/bad BSL)
python bslgen.py --selftest

# Generate one function — Claude Code drives this, passing a plan + grounding
python bslgen.py --category zapros --plan-text "<plan>" --context-text "<config metadata>"
# plan/context can also come from files (--plan f.txt --context f.txt) or stdin (--plan -)

# Build the training dataset from the hand-authored seed (gates each example)
python build_dataset.py

# Expand the dataset with synthetic templated examples (idempotent, re-runnable)
python gen_synthetic.py
```

Run scripts with the project dir as CWD, or use absolute paths (the modules import each other). `bslgen.py` anchors `data/` and `tools/` to its own location, so it works from any CWD.

### Environment variables (bslgen)
`BSLGEN_MODEL` (executor, default `qwen3.5:4b`), `BSLGEN_CTX` (4096), `BSLGEN_MAXTOK` (2048; was 768 — bumped after the baseline run showed 23% of bench outputs truncated mid-function), `OLLAMA_URL`, `BSLGEN_EMBED` (`nomic-embed-text`), `BSL_LS_JAR` / `BSLGEN_JAVA` (override bundled tools), `BSLGEN_GATE_IGNORE` (diagnostic codes to treat as non-fatal).

## Non-obvious rules

- **Iron rule: never train on un-gated code.** Only BSL LS-passing code enters `data/dataset.jsonl`.
- **The executor must be non-thinking.** Ollama "thinking" models (e.g. `qwen3.5:*`) put their answer in a separate reasoning field and leave `response` empty — `bslgen` sends `think:false` to `/api/generate` to prevent this. If you swap models or call Ollama elsewhere, disable thinking or you'll get empty output.
- **The gate is a *compile* proxy, not a linter.** Only BSL LS `severity=Error` fails the gate, **minus** style diagnostics listed in `BSLGEN_GATE_IGNORE` (default `FunctionShouldHaveReturn` — a 1C `Функция` without `Возврат` still compiles). It catches syntax / virtual-table / parse errors but, without a running 1C platform, cannot verify method existence or runtime correctness.
- **Cyrillic everywhere.** Read/write all BSL files as UTF-8; the Windows console mojibakes Cyrillic on display but files are correct. Author BSL with Russian keywords (`Процедура`/`Функция`/`Запрос`), `&Параметр` (not `:param`), `ГДЕ` (not `WHERE`), virtual tables **with** parameters.

## 1C platform & evaluation

- **Local 1C is the *training* edition** (`C:\Program Files (x86)\1cv8t\...\bin\1cv8t.exe`, 8.3.27). **No COM/Automation** (not registered in the training edition), but **batch CLI works** (`CREATEINFOBASE`, `DESIGNER /DumpConfigToFiles`, etc.). Use batch mode for config dumps / grounding; point any tool's `1cv8.exe` at `1cv8t.exe`.
- **Executable evaluation = 1C Code Bench** (sibling repo `C:\ClaudeProjects\1cbench`, `github.com/1cbench/bench`): 147 tasks, compile→execute→verify. Its runtime is a **"1C MCP Toolkit" server on `localhost:6003`**, easiest via their Dockerized image (the demo base + 1C + MCP are inside the container; bring it up with Docker, then `run_bench.py`). `1cbench/gen_bench_outputs.py` runs *our* `bslgen` over the bench tasks and writes the output CSV the bench scores. Our static gate over-counts compile (misses runtime errors) — the container is the source of truth.
```
