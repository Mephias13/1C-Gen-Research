#!/usr/bin/env python3
"""
bslgen — локальный генератор кода 1С (BSL), которым дирижирует Claude Code.

Поток: план (от Claude Code) + RAG-грунт -> локальная модель (Ollama) -> гейт -> ремонт (<=2) -> код.
Каждая попытка логируется в data/pairs.jsonl; прошедшие гейт пары — будущий датасет для QLoRA.

БЕЗ Anthropic API. «Opus» (планирование/ревью/эскалация) — это Claude Code на подписке, снаружи этого скрипта.

Пример:
  python bslgen.py --category zapros --plan-text "Функция ОстаткиТоваров(Склад): запрос к РегНакопления ТоварыНаСкладах, СрезПоследних, вернуть таблицу" --context context.txt
  echo "<план>" | python bslgen.py --category formy --plan -
  python bslgen.py --selftest   # проверка гейта без модели/сети
"""
import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
TOOLS = ROOT / "tools"
_jre = list((TOOLS / "jre21").glob("*/bin/java.exe"))
JAVA = os.environ.get("BSLGEN_JAVA") or (str(_jre[0]) if _jre else "java")  # BSL LS требует Java 21
_jar = TOOLS / "bsl-ls.jar"
BSL_LS_JAR = os.environ.get("BSL_LS_JAR") or (str(_jar) if _jar.exists() else "")
# severity=Error диагностики, которые НЕ ломают компиляцию/выполнение (стилевые) — игнор для compile-прокси гейта.
_GATE_IGNORE = set(c for c in os.environ.get("BSLGEN_GATE_IGNORE", "FunctionShouldHaveReturn").split(",") if c)
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("BSLGEN_MODEL", "qwen3.5:4b")  # 4b быстр и влезает в 8 ГБ; 9b — медленно (спилл в CPU)

SYSTEM = (
    "Ты — генератор кода 1С:Предприятие 8 (встроенный язык, BSL). "
    "По плану и контексту конфигурации верни ТОЛЬКО рабочий код на встроенном языке 1С "
    "(русские ключевые слова: Процедура/Функция, Если/Тогда/КонецЕсли, Цикл/КонецЦикла, Запрос и т.д.). "
    "Используй ТОЛЬКО имена объектов/реквизитов/процедур из контекста — ничего не выдумывай. "
    "Без пояснений, без markdown-заборов, без текста до или после кода."
    # /no_think: soft-switch Qwen3 — наш дообученный bsl-exec (unsloth-Modelfile)
    # игнорирует API-флаг think:false, а /no_think в system гасит размышления.
    # Для не-thinking моделей это безвредный текст. extract_code срежет <think></think>.
    " /no_think"
)

# Пары «открывающее <-> закрывающее» ключевое слово (двуязычно). Эвристический структурный гейт.
# ponytail: это дешёвая структурная проверка (баланс блоков + непустота), НЕ настоящий статанализ.
#           upgrade: подключить BSL Language Server (env BSL_LS_JAR) для реального линта/компиляции.
BLOCKS = [
    (r"\bПроцедура\b|\bProcedure\b", r"\bКонецПроцедуры\b|\bEndProcedure\b"),
    (r"\bФункция\b|\bFunction\b", r"\bКонецФункции\b|\bEndFunction\b"),
    (r"\bЕсли\b|\bIf\b", r"\bКонецЕсли\b|\bEndIf\b"),
    (r"\bЦикл\b|\bDo\b", r"\bКонецЦикла\b|\bEndDo\b"),
    (r"\bПопытка\b|\bTry\b", r"\bКонецПопытки\b|\bEndTry\b"),
]


def ollama_generate(prompt: str, model: str, timeout: int = 180) -> str:
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "system": SYSTEM,
        "stream": False,
        "think": False,  # исполнитель не рассуждает — план даёт Claude Code; иначе вся квота уходит в thinking
        # num_ctx ограничивает контекст -> KV-кэш не раздувает память (дефолт 64k у модели = спилл в CPU);
        # num_predict ограничивает вывод -> модель не уходит в бесконечную генерацию/размышления.
        "options": {"temperature": 0.2,
                    "num_ctx": int(os.environ.get("BSLGEN_CTX", "4096")),
                    "num_predict": int(os.environ.get("BSLGEN_MAXTOK", "2048"))},
    }).encode("utf-8")
    req = urllib.request.Request(OLLAMA_URL, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8")).get("response", "")


def extract_code(text: str) -> str:
    """Снять <think>-блок (thinking-модели вроде Qwen3) и markdown-заборы."""
    # Qwen3 и др. оборачивают рассуждения в <think>...</think> — вырезаем,
    # даже если думали пустым блоком при /no_think. Если закрывающего тега нет
    # (вывод обрезан внутри размышления), берём хвост после <think>.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    text = re.sub(r"^.*?<think>", "", text, flags=re.S | re.I)  # незакрытый think → отрезать начало
    m = re.search(r"```(?:bsl|1c|1с)?\s*\n(.*?)```", text, re.S | re.I)
    return (m.group(1) if m else text).strip()


def gate(code: str) -> tuple[bool, list[str], str]:
    """Вернуть (прошло, список_ошибок, источник). BSL LS если есть, иначе структурный fallback."""
    if BSL_LS_JAR and Path(BSL_LS_JAR).exists():
        ok, errs = _gate_bsl_ls(code, BSL_LS_JAR)
        return ok, errs, "bsl-ls"
    return (*_gate_structural(code), "structural-fallback")


def _gate_structural(code: str) -> tuple[bool, list[str]]:
    errs: list[str] = []
    if not code.strip():
        return False, ["пустой вывод"]
    low = code.lower()
    if re.search(r"^\s*(def |class |import |function\s*\()", low, re.M) and "процедура" not in low and "функция" not in low:
        errs.append("похоже не на BSL (Python/JS?)")
    for opener, closer in BLOCKS:
        o = len(re.findall(opener, code, re.I))
        c = len(re.findall(closer, code, re.I))
        if o != c:
            errs.append(f"несбалансированный блок: {opener.split('|')[0]} ({o}) != {closer.split('|')[0]} ({c})")
    return (not errs), errs


def _gate_bsl_ls(code: str, jar: str) -> tuple[bool, list[str]]:
    """BSL LS analyze: severity=Error -> гейт валит; Warning пропускаем. Без платформы методы не резолвятся,
    но синтаксис/запросы/каноника ловятся. ponytail: с установленной 1С точность гейта вырастет (резолв методов)."""
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "Module.bsl").write_text(code, encoding="utf-8")
        try:
            subprocess.run([JAVA, "-jar", jar, "analyze", "--srcDir", d,
                            "--reporter", "json", "--outputDir", d, "--silent"],
                           capture_output=True, timeout=180, check=False)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            return False, [f"BSL LS недоступен: {e}"]
        report = Path(d) / "bsl-json.json"
        if not report.exists():
            return True, []  # нет отчёта — без замечаний (консервативно)
        try:
            data = json.loads(report.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return True, []
        errs = []
        for fi in data.get("fileinfos", []):
            for diag in fi.get("diagnostics", []):
                if str(diag.get("severity", "")).lower() == "error" and diag.get("code") not in _GATE_IGNORE:
                    line = diag.get("range", {}).get("start", {}).get("line", "?")
                    errs.append(f"[{diag.get('code')}] стр.{line}: {diag.get('message', '')}")
        return (not errs), errs


def build_prompt(plan: str, context: str, category: str, prior_error: str, examples=None) -> str:
    parts = []
    if examples:
        ex = "\n\n".join(f"# Задача\n{e['user']}\n# Код\n{e['code']}" for e in examples)
        parts.append("Примеры КОРРЕКТНОГО кода 1С (следуй этому синтаксису, API и стилю):\n" + ex)
    parts.append(f"Категория задачи: {category}")
    if context.strip():
        parts.append("Контекст конфигурации (используй ТОЛЬКО эти имена):\n" + context.strip())
    parts.append("План реализации:\n" + plan.strip())
    if prior_error:
        parts.append("Предыдущий код НЕ прошёл проверку:\n" + prior_error + "\nИсправь и верни только код.")
    # grounding-adherence (исполняемые провалы шли от чужих имён и переименования
    # функции; статгейт это не ловит, реальная 1С — ловит):
    parts.append(
        "ЖЁСТКИЕ ТРЕБОВАНИЯ:\n"
        "1. Сохрани сигнатуру функции/процедуры из плана ДОСЛОВНО — то же имя и те же параметры. НЕ переименовывай.\n"
        "2. Используй СТРОГО имена объектов, реквизитов, регистров, измерений и ресурсов из контекста. "
        "Не подставляй типовые имена (Номенклатура, Контрагент и т.п.), если их нет в контексте.\n"
        "Верни только код BSL."
    )
    return "\n\n".join(parts)


def log_pair(rec: dict) -> None:
    DATA.mkdir(exist_ok=True)
    with (DATA / "pairs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(plan: str, context: str, category: str, model: str, max_repair: int,
        do_log: bool, fewshot: bool = True) -> dict:
    examples = []
    if fewshot:
        try:
            from fewshot import retrieve
            examples = retrieve((plan + "\n" + context).strip(), k=3)
        except Exception as e:  # эмбеддинги недоступны / нет датасета — работаем без few-shot
            sys.stderr.write(f"[fewshot off: {e}]\n")
    prior, code, ok, errs, src = "", "", False, [], ""
    attempts = 0
    for attempt in range(max_repair + 1):
        attempts = attempt + 1
        raw = ollama_generate(build_prompt(plan, context, category, prior, examples), model)
        code = extract_code(raw)
        ok, errs, src = gate(code)
        if ok:
            break
        prior = "\n".join(errs)
    rec = {
        "ts": dt.datetime.now().isoformat(timespec="seconds"),
        "category": category, "plan": plan, "context": context,
        "code": code, "gate_ok": ok, "gate_errors": errs, "gate_source": src,
        "attempts": attempts, "model": model, "fewshot": len(examples),
    }
    if do_log:
        log_pair(rec)
    return rec


def selftest() -> int:
    good = "Функция Сумма(А, Б)\n    Возврат А + Б;\nКонецФункции"
    bad = "Функция Сумма(А, Б)\n    Возврат А + Б;"  # нет КонецФункции
    py = "def add(a, b):\n    return a + b"
    assert _gate_structural(good)[0], "good должен пройти"
    assert not _gate_structural(bad)[0], "bad (без КонецФункции) должен упасть"
    assert not _gate_structural(py)[0], "python должен упасть"
    assert not _gate_structural("")[0], "пустой должен упасть"
    print("structural selftest OK")

    if BSL_LS_JAR and Path(BSL_LS_JAR).exists():
        good_q = '''Функция ПолучитьОстатокТовара(Номенклатура, Склад) Экспорт
    Запрос = Новый Запрос;
    Запрос.Текст =
    "ВЫБРАТЬ
    |   ОстаткиТоваров.КоличествоОстаток КАК КоличествоОстаток
    |ИЗ
    |   РегистрНакопления.ТоварыНаСкладах.Остатки(, Номенклатура = &Номенклатура И Склад = &Склад) КАК ОстаткиТоваров";
    Запрос.УстановитьПараметр("Номенклатура", Номенклатура);
    Запрос.УстановитьПараметр("Склад", Склад);
    Выборка = Запрос.Выбрать();
    Если Выборка.Следующий() Тогда
        Возврат Выборка.КоличествоОстаток;
    КонецЕсли;
    Возврат 0;
КонецФункции'''
        bad_q = '''Функция ПолучитьОстатокТовара(Номенклатура, Склад) Экспорт:
    Запрос = Новый Запрос;
    Запрос.Текст = "ВЫБРАТЬ ИМЯ.КоличествоОстаток ИЗ РегистрНакопления.ТоварыНаСкладах.Остатки WHERE ИМЯ.Номенклатура = :Номенклатура";
    Возврат 0;
КонецФункции'''
        ok_g, errs_g, src_g = gate(good_q)
        ok_b, errs_b, _ = gate(bad_q)
        assert src_g == "bsl-ls", f"ожидался bsl-ls, был {src_g}"
        assert ok_g, f"корректный BSL должен пройти, гейт дал: {errs_g}"
        assert not ok_b, "битый BSL должен упасть в BSL LS"
        print(f"bsl-ls selftest OK (good чисто; bad ошибок: {len(errs_b)})")
    else:
        print("bsl-ls selftest SKIP (нет jar)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Локальный генератор BSL под управлением Claude Code (без API).")
    ap.add_argument("--plan", help="файл с планом, или '-' для stdin")
    ap.add_argument("--plan-text", help="план строкой")
    ap.add_argument("--context", help="файл с RAG-грунтом (имена объектов/реквизитов/процедур)")
    ap.add_argument("--context-text", help="RAG-грунт строкой")
    ap.add_argument("--category", default="general", help="срез: zapros|formy|provedenie|...")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-repair", type=int, default=2)
    ap.add_argument("--out", help="куда записать итоговый код")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--no-fewshot", action="store_true", help="отключить few-shot RAG")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()

    if a.plan_text:
        plan = a.plan_text
    elif a.plan == "-":
        plan = sys.stdin.read()
    elif a.plan:
        plan = Path(a.plan).read_text(encoding="utf-8")
    else:
        ap.error("нужен --plan-text, --plan <файл> или --plan -")
    if a.context_text:
        context = a.context_text
    elif a.context:
        context = Path(a.context).read_text(encoding="utf-8")
    else:
        context = ""

    rec = run(plan, context, a.category, a.model, a.max_repair, not a.no_log, fewshot=not a.no_fewshot)
    if a.out:
        Path(a.out).write_text(rec["code"], encoding="utf-8")
    sys.stderr.write(f"[gate {'OK' if rec['gate_ok'] else 'FAIL'} via {rec['gate_source']}, "
                     f"попыток={rec['attempts']}] {rec['gate_errors']}\n")
    print(rec["code"])
    return 0 if rec["gate_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
