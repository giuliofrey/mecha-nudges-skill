#!/usr/bin/env python3
"""
mecha-nudge - measure and optimize how much a piece of text informs an AI agent's decision.

Companion tool to "Mecha-nudges for Machines" (Frey & Ethayarajh, 2026).

Background
----------
Pointwise V-Information (PVI) measures, in bits, how much a text increases the
*predictability* of an AI agent's decision - how much usable information the text
carries about which label the agent picks - relative to a reference prior:

    PVI = -log2 p_baseline(y) + log2 p_text(y) = log2[ p_text(y) / p_baseline(y) ]

`p_text(y)` is the probability the model assigns label y after reading the text;
`p_baseline(y)` is a reference prior. PVI > 0 means the text makes the decision
more predictable; PVI < 0 means the model predicts it better by IGNORING the text
(the decision is less predictable than the baseline). The paper computes PVI at
the agent's *observed* decision y; this tool fixes y = target_label, repurposing
the same quantity to measure how much a text pushes the agent toward that decision.

Two reference points are supported (`--baseline`):

  * neutral (DEFAULT): a uniform prior over the labels (1/K each). PVI is then
    "bits above chance" - 0 is uninformative, up to +log2(K) means the text
    fully decides the label, negative means it points away. Needs no API call
    and gives the interpretable scale the docs' thresholds assume.
  * empty: the model's response to an empty input, p(y|empty). This is the
    analogue of the paper's H(Y|empty), but a zero-shot prompted model answers
    an empty input the way a *rational agent* would (e.g. "SKIP, no info"),
    NOT with the dataset's label marginal that the paper's fine-tuned null model
    learns. That makes p(y|empty) extreme and inflates every score, so it is an
    opt-in diagnostic, not the default. Computed once per task+model and cached.

The paper computes PVI with two *fine-tuned* classifiers. Here we make a single,
pragmatic assumption: a general instruction-tuned model already behaves as a
competent classifier for an arbitrary task when prompted. We treat that model as
the "V" family. This gives a *pseudo-PVI*: a portable, training-free PROXY, not a
reproduction of the paper's numbers.

Scoring uses the OpenAI API: it reads true label logprobs with the labels
constrained via `logit_bias` (a label-masking trick), single-shot per text.

Caveats (surfaced to users by the skill):
  * Scores reflect ONE proxy model. High pseudo-PVI need not transfer to other
    agents (Goodhart's law). Absolute values are not comparable across models.
  * The OpenAI API is not perfectly deterministic at temperature 0; PVI in the
    near-0 / near-1 probability regime can wobble by ~1 bit run-to-run.
  * Optimized text must be checked for faithfulness; the loop is instructed to
    preserve meaning but does not guarantee it.

Requires: OPENAI_API_KEY, `pip install openai tiktoken`
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

FLOOR = 1e-12  # probability floor, matching the paper's v_info.py


# ===========================================================================
# Task definition
# ===========================================================================
def load_task(spec: str) -> dict:
    """Load a task from a JSON file path or an inline JSON string."""
    p = Path(spec)
    raw = p.read_text() if p.exists() else spec
    try:
        task = json.loads(raw)
    except json.JSONDecodeError as e:
        sys.exit(f"[mecha-nudge] Could not parse task ('{spec}'): {e}")
    for field in ("question", "labels"):
        if field not in task:
            sys.exit(f"[mecha-nudge] Task is missing required field '{field}'.")
    if len(task["labels"]) < 2:
        sys.exit("[mecha-nudge] Task needs at least two labels.")
    return task


def build_messages(task: dict, input_text: str) -> list[dict]:
    """Chat messages for the OpenAI logprobs scorer (one forced label token)."""
    labels = task["labels"]
    system = (
        "You are a careful decision-maker. Read the input and answer with "
        f"EXACTLY ONE of these options: {', '.join(labels)}. "
        "Reply with only that single word and nothing else."
    )
    user = f"{task['question']}\n\nInput:\n{input_text}\n\nAnswer:"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _bits(p: float) -> float:
    return -math.log2(max(p, FLOOR))


def _normalize(probs: dict, labels: list[str]) -> dict:
    out = {lab: max(probs.get(lab, FLOOR), FLOOR) for lab in labels}
    total = sum(out.values()) or 1.0
    return {lab: out[lab] / total for lab in labels}


# ===========================================================================
# Scorer (OpenAI)
# ===========================================================================
class OpenAIBackend:
    """Exact pseudo-PVI: reads label logprobs with the labels constrained."""

    kind = "openai"

    def __init__(self, model: str):
        self.model = model
        self.client = _openai_client()
        self.enc = _encoding_for(model)

    def distribution(self, task: dict, input_text: str) -> dict:
        return _label_logprobs(self.client, self.model, build_messages(task, input_text),
                               task["labels"], self.enc)

    def generate(self, prompt: str, model: str | None, temperature: float = 0.8) -> str:
        resp = self.client.chat.completions.create(
            model=model or self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


def config_home() -> Path:
    """Directory holding the baseline cache and the global .env.
    Override with MECHA_NUDGE_HOME; else $XDG_CONFIG_HOME/mecha-nudge
    (default ~/.config/mecha-nudge)."""
    override = os.environ.get("MECHA_NUDGE_HOME")
    if override:
        return Path(override).expanduser()
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / "mecha-nudge"


def cache_dir() -> Path:
    return config_home() / "cache"


def _load_dotenv_file(p: Path) -> None:
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _load_dotenv() -> None:
    """Fill os.environ from .env files (KEY=value lines) without overriding the
    real environment. A local ./.env wins over ~/.config/mecha-nudge/.env."""
    _load_dotenv_file(Path(".env"))
    _load_dotenv_file(config_home() / ".env")


def _openai_client():
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("[mecha-nudge] No OpenAI key found. Pass --api-key sk-..., set OPENAI_API_KEY, "
                 "or put OPENAI_API_KEY=sk-... in a .env file (./.env or ~/.config/mecha-nudge/.env).")
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("[mecha-nudge] Install deps: pip install openai tiktoken")
    return OpenAI()


def _encoding_for(model: str):
    import tiktoken

    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("o200k_base")


def _first_token_ids(label: str, enc) -> set[int]:
    ids = set()
    for variant in (label, " " + label):
        toks = enc.encode(variant)
        if toks:
            ids.add(toks[0])
    return ids


_warned_collision = False


def _id_to_label(labels: list[str], enc) -> dict[int, str]:
    """Map each label's first-token id(s) back to the label; warn on collisions."""
    global _warned_collision
    mapping: dict[int, str] = {}
    for lab in labels:
        for tid in _first_token_ids(lab, enc):
            if tid in mapping and mapping[tid] != lab and not _warned_collision:
                print(
                    f"[mecha-nudge] Warning: labels '{mapping[tid]}' and '{lab}' start with the "
                    "same token; scores between them will be unreliable. Choose labels "
                    "with distinct first words.",
                    file=sys.stderr,
                )
                _warned_collision = True
            mapping.setdefault(tid, lab)
    return mapping


def _label_logprobs(client, model: str, messages: list[dict], labels: list[str], enc) -> dict:
    """{label: probability} from constrained logprobs (a label-masking trick)."""
    id_to_label = _id_to_label(labels, enc)
    bias = {tid: 100 for tid in id_to_label}
    resp = client.chat.completions.create(
        model=model, messages=messages, max_tokens=1, temperature=0,
        logprobs=True, top_logprobs=20, logit_bias=bias,
    )
    top = resp.choices[0].logprobs.content[0].top_logprobs
    logprob_by_label: dict[str, float] = {}
    for entry in top:
        enc_ids = enc.encode(entry.token)
        lab = id_to_label.get(enc_ids[0]) if enc_ids else None
        if lab is None:
            tok = entry.token.strip().lower()
            lab = next((l for l in labels if tok and l.lower().startswith(tok)), None)
        if lab is None:
            continue
        logprob_by_label[lab] = max(logprob_by_label.get(lab, -math.inf), entry.logprob)
    return _normalize({lab: math.exp(lp) for lab, lp in logprob_by_label.items()}, labels)


# ===========================================================================
# Baseline p_baseline(y) - the PVI reference point
#   neutral: uniform prior over labels (no model call; the default)
#   empty:   p(y | empty input), computed once per task+model and cached
# ===========================================================================
def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s)


def baseline_path(task: dict, backend) -> Path:
    name = _slug(task.get("name", "task"))
    return cache_dir() / f"{name}__{backend.kind}_{_slug(backend.model)}__baseline.json"


def compute_baseline(backend, task: dict, mode: str = "neutral", use_cache: bool = True) -> dict:
    labels = task["labels"]
    if mode == "neutral":
        # Uniform prior over the labels: a max-entropy reference that makes PVI
        # "bits above chance". Needs no API call, so nothing to cache.
        return {lab: 1.0 / len(labels) for lab in labels}
    # mode == "empty": the model's response to an empty input (the paper's
    # H(Y|empty) analogue). Depends only on the task, so compute once and cache.
    path = baseline_path(task, backend)
    if use_cache and path.exists():
        return json.loads(path.read_text())
    dist = backend.distribution(task, "")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dist, indent=2))
    return dist


# ===========================================================================
# Core: score, attribute, optimize  (all backend-agnostic)
# ===========================================================================
def score_text(backend, task: dict, text: str, baseline: dict) -> dict:
    px = backend.distribution(task, text)
    target = task.get("target_label")
    y = target if target else max(px, key=px.get)
    h_yx, h_yb = _bits(px[y]), _bits(baseline[y])
    return {
        "label": y,
        "pvi": round(h_yb - h_yx, 4),
        "H_yb": round(h_yb, 4),
        "H_yx": round(h_yx, 4),
        "p_target_given_text": round(px[y], 4),
        "p_target_baseline": round(baseline[y], 4),
        "distribution": {k: round(v, 4) for k, v in px.items()},
    }


def _map_parallel(fn, items, workers: int):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))


WORD_RE = re.compile(r"\S+")


def _spans(text: str, granularity: str) -> list[str]:
    if granularity == "sentence":
        return [p for p in re.split(r"(?<=[.!?])\s+", text.strip()) if p.strip()]
    return list(dict.fromkeys(WORD_RE.findall(text)))


def attribute(backend, task, text, baseline, granularity="word", workers=8) -> dict:
    base = score_text(backend, task, text, baseline)
    spans = _spans(text, granularity)

    def ablate(span: str) -> str:
        if granularity == "sentence":
            return re.sub(re.escape(span), "", text, count=1).strip()
        return re.sub(rf"(?<!\S){re.escape(span)}(?!\S)", "", text).strip()

    def delta_for(span: str) -> dict:
        s = score_text(backend, task, ablate(span), baseline)
        return {"span": span, "delta": round(base["pvi"] - s["pvi"], 4), "pvi_without": s["pvi"]}

    results = sorted(_map_parallel(delta_for, spans, workers), key=lambda r: r["delta"], reverse=True)
    return {"pvi": base["pvi"], "label": base["label"], "spans": results}


def _gen_candidates(backend, gen_model, task, text, target, hints, k) -> list[str]:
    helps = [h["span"] for h in hints if h["delta"] > 0][:8]
    hurts = [h["span"] for h in hints if h["delta"] < 0][:8]
    guidance = ""
    if helps:
        guidance += f"\nWords/phrases that currently HELP the '{target}' decision: {', '.join(helps)}."
    if hurts:
        guidance += f"\nWords/phrases that currently DILUTE it: {', '.join(hurts)}."
    prompt = (
        f"Task the AI agent performs: {task['question']}\n"
        f"We want to make the agent more likely to choose: {target}\n\n"
        f"Original text:\n{text}\n{guidance}\n\n"
        f"Rewrite the text in {k} different ways so an AI agent is more likely to "
        f"choose '{target}'. Hard rules: stay truthful to the original, invent no "
        f"new facts, keep the meaning and roughly the length, and keep it natural "
        f"for a human reader. Return ONLY a JSON array of {k} strings."
    )
    out = backend.generate(prompt, gen_model, temperature=0.8)
    match = re.search(r"\[.*\]", out, re.DOTALL)
    if not match:
        return []
    try:
        return [c for c in json.loads(match.group(0)) if isinstance(c, str) and c.strip()]
    except json.JSONDecodeError:
        return []


def optimize(backend, task, text, baseline, gen_model=None, rounds=3, candidates=5, workers=8) -> dict:
    target = task.get("target_label")
    if not target:
        sys.exit("[mecha-nudge] optimize needs a 'target_label' in the task (the decision to optimize toward).")

    current = text
    current_score = score_text(backend, task, current, baseline)
    trajectory = [{"round": 0, "pvi": current_score["pvi"], "text": current, "improved": None}]

    for r in range(1, rounds + 1):
        hints = attribute(backend, task, current, baseline, "word", workers)["spans"]
        cands = _gen_candidates(backend, gen_model, task, current, target, hints, candidates)
        if not cands:
            break
        scored = _map_parallel(lambda c: (c, score_text(backend, task, c, baseline)), cands, workers)
        best_c, best_s = max(scored, key=lambda cs: cs[1]["pvi"])
        improved = best_s["pvi"] > current_score["pvi"] + 1e-6
        trajectory.append({"round": r, "pvi": best_s["pvi"], "text": best_c, "improved": improved})
        if not improved:
            break
        current, current_score = best_c, best_s

    return {
        "original_text": text,
        "original_pvi": trajectory[0]["pvi"],
        "best_text": current,
        "best_pvi": current_score["pvi"],
        "gain": round(current_score["pvi"] - trajectory[0]["pvi"], 4),
        "trajectory": trajectory,
    }


# ===========================================================================
# Human-readable formatting (used when output is a terminal)
# ===========================================================================
def _interpret(pvi: float, label: str) -> str:
    if pvi > 0.5:
        return f"The text strongly informs the '{label}' decision."
    if pvi > 0.05:
        return f"The text mildly informs the '{label}' decision."
    if pvi < -0.05:
        return f"The text points the agent AWAY from '{label}'."
    return "The text is essentially uninformative to the agent."


def format_human(cmd: str, out: dict) -> str:
    if cmd == "baseline":
        mode = out.get("baseline_mode", "empty")
        src = "uniform prior over labels" if mode == "neutral" else "model's empty-input response"
        lines = [f"Baseline ({mode}: {src}) for task '{out['task']}' via {out['model']}:"]
        for lab, h in out["H_yb_per_label"].items():
            lines.append(f"  surprisal for '{lab}' = {h:+.3f} bits   (prior p={out['baseline_distribution'][lab]:.3f})")
        return "\n".join(lines)

    if cmd == "score" and "per_record" in out:
        return (f"Scored {out['n']} texts.\n"
                f"  pseudo V-information (mean PVI): {out['v_information']:+.3f} bits\n"
                f"  mean baseline surprisal={out['mean_H_yb']:.3f}  mean text surprisal={out['mean_H_yx']:.3f}")

    if cmd == "score":
        dist = "  ".join(f"{k} {v:.3f}" for k, v in out["distribution"].items())
        return (f"PVI: {out['pvi']:+.3f} bits   (decision: {out['label']}, "
                f"p={out['p_target_given_text']:.3f})\n"
                f"  baseline surprisal={out['H_yb']:.3f}  ->  text surprisal={out['H_yx']:.3f}\n"
                f"  {_interpret(out['pvi'], out['label'])}\n"
                f"  distribution: {dist}")

    if cmd == "attribute":
        helps = [s for s in out["spans"] if s["delta"] > 0][:10]
        hurts = [s for s in out["spans"] if s["delta"] < 0][-10:]
        lines = [f"PVI: {out['pvi']:+.3f} bits  (decision: {out['label']})", "",
                 "Adds information (removing it lowers PVI):"]
        lines += [f"  {s['delta']:+.3f}  {s['span']}" for s in helps] or ["  (none)"]
        lines += ["", "Dilutes (removing it raises PVI):"]
        lines += [f"  {s['delta']:+.3f}  {s['span']}" for s in reversed(hurts)] or ["  (none)"]
        return "\n".join(lines)

    if cmd == "optimize":
        lines = [f"Optimizing toward the target decision...", ""]
        for t in out["trajectory"]:
            mark = "" if t["improved"] is None else (" up" if t["improved"] else " (no gain, stop)")
            tag = "original" if t["round"] == 0 else f"round {t['round']}"
            lines.append(f"  {tag:9s} {t['pvi']:+.3f} bits{mark}")
        lines += ["", f"Best: {out['original_pvi']:+.3f} -> {out['best_pvi']:+.3f} bits "
                      f"(gain {out['gain']:+.3f})", "", out["best_text"], "",
                  "Check the rewrite stays truthful before using it."]
        return "\n".join(lines)

    return json.dumps(out, indent=2, ensure_ascii=False)


# ===========================================================================
# Interactive task builder (so non-coders never touch JSON)
# ===========================================================================
def run_init(dest: str, prog: str = "mecha-nudge"):
    print("Let's define the decision your AI agent makes.\n")
    name = input("Short name for this task (e.g. buy-or-skip): ").strip() or "task"
    question = input("The decision, phrased to the agent\n  (e.g. 'You are a shopping agent. Based only on this text, BUY or SKIP?'):\n  ").strip()
    raw_labels = input("The options, comma-separated (short, distinct words, e.g. BUY, SKIP): ").strip()
    labels = [l.strip() for l in raw_labels.split(",") if l.strip()]
    if len(labels) < 2 or not question:
        sys.exit("[mecha-nudge] Need a question and at least two labels.")
    target = input(f"Optimize toward which option? (one of {', '.join(labels)}; blank = none): ").strip()
    task = {"name": name, "question": question, "labels": labels}
    if target:
        task["target_label"] = target
    out = dest or "task.json"
    Path(out).write_text(json.dumps(task, indent=2) + "\n")
    print(f"\nWrote {out}:\n{json.dumps(task, indent=2)}")
    print(f"\nNext: {prog} --task {out} score --text \"your text here\"")


# ===========================================================================
# CLI
# ===========================================================================
def _read_records(path: str, field: str) -> list[str]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line)[field])
    return rows


def main():
    ap = argparse.ArgumentParser(
        description="Measure and optimize text for an AI agent's decision (pseudo-PVI).",
        epilog="Run the 'init' command first if you don't have a task file yet.",
    )
    ap.add_argument("--model",
                    help="OpenAI scorer model. Must support logprobs + logit_bias on the "
                         "Chat Completions API. Falls back to $MECHA_NUDGE_MODEL.")
    ap.add_argument("--api-key", help="OpenAI API key. Overrides OPENAI_API_KEY and .env. "
                    "Note: visible in shell history and `ps`; a .env file is safer.")
    ap.add_argument("--task", help="Task JSON file path or inline JSON. (Created by 'init'.)")
    ap.add_argument("--format", choices=["auto", "json", "human"], default="auto",
                    help="auto = human in a terminal, json when piped/used by an agent.")
    ap.add_argument("--baseline", choices=["neutral", "empty"], default="neutral",
                    help="PVI reference point. neutral (default) = uniform prior over labels "
                         "(interpretable 'bits above chance', no API call); empty = the model's "
                         "response to an empty input (paper's H(Y|empty) analogue, inflated for "
                         "zero-shot models).")
    ap.add_argument("--no-cache", action="store_true", help="Recompute the empty-input baseline.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Interactively create a task file (no coding).")
    sub.add_parser("baseline", help="Show the PVI reference distribution (neutral or --baseline empty).")

    sp = sub.add_parser("score", help="PVI for one text or a dataset.")
    sp.add_argument("--text")
    sp.add_argument("--data", help="JSONL file; averaged into pseudo V-information.")
    sp.add_argument("--text-field", default="text")
    sp.add_argument("--workers", type=int, default=8)

    sp = sub.add_parser("attribute", help="Which words/sentences carry the information.")
    sp.add_argument("--text", required=True)
    sp.add_argument("--granularity", choices=["word", "sentence"], default="word")
    sp.add_argument("--workers", type=int, default=8)

    sp = sub.add_parser("optimize", help="Rewrite text to raise PVI toward target_label.")
    sp.add_argument("--text", required=True)
    sp.add_argument("--rounds", type=int, default=3)
    sp.add_argument("--candidates", type=int, default=5)
    sp.add_argument("--gen-model", help="Model that proposes rewrites (defaults to the scorer model).")
    sp.add_argument("--workers", type=int, default=8)

    args = ap.parse_args()

    # Resolve the API key: real env wins over .env; --api-key overrides both.
    _load_dotenv()
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key

    # init needs no scorer/baseline and may create the task file.
    if args.cmd == "init":
        run_init(args.task, ap.prog)
        return

    if not args.task:
        sys.exit("[mecha-nudge] --task is required. Run the 'init' command to create one.")

    task = load_task(args.task)

    # Validate command arguments BEFORE any (billed) API call, so a forgotten
    # flag is reported instantly instead of after computing the baseline.
    if args.cmd == "score" and not args.data and not args.text:
        sys.exit("[mecha-nudge] score needs --text or --data.")

    model = args.model or os.environ.get("MECHA_NUDGE_MODEL")
    if not model:
        sys.exit("[mecha-nudge] No scorer model set. Pass --model <name> or set MECHA_NUDGE_MODEL. "
                 "It must support logprobs + logit_bias on the Chat Completions API.")
    backend = OpenAIBackend(model)

    try:
        out = run_command(args, task, backend)
    except _OpenAIError as e:
        sys.exit(_api_error_message(e))

    use_human = args.format == "human" or (args.format == "auto" and sys.stdout.isatty())
    print(format_human(args.cmd, out) if use_human else json.dumps(out, indent=2, ensure_ascii=False))


try:  # base class for every OpenAI SDK error; dummy fallback if the SDK is absent
    from openai import OpenAIError as _OpenAIError
except Exception:  # pragma: no cover
    class _OpenAIError(Exception):
        pass


def _api_error_message(e: Exception) -> str:
    """Turn a raw OpenAI exception into a clean, actionable one-liner."""
    name = type(e).__name__
    hint = ""
    if name == "AuthenticationError":
        hint = " (the API key was rejected - check OPENAI_API_KEY / --api-key / your .env)"
    elif name in ("NotFoundError", "BadRequestError", "PermissionDeniedError"):
        hint = (" (check --model exists and supports logprobs + logit_bias on the "
                "Chat Completions API)")
    elif name == "RateLimitError":
        hint = " (rate limit or quota - wait and retry, or lower --workers)"
    return f"[mecha-nudge] OpenAI API error: {e}{hint}"


def run_command(args, task: dict, backend) -> dict:
    """Compute the baseline and dispatch the subcommand. Raises OpenAIError on API failure."""
    baseline = compute_baseline(backend, task, args.baseline, use_cache=not args.no_cache)

    if args.cmd == "baseline":
        return {"task": task.get("name", "task"), "model": backend.model,
                "baseline_mode": args.baseline,
                "H_yb_per_label": {k: round(_bits(v), 4) for k, v in baseline.items()},
                "baseline_distribution": {k: round(v, 4) for k, v in baseline.items()}}
    if args.cmd == "score":
        if args.data:
            texts = _read_records(args.data, args.text_field)
            results = _map_parallel(lambda t: score_text(backend, task, t, baseline), texts, args.workers)
            pvis = [r["pvi"] for r in results]
            return {"n": len(pvis),
                    "v_information": round(sum(pvis) / len(pvis), 4) if pvis else 0.0,
                    "mean_H_yb": round(sum(r["H_yb"] for r in results) / max(len(results), 1), 4),
                    "mean_H_yx": round(sum(r["H_yx"] for r in results) / max(len(results), 1), 4),
                    "per_record": results}
        return score_text(backend, task, args.text, baseline)  # --text guaranteed by the early check
    if args.cmd == "attribute":
        return attribute(backend, task, args.text, baseline, args.granularity, args.workers)
    if args.cmd == "optimize":
        return optimize(backend, task, args.text, baseline, args.gen_model,
                        args.rounds, args.candidates, args.workers)
    raise SystemExit(f"[mecha-nudge] unknown command: {args.cmd}")


if __name__ == "__main__":
    main()
