# pvi - measure & optimize text for AI agents

A tiny tool to score and improve how much a piece of text informs an AI agent's
decision, using **Pointwise V-Information (PVI)**. Companion to
[*Mecha-nudges for Machines*](https://giuliofrey.eu/mecha-nudges/) (Frey &
Ethayarajh, 2026).

Use it three ways:
- **Yourself, from the terminal** - readable output, an interactive task wizard.
- **Through an AI agent** (Claude) - `SKILL.md` makes Claude drive it for you.
- **From other tools** - everything also prints JSON.

## What is PVI here?

PVI measures, in bits, how much a text reduces an AI's uncertainty about a
decision:

```
PVI = H(Y | empty) - H(Y | text)
```

- `PVI > 0`  the text makes the decision more predictable (it informs the agent).
- `PVI ~ 0`  uninformative.
- `PVI < 0`  the text pushes the agent *away* from the target decision.

The paper computes this with two fine-tuned classifiers. This tool makes one
pragmatic assumption: **a general instruction-tuned model, prompted zero-shot, is
a good enough stand-in for that classifier on an arbitrary task.** That gives a
training-free *pseudo-PVI* that runs entirely through an API / CLI.

**Single-shot.** `H(Y|empty)` depends only on the task, so it is computed once and
cached. After that, scoring any text is one model call.

## Backends

| Backend | How it gets P(label) | Notes |
|---|---|---|
| `openai` (default) | True token logprobs, labels constrained via `logit_bias`. | Most faithful, smooth signal. Best for `optimize`. Needs `OPENAI_API_KEY`. |
| `claude` | `claude -p` reports a probability per label (verbalized). | No API key, no python deps. Probabilities are coarse, so noisier - fine for `score`, weak for `optimize`. |

Absolute PVI is **not** comparable across backends or models - compare within one.

## Install

```bash
# openai backend
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...

# claude backend: just needs a working `claude` CLI (Claude Code). No pip needed.
```

## Define a task (no JSON editing required)

```bash
python pvi.py init          # asks for the decision, options, and target; writes task.json
```

Or write it by hand (`examples/task.json`):

```json
{
  "name": "buy-or-skip",
  "question": "You are an autonomous shopping agent. Based only on the product text, do you BUY this product or SKIP it?",
  "labels": ["BUY", "SKIP"],
  "target_label": "BUY"
}
```

Keep `labels` short, single-word, and with **distinct first letters**.
`target_label` is required for `optimize`.

## Use

Global flags go **before** the subcommand. In a terminal you get readable output;
piped or used by an agent you get JSON (override with `--format`).

```bash
# Score one text
python pvi.py --task task.json score --text "Handmade ceramic mug, 12oz, lead-free glaze, dishwasher safe."

# Score a dataset -> mean PVI (pseudo V-information)
python pvi.py --task task.json score --data examples/data.jsonl --text-field text

# Which words carry the information?
python pvi.py --task task.json attribute --text "..."

# Rewrite to raise PVI toward target_label (use the openai backend for best results)
python pvi.py --task task.json --backend openai optimize --text "..." --rounds 3 --candidates 5

# No OpenAI key? Run the whole thing through Claude:
python pvi.py --backend claude --model sonnet --task task.json score --text "..."
```

Flags: `--backend {openai,claude}`, `--model`, `--gen-model` (optimize rewriter),
`--rounds`, `--candidates`, `--format {auto,json,human}`, `--no-cache`.

## How `optimize` works

A loop: attribute which spans help/hurt -> ask `--gen-model` for N faithful
rewrites primed with those insights -> score each with PVI -> keep the best ->
repeat until a round stops improving. Returns before/after PVI, the gain, and the
full trajectory. If gains stall, prefer `--backend openai`, raise `--candidates`,
or use a stronger `--gen-model`.

## Caveats (please read)

- **One-proxy Goodhart risk.** Scores reflect the single `--model` you choose.
  Re-score with another `--model` to sanity-check direction.
- **Faithfulness.** `optimize` is instructed to stay truthful and preserve
  meaning/length, but it can drift. Review rewrites before using them.
- **Pseudo, not exact.** A zero-shot proxy for the paper's fine-tuned
  V-information, not a reproduction of it.
- **claude backend is coarse.** Verbalized probabilities are low-resolution.

## Model requirements

The `openai` scorer needs a model with `logprobs` + `logit_bias` (`gpt-4o-mini`,
`gpt-4o`). The `claude` scorer works with any model the `claude` CLI accepts
(`haiku`, `sonnet`, `opus`).

## For AI agents

`AGENTS.md` is the full agent-facing reference (output schemas, decision tree,
error handling, example session). `SKILL.md` is the Claude Code skill manifest.
