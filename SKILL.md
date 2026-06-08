---
name: pvi
description: "Measure and optimize how much a piece of text informs an AI agent's decision, using pointwise V-information (PVI). Use when the user wants to (1) score how 'machine-readable' or persuasive-to-AI a text is, (2) see which words/phrases drive an AI's decision, or (3) rewrite text so an AI agent is more likely to make a target decision (a 'mecha-nudge'). Triggers on 'PVI', 'V-information', 'optimize for AI', 'make this rank with AI agents', 'how informative is this to an LLM', 'mecha-nudge'."
---

# PVI Skill

Guide a non-technical user through measuring and optimizing text for an AI
agent's decision, using **pseudo-PVI**. The user never has to write code; you run
the `pvi.py` CLI and explain the results in plain language.

Companion to *Mecha-nudges for Machines* (Frey & Ethayarajh, 2026).
**Read `AGENTS.md` for the full reference** (math, every flag, decision tree,
error handling, example session). This file is the quick start.

## What it does

PVI measures, in bits, how much a text reduces an AI's uncertainty about a
decision: `PVI = H(Y|empty) - H(Y|text)`. Higher = the text makes the decision
more decisive. We approximate the paper's fine-tuned classifier with a general
model prompted zero-shot, so no training is needed and every score is a single
model call (the empty-input baseline is computed once and cached).

## Backends

- `--backend openai` (default): true label logprobs. Most faithful; **prefer for
  `optimize`**. Needs `OPENAI_API_KEY` + `pip install -r requirements.txt`.
- `--backend claude`: runs `claude -p` and asks for a probability per label
  (verbalized). No API key, no python deps - but probabilities are coarse, so
  it's noisier (fine for `score`, weak for `optimize`).

If the user has no OpenAI key, use `--backend claude`. Absolute PVI is **not**
comparable across backends/models - compare only within one.

## Setup

`OPENAI_API_KEY` for the openai backend (else `! export OPENAI_API_KEY=...`), or a
working `claude` CLI for the claude backend. Run commands from this directory.

## Workflow

### 1. Define the task (always first)
If the user has no task file, run the wizard or write the JSON yourself:
```
python pvi.py init          # asks: name, question, options, target -> writes task.json
```
Fields: `name`, `question` (the decision phrased to the agent), `labels` (short,
single words, **distinct first letters**), `target_label` (the choice to optimize
toward; required for `optimize`).

### 2. Run the command (global flags go BEFORE the subcommand)

| User wants | Command |
|---|---|
| Score one text | `python pvi.py --task task.json score --text "..."` |
| Score a dataset (V-info) | `python pvi.py --task task.json score --data data.jsonl --text-field text` |
| See which words matter | `python pvi.py --task task.json attribute --text "..."` |
| Improve the text | `python pvi.py --task task.json --backend openai optimize --text "..."` |

Flags: `--backend`, `--model`, `--gen-model` (optimize rewriter), `--rounds`,
`--candidates`, `--format {auto,json,human}`, `--no-cache`.

### 3. Read the output
Add `--format json` when you run it so you get parseable output (auto-format
prints JSON when not in a terminal, which is your case). Translate it: report
`pvi` in bits and what it means, list top `attribute` spans, or show
original->best for `optimize`. Output schemas are in `AGENTS.md`.

## Always tell the user
1. **It's a proxy.** Reflects ONE model; may not transfer. Offer to re-score with
   a different `--model`.
2. **Check faithfulness.** `optimize` can drift; review rewrites.

## Notes
- Baseline cached in `.pvi_cache/` per task+backend+model; `--no-cache` recomputes.
- No task yet? `python pvi.py init`, or adapt `examples/task.json`.
