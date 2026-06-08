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

## Scorer

P(label) comes from the **OpenAI API**: true label logprobs with the labels
constrained via `logit_bias` (the paper's masking trick) - smooth and faithful.
Pick the model with `--model` (default `gpt-4o-mini`; it must support `logprobs`
+ `logit_bias`, e.g. `gpt-4o-mini`, `gpt-4o`). Absolute PVI is **not** comparable
across models - compare only within one `--model`.

## Setup

Install deps with `pip install -r requirements.txt` (from this skill dir), or
`pip install pvi-skill` / `pip install "git+<repo-url>"` to also get a `pvi`
command on PATH.

**Invoking the CLI** (commands run from anywhere — the cache is global):
- pip-installed: `pvi …`
- as a Claude plugin: `python "$CLAUDE_PLUGIN_ROOT/skills/pvi/pvi.py" …`
- plain skill / clone: `python pvi.py` from this folder.

**OpenAI key** — supply one of (precedence:
`--api-key` > `OPENAI_API_KEY` env var > `./.env` > `~/.config/pvi/.env`):
- `! export OPENAI_API_KEY=sk-...`
- `OPENAI_API_KEY=sk-...` in `~/.config/pvi/.env` (persistent; gitignored) or a
  local `./.env` (`cp .env.example ~/.config/pvi/.env`)
- `--api-key sk-...` on the command (warn: lands in shell history / `ps`)

## Workflow

### 1. Define the task (always first)
If the user has no task file, run the wizard or write the JSON yourself
(`pvi` below = whichever invocation from Setup applies):
```
pvi init          # asks: name, question, options, target -> writes task.json
```
Fields: `name`, `question` (the decision phrased to the agent), `labels` (short,
single words, **distinct first letters**), `target_label` (the choice to optimize
toward; required for `optimize`).

### 2. Run the command (global flags go BEFORE the subcommand)

| User wants | Command |
|---|---|
| Score one text | `pvi --task task.json score --text "..."` |
| Score a dataset (V-info) | `pvi --task task.json score --data data.jsonl --text-field text` |
| See which words matter | `pvi --task task.json attribute --text "..."` |
| Improve the text | `pvi --task task.json optimize --text "..."` |

Flags: `--model`, `--gen-model` (optimize rewriter), `--rounds`,
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
- Baseline cached in `~/.config/pvi/cache/` per task+model; `--no-cache` recomputes.
- No task yet? `pvi init`, or adapt `examples/task.json`.
