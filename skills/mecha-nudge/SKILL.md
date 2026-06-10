---
name: mecha-nudge
description: "Measure and optimize how much a piece of text informs an AI agent's decision, using pointwise V-information (PVI). Use when the user wants to (1) score how 'machine-readable' or persuasive-to-AI a text is, (2) see which words/phrases drive an AI's decision, or (3) rewrite text so an AI agent is more likely to make a target decision (a 'mecha-nudge'). Triggers on 'PVI', 'V-information', 'optimize for AI', 'make this rank with AI agents', 'how informative is this to an LLM', 'mecha-nudge'."
---

# Mecha-nudge Skill

Guide a non-technical user through measuring and optimizing text for an AI
agent's decision, using **pseudo-PVI**. The user never has to write code; you run
the `mecha_nudge.py` CLI and explain the results in plain language.

Companion to *Mecha-nudges for Machines* (Frey & Ethayarajh, 2026).
**Read `AGENTS.md` for the full reference** (math, every flag, decision tree,
error handling, example session). This file is the quick start.

## What it does

PVI measures, in bits, how much a text increases the **predictability** of the
agent's decision (its usable information about which label the agent picks),
relative to a reference prior: `PVI = log2[ p_text(y) / p_baseline(y) ]`. The paper
scores the agent's *observed* decision; this tool fixes `y = target_label`, so a
higher PVI means the text makes the agent **more likely to (predictably) choose
the target**. We approximate the paper's fine-tuned classifier with a general model
prompted zero-shot, so no training is needed and every score is a single model call.

**Baseline (`--baseline`, a global flag).** The reference `p_baseline(y)`:
- `neutral` (**default**) — uniform prior over labels, so PVI is *bits above chance*
  (0 = uninformative, `+log2(K)` = fully decides, negative = points away). No API call.
- `empty` — the model's empty-input response (the paper's `H(Y|empty)` analogue). A
  zero-shot model treats an empty input as "no info → decline", so this is **extreme
  and inflates scores**; opt in only as a diagnostic. Computed once per task+model and cached.

## Scorer

P(label) comes from the **OpenAI API**: true label logprobs with the labels
constrained via `logit_bias` (a label-masking trick) - smooth and faithful.
Set the scorer with `--model <name>` or `$MECHA_NUDGE_MODEL` (there is no default); it
must support `logprobs` + `logit_bias` on the Chat Completions API. Absolute PVI
is **not** comparable across models - compare only within one `--model`.

## Setup

Install deps with `pip install -r requirements.txt` (from this skill dir), or
`pip install mecha-nudge` / `pip install "git+https://github.com/giuliofrey/mecha-nudges-skill"` to also get a `mecha-nudge`
command on PATH.

**Invoking the CLI** (commands run from anywhere — the cache is global):
- pip-installed: `mecha-nudge …`
- skill / clone: `python mecha_nudge.py` from this folder.

**OpenAI key** — supply one of (precedence:
`--api-key` > `OPENAI_API_KEY` env var > `./.env` > `~/.config/mecha-nudge/.env`):
- `! export OPENAI_API_KEY=sk-...`
- `OPENAI_API_KEY=sk-...` in `~/.config/mecha-nudge/.env` (persistent; gitignored) or a
  local `./.env` (`cp .env.example ~/.config/mecha-nudge/.env`)
- `--api-key sk-...` on the command (warn: lands in shell history / `ps`)

## Workflow

### 1. Define the task (always first)
If the user has no task file, run the wizard or write the JSON yourself
(`mecha-nudge` below = whichever invocation from Setup applies):
```
mecha-nudge init          # asks: name, question, options, target -> writes task.json
```
Fields: `name`, `question` (the decision phrased to the agent), `labels` (short,
single words, **distinct first letters**), `target_label` (the choice to optimize
toward; required for `optimize`).

### 2. Run the command (global flags go BEFORE the subcommand)

| User wants | Command |
|---|---|
| Score one text | `mecha-nudge --task task.json score --text "..."` |
| Score a dataset (V-info) | `mecha-nudge --task task.json score --data data.jsonl --text-field text` |
| See which words matter | `mecha-nudge --task task.json attribute --text "..."` |
| Improve the text | `mecha-nudge --task task.json optimize --text "..."` |

Flags: `--baseline {neutral,empty}` (default neutral), `--model`, `--gen-model`
(optimize rewriter), `--rounds`, `--candidates`, `--format {auto,json,human}`, `--no-cache`.

### 3. Read the output
Add `--format json` when you run it so you get parseable output (auto-format
prints JSON when not in a terminal, which is your case). Translate it: report
`pvi` in bits and what it means, list top `attribute` spans, or show
original->best for `optimize`. Output schemas are in `AGENTS.md`.

## Always tell the user
1. **It's a proxy, not the paper.** No fine-tuning; reflects ONE model; may not
   transfer. Offer to re-score with a different `--model` and compare direction,
   not absolute bits.
2. **The default baseline is `neutral`** (bits above chance). `--baseline empty`
   is a diagnostic and inflates the numbers — say which one a score used.
3. **Bits near p≈0 or p≈1 are noisy** (~1 bit, API nondeterminism); don't over-read
   sub-bit differences. `attribute` is uninformative on a saturated text (all deltas ~0).
4. **Check faithfulness.** `optimize` can drift; review rewrites.

## Notes
- `--baseline empty` is cached in `~/.config/mecha-nudge/cache/` per task+model;
  `--no-cache` recomputes it. `neutral` needs no call and no cache.
- No task yet? `mecha-nudge init`, or adapt `examples/task.json`.
