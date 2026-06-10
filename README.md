# Mecha-nudges: measure & optimize how much a text informs an AI agent's decision

A tiny tool to score and improve how much a piece of text informs an AI agent's
decision, using **Pointwise V-Information (PVI)**. Companion to
[*Mecha-nudges for Machines*](https://giuliofrey.eu/mecha-nudges/) (Frey &
Ethayarajh, 2026).

Use it three ways:
- **Yourself, from the terminal**: readable output, an interactive task wizard.
- **Through an AI agent** (Claude): installs as a skill so Claude drives it.
- **From other tools**: everything also prints JSON.

## What is PVI here?

PVI measures, in bits, how much a text increases the **predictability** of an AI
agent's decision — how much usable information the text carries about which label
the agent picks — relative to a reference prior:

```
PVI = -log2 p_baseline(y) + log2 p_text(y) = log2[ p_text(y) / p_baseline(y) ]
```

- `PVI > 0`  the text makes the decision **more predictable** (it informs the agent).
- `PVI ~ 0`  the text carries no usable information about the decision.
- `PVI < 0`  the model predicts the decision **better by ignoring** the text.

The paper computes PVI at the agent's *observed* decision, to measure
predictability. This tool fixes `y = target_label`, so in practice a higher PVI
means the text makes the agent **more likely to (predictably) choose that target**,
and `PVI < 0` means it favors the other option — a deliberate repurposing of the
same quantity for optimization.

### The reference point (`--baseline`)

The baseline `p_baseline(y)` is what PVI is measured *against*, and you can pick it:

- **`neutral` (default)** — a uniform prior over the labels (`1/K` each). PVI is then
  **"bits above chance"**: `0` is uninformative, up to `+log2(K)` means the text fully
  decides the label, negative means it points away. This is the interpretable scale,
  and it needs no API call.
- **`empty`** — the model's response to an *empty* input, `p(y|empty)`. This is the
  analogue of the paper's `H(Y|empty)`, but see the disclaimer below: a zero-shot
  prompted model answers an empty input the way a rational agent would (often near-
  certain), **not** with the dataset's label marginal, so it inflates every score.
  Opt in with `--baseline empty`; it is computed once per task+model and cached in
  `~/.config/mecha-nudge/cache/`.

The paper computes this with two fine-tuned classifiers. This tool makes one
pragmatic assumption: **a general instruction-tuned model, prompted zero-shot, is
a good enough stand-in for that classifier on an arbitrary task.** That gives a
training-free *pseudo-PVI* that runs entirely through an API / CLI.

**Single-shot.** Scoring any text is one model call (the `empty` baseline, if used,
is cached after the first run).

## Scorer

P(label) comes from the **OpenAI API**: true token logprobs with the labels
constrained via `logit_bias` (a label-masking trick), giving a smooth,
faithful signal. Set the model with `--model <name>` or `$MECHA_NUDGE_MODEL` (no default);
it must support `logprobs` + `logit_bias` on the Chat Completions API.


## Repository layout

```
pyproject.toml            # pip-installable: gives you the `mecha-nudge` command
skills/mecha-nudge/       # the self-contained skill
  ├── SKILL.md            #   Claude Code skill manifest
  ├── AGENTS.md           #   full agent-facing reference
  ├── mecha_nudge.py      #   the CLI / importable module
  ├── requirements.txt
  └── examples/
INSTALL.md                # paste-the-link agent recipe
```

## Install

Pick the path that matches how you'll use it (or see [`INSTALL.md`](INSTALL.md),
which an AI agent can follow from just the repo link).

### A. In code / the terminal (the `mecha-nudge` command)

```bash
pip install "git+https://github.com/giuliofrey/mecha-nudges-skill"   # or: pip install . from a clone
mecha-nudge --help
```

### B. In Claude Code as a skill (drop-in)

```bash
git clone https://github.com/giuliofrey/mecha-nudges-skill /tmp/mecha-nudge
cp -R /tmp/mecha-nudge/skills/mecha-nudge ~/.claude/skills/mecha-nudge   # auto-discovered
```

> The examples below use the `mecha-nudge` command (path A). Without a pip install, run the
> equivalent `python skills/mecha-nudge/mecha_nudge.py …` from a clone.

## Configure the OpenAI key

Provide your key in **any** of these ways, checked in this order:
`--api-key` > `OPENAI_API_KEY` env var > `./.env` > `~/.config/mecha-nudge/.env`.

```bash
# 1. environment variable
export OPENAI_API_KEY=sk-...

# 2. persistent .env (gitignored, never hits shell history), recommended
mkdir -p ~/.config/mecha-nudge && cp .env.example ~/.config/mecha-nudge/.env   # then paste your key
#   (a ./.env in your working dir also works and takes precedence)

# 3. per-command flag (visible in shell history / `ps`), least safe
mecha-nudge --api-key sk-... --task task.json score --text "..."
```

## Define a task (no JSON editing required)

```bash
mecha-nudge init          # asks for the decision, options, and target; writes task.json
```

Or write it by hand (see `skills/mecha-nudge/examples/task.json`):

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
piped or used by an agent you get JSON (override with `--format`). Set a scorer
model first with `--model <name>` or `export MECHA_NUDGE_MODEL=<name>`.

```bash
# Score one text
mecha-nudge --task task.json score --text "Handmade ceramic mug, 12oz, lead-free glaze, dishwasher safe."

# Score a dataset -> mean PVI (pseudo V-information)
mecha-nudge --task task.json score --data skills/mecha-nudge/examples/data.jsonl --text-field text

# Which words carry the information?
mecha-nudge --task task.json attribute --text "..."

# Rewrite to raise PVI toward target_label
mecha-nudge --task task.json optimize --text "..." --rounds 3 --candidates 5

# Score with a specific model (or set MECHA_NUDGE_MODEL once)
mecha-nudge --task task.json --model <model-name> score --text "..."

# Use the empty-input baseline instead of the neutral default (see disclaimer)
mecha-nudge --task task.json --baseline empty score --text "..."
```

Flags: `--baseline {neutral,empty}`, `--model`, `--gen-model` (optimize rewriter),
`--rounds`, `--candidates`, `--format {auto,json,human}`, `--no-cache`.

## How `optimize` works

A loop: attribute which spans help/hurt -> ask `--gen-model` for N faithful
rewrites primed with those insights -> score each with PVI -> keep the best ->
repeat until a round stops improving. Returns before/after PVI, the gain, and the
full trajectory. If gains stall, raise `--candidates`, add `--rounds`, or use a
stronger `--gen-model`.

## Caveats

This tool is a **proxy**, not a reproduction of *Mecha-nudges for Machines*. The
numbers it prints are useful for *relative* comparison and *direction*, not as the
paper's exact V-information. Concretely:

- **No fine-tuning.** The paper estimates PVI with **two classifiers fine-tuned on
  the task**: `g'` predicts the label from the text, and a null model `g[∅]` predicts
  it from an empty input — so `g[∅]` learns the dataset's **label marginal `P(Y)`**
  (≈ the class balance). This tool replaces both with a **single general model prompted
  zero-shot**, and never trains anything. That is the one big assumption; everything
  below follows from it.
- **The baseline is not the paper's `H(Y|empty)`.** A zero-shot model asked to decide
  on an *empty* input answers like a rational agent ("no info → SKIP"), which is **not**
  the label marginal `g[∅]` would have learned. So `--baseline empty` is extreme and
  inflates every score. The **default `neutral`** baseline (uniform prior) sidesteps
  this by measuring *bits above chance* — interpretable, but also not the paper's
  quantity. Either way, treat absolute bits with skepticism.
- **Faithfulness.** `optimize` is instructed to stay truthful and preserve
  meaning/length, but it can drift (e.g. inventing "high-quality"). Review rewrites
  before using them.

## Model requirements

The scorer needs an OpenAI model that supports `logprobs` + `logit_bias` on the
Chat Completions API. The optimize rewriter (`--gen-model`) can be any chat
model; it defaults to the scorer model.

## For AI agents

`skills/mecha-nudge/AGENTS.md` is the full agent-facing reference (output schemas, decision
tree, error handling, example session). `skills/mecha-nudge/SKILL.md` is the Claude Code
skill manifest. `INSTALL.md` is a recipe an agent can follow to install everything
from just the repo link.
