# pvi: measure & optimize text for AI agents

A tiny tool to score and improve how much a piece of text informs an AI agent's
decision, using **Pointwise V-Information (PVI)**. Companion to
[*Mecha-nudges for Machines*](https://giuliofrey.eu/mecha-nudges/) (Frey &
Ethayarajh, 2026).

Use it three ways:
- **Yourself, from the terminal**: readable output, an interactive task wizard.
- **Through an AI agent** (Claude): installs as a skill/plugin so Claude drives it.
- **From other tools**: everything also prints JSON.

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
cached (in `~/.config/pvi/cache/`). After that, scoring any text is one model call.

## Scorer

P(label) comes from the **OpenAI API**: true token logprobs with the labels
constrained via `logit_bias` (the paper's masking trick), giving a smooth,
faithful signal. Set the model with `--model <name>` or `$PVI_MODEL` (no default);
it must support `logprobs` + `logit_bias` on the Chat Completions API.

Absolute PVI is **not** comparable across models; compare within one `--model`.

## Repository layout

```
pyproject.toml            # pip-installable: gives you the `pvi` command
.claude-plugin/           # marketplace.json + plugin.json (Claude Code plugin)
skills/pvi/               # the self-contained skill
  ├── SKILL.md            #   Claude Code skill manifest
  ├── AGENTS.md           #   full agent-facing reference
  ├── pvi.py              #   the CLI / importable module
  ├── requirements.txt
  └── examples/
INSTALL.md                # paste-the-link agent recipe
```

## Install

Pick the path that matches how you'll use it (or see [`INSTALL.md`](INSTALL.md),
which an AI agent can follow from just the repo link).

### A. In code / the terminal (the `pvi` command)

```bash
pip install "git+https://github.com/<USER>/<REPO>"   # or: pip install . from a clone
pvi --help
```

### B. In Claude Code as a plugin (versioned, `/plugin update`)

```
/plugin marketplace add <USER>/<REPO>
/plugin install pvi@pvi-skill
```

### C. In Claude Code as a plain skill (no plugin)

```bash
git clone https://github.com/<USER>/<REPO> /tmp/pvi-skill
cp -R /tmp/pvi-skill/skills/pvi ~/.claude/skills/pvi   # auto-discovered
```

> The examples below use the `pvi` command (path A). Without a pip install, run the
> equivalent `python skills/pvi/pvi.py …` from a clone.

## Configure the OpenAI key

Provide your key in **any** of these ways, checked in this order:
`--api-key` > `OPENAI_API_KEY` env var > `./.env` > `~/.config/pvi/.env`.

```bash
# 1. environment variable
export OPENAI_API_KEY=sk-...

# 2. persistent .env (gitignored, never hits shell history), recommended
mkdir -p ~/.config/pvi && cp .env.example ~/.config/pvi/.env   # then paste your key
#   (a ./.env in your working dir also works and takes precedence)

# 3. per-command flag (visible in shell history / `ps`), least safe
pvi --api-key sk-... --task task.json score --text "..."
```

## Define a task (no JSON editing required)

```bash
pvi init          # asks for the decision, options, and target; writes task.json
```

Or write it by hand (see `skills/pvi/examples/task.json`):

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
model first with `--model <name>` or `export PVI_MODEL=<name>`.

```bash
# Score one text
pvi --task task.json score --text "Handmade ceramic mug, 12oz, lead-free glaze, dishwasher safe."

# Score a dataset -> mean PVI (pseudo V-information)
pvi --task task.json score --data skills/pvi/examples/data.jsonl --text-field text

# Which words carry the information?
pvi --task task.json attribute --text "..."

# Rewrite to raise PVI toward target_label
pvi --task task.json optimize --text "..." --rounds 3 --candidates 5

# Score with a specific model (or set PVI_MODEL once)
pvi --task task.json --model <model-name> score --text "..."
```

Flags: `--model`, `--gen-model` (optimize rewriter), `--rounds`, `--candidates`,
`--format {auto,json,human}`, `--no-cache`.

## How `optimize` works

A loop: attribute which spans help/hurt -> ask `--gen-model` for N faithful
rewrites primed with those insights -> score each with PVI -> keep the best ->
repeat until a round stops improving. Returns before/after PVI, the gain, and the
full trajectory. If gains stall, raise `--candidates`, add `--rounds`, or use a
stronger `--gen-model`.

## Caveats (please read)

- **One-proxy Goodhart risk.** Scores reflect the single `--model` you choose.
  Re-score with another `--model` to sanity-check direction.
- **Faithfulness.** `optimize` is instructed to stay truthful and preserve
  meaning/length, but it can drift. Review rewrites before using them.
- **Pseudo, not exact.** A zero-shot proxy for the paper's fine-tuned
  V-information, not a reproduction of it.

## Model requirements

The scorer needs an OpenAI model that supports `logprobs` + `logit_bias` on the
Chat Completions API. The optimize rewriter (`--gen-model`) can be any chat
model; it defaults to the scorer model.

## For AI agents

`skills/pvi/AGENTS.md` is the full agent-facing reference (output schemas, decision
tree, error handling, example session). `skills/pvi/SKILL.md` is the Claude Code
skill manifest. `INSTALL.md` is a recipe an agent can follow to install everything
from just the repo link.
