# pvi - thorough guide for agents

You (an AI agent, usually Claude) are driving this CLI on behalf of a user who
may not code. Your job: turn their goal into a task, run the right command,
read the JSON, and explain it in plain language with the caveats. This file is
the deep reference; `SKILL.md` is the short trigger.

## Mental model

PVI = "how many bits does this text remove from an AI agent's uncertainty about
a decision." Formally `PVI = H(Y|empty) - H(Y|text)`, in bits:

- `H(Y|text)` = uncertainty about the decision after reading the text.
- `H(Y|empty)` = uncertainty with no input (the model's prior). Constant per
  task, so computed once and cached.
- `PVI > 0`: text informs the decision. `~0`: uninformative. `< 0`: text points
  the agent *away* from the target decision.

The paper uses two fine-tuned classifiers. This tool substitutes a general
prompted model, so the numbers are a **pseudo-PVI proxy**, not the paper's exact
values. Say so when it matters.

## Scorer

P(label) comes from the **OpenAI API**: true token logprobs with the labels
constrained via `logit_bias` (the paper's masking trick), so the distribution is
smooth and fine-grained - exactly what `optimize` needs to detect small gains.
Needs `OPENAI_API_KEY` + `pip install openai tiktoken`.

- Set the scorer with `--model <name>` or `$PVI_MODEL` (no default). It **must**
  support `logprobs` + `logit_bias` on the Chat Completions API.
- **Absolute PVI is not comparable across models.** Compare only within one
  `--model`. To sanity-check a finding, re-run with a different `--model` and look
  at the *direction/ranking*, not the bits.

## Setup checks (do once)

1. `pip install -r requirements.txt` for the deps (or `pip install pvi-skill` /
   `pip install "git+<repo-url>"` to also get a `pvi` command on PATH).
2. **How you invoke the CLI** — `pvi` below means whichever applies:
   - pip-installed: `pvi …`
   - Claude plugin: `python "$CLAUDE_PLUGIN_ROOT/skills/pvi/pvi.py" …`
   - plain skill / clone: `python pvi.py …` from the skill folder.
3. Supply an OpenAI key (precedence: `--api-key` > `OPENAI_API_KEY` env var >
   `./.env` > `~/.config/pvi/.env`). If none is set, either ask the user to run
   `! export OPENAI_API_KEY=...`, drop `OPENAI_API_KEY=sk-...` into
   `~/.config/pvi/.env` (gitignored; safest - no shell history), or pass
   `--api-key sk-...` per command (warn it shows up in shell history / `ps`).
4. Set a scorer model with `--model <name>` or `PVI_MODEL` — there is **no
   default**, and it must support logprobs + logit_bias. The user tells you which.
5. Commands run from any directory — the baseline cache lives in
   `~/.config/pvi/cache/` (override the location with `PVI_HOME`).

## Step 1 - always define the task first

A task is the decision the agent makes. Either write the JSON yourself from the
user's description, or run the wizard:

```
pvi init                      # interactive; writes task.json
pvi --task task.json init     # write to a specific path
```

Task fields:
- `name` (string): short slug; namespaces the baseline cache.
- `question` (string): the decision phrased to the agent.
- `labels` (list): the choices. **Rules that matter:** short, single words,
  and with *distinct first tokens*. Good: `BUY`/`SKIP`, `YES`/`NO`,
  `RELEVANT`/`IRRELEVANT`. Bad: `RELEVANT`/`REJECT` (both start with `RE` -> the
  scorer prints a collision warning and scores between them are unreliable).
- `target_label` (string, optional): the choice to measure/optimize toward.
  Required for `optimize`. For `score`, omitting it means "score toward whatever
  the model itself picks" (descriptive); setting it means "how much does the text
  push toward this specific decision."

## Step 2 - pick the command

| User goal | Command |
|---|---|
| How informative is this one text? | `score --text "..."` |
| Average over a dataset (V-information) | `score --data file.jsonl --text-field text` |
| Which words/sentences carry the info? | `attribute --text "..." [--granularity word\|sentence]` |
| Make the text more persuasive to the agent | `optimize --text "..." [--rounds N --candidates K --gen-model M]` |
| Just see the prior baseline | `baseline` |

Global flags (put them **before** the subcommand): `--model`, `--task`,
`--format {auto,json,human}`, `--no-cache`.

`--format auto` (default) prints human text in a terminal and JSON when piped or
captured by an agent. **When you run it, you get JSON** - parse that. If you want
the pretty version to show the user verbatim, pass `--format human`.

## Step 3 - interpret the JSON

`score` (single):
```json
{"label":"BUY","pvi":0.68,"H_yb":1.0,"H_yx":0.32,
 "p_target_given_text":0.8,"p_target_baseline":0.5,
 "distribution":{"BUY":0.8,"SKIP":0.2}}
```
Report `pvi` in bits, the decision, and what it means. `pvi>0.5` strong, `0.05-0.5`
mild, `~0` none, `<0` points away.

`score` (dataset): `{"n":..,"v_information":..,"mean_H_yb":..,"mean_H_yx":..,"per_record":[...]}`.
`v_information` is the mean PVI = the paper's headline quantity.

`attribute`: `{"pvi":..,"label":..,"spans":[{"span":"lead-free","delta":0.5,"pvi_without":0.18}, ...]}`.
`delta = PVI_full - PVI_without`. Positive = the span *adds* information; negative
= it *dilutes*. Sorted high-to-low. Surface the top helpers and the diluters.

`optimize`: `{"original_pvi":..,"best_pvi":..,"gain":..,"best_text":"...","trajectory":[...]}`.
Show original->best and the gain, present `best_text`, and remind the user to
check faithfulness. If `gain` is 0, see the next section.

## The optimize loop (internals & expectations)

Each round: attribute (word level) -> ask `--gen-model` for `candidates` faithful
rewrites primed with the helping/diluting words -> score each -> keep the best ->
stop when a round yields no improvement.

If it doesn't improve:
- Weak rewriter: switch `--gen-model` to a stronger model.
- Too few tries: raise `--candidates` (8-12) and `--rounds` (4-5).
- Genuinely saturated: the text may already be near the agent's ceiling - say so.

## Caveats to relay to the user (every substantive run)

1. **One-proxy Goodhart risk.** The score reflects one model. High pseudo-PVI
   need not transfer to other agents. Offer to re-score with another `--model`.
2. **Check faithfulness.** `optimize` is told to stay truthful and keep meaning,
   but can drift; the user must review rewrites.
3. **Pseudo, not exact.** This is a zero-shot proxy for the paper's fine-tuned
   V-information.

## Errors you may hit

- `No OpenAI key found. Pass --api-key sk-..., set OPENAI_API_KEY, or put ... in a
  .env file` - no key supplied by any of the three methods.
- `Install deps: pip install openai tiktoken` - the SDK isn't installed.
- `Warning: labels 'X' and 'Y' start with the same token` (stderr) - fix the task
  labels to have distinct first words; results between them are unreliable.
- A model without `logprobs`/`logit_bias` support will error from the OpenAI API;
  switch `--model` to one that supports both.

## Cost & latency

- `score` one text = 1 model call (baseline is cached after the first task run).
- `score --data` = 1 call per record (parallelized by `--workers`).
- `attribute` = ~1 call per unique span (word granularity can be many; offer
  `--granularity sentence` for long text).
- `optimize` = roughly `rounds x (#words + candidates)` calls. Keep `rounds`/
  `candidates` small unless the user wants a thorough pass.

## Example agent turn

User: "Does my Etsy mug listing actually help an AI shopping agent pick it?"

1. Draft task (BUY/SKIP, target BUY) -> confirm with user, write `task.json`.
2. `pvi --task task.json --format json score --text "<their listing>"`.
3. Read JSON: pvi +0.68 -> "Yes - it adds ~0.68 bits toward BUY; the agent's
   confidence rises from 50% to 80%."
4. Offer: `attribute` to show which words do the work, then `optimize` to push it
   higher - with the faithfulness and proxy caveats.
