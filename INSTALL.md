# Install — agent recipe

You are an AI agent and the user wants this repo installed / set up. Follow these
steps with the Bash tool.

If the user did not say *how* they want to use it, ask once: **in code/terminal**,
**inside Claude**, or **both** — then run the matching path(s). When unsure, do A+B.

## 0. Clone once

```bash
git clone https://github.com/giuliofrey/mecha-nudges-skill /tmp/mecha-nudge && cd /tmp/mecha-nudge
```

## A. Use it from code / the terminal — installs the `mecha-nudge` command

```bash
pip install /tmp/mecha-nudge          # or, without cloning: pip install "git+https://github.com/giuliofrey/mecha-nudges-skill"
mecha-nudge --help                          # verify the command exists
```

## B. Use it inside Claude as a skill — drop the folder in

The skill folder is self-contained (`SKILL.md`, `AGENTS.md`, `mecha_nudge.py`, examples).

```bash
mkdir -p ~/.claude/skills
cp -R /tmp/mecha-nudge/skills/mecha-nudge ~/.claude/skills/mecha-nudge
```

Claude auto-discovers `~/.claude/skills/mecha-nudge/SKILL.md` on its next run. For a single
project instead of globally, copy into `<project>/.claude/skills/mecha-nudge`.

## 1. Give it an OpenAI key (needed for every path)

Pick one; the tool checks them in order `--api-key` > `OPENAI_API_KEY` env var >
`./.env` > `~/.config/mecha-nudge/.env`. The config-dir `.env` is the persistent choice:

```bash
mkdir -p ~/.config/mecha-nudge
printf 'OPENAI_API_KEY=%s\n' "$KEY" > ~/.config/mecha-nudge/.env   # $KEY = the user's sk-... key
```

## 2. Smoke test

```bash
mecha-nudge init                            # interactive task builder; writes task.json
#   (no pip install? use:  python /tmp/mecha-nudge/skills/mecha-nudge/mecha_nudge.py init )
mecha-nudge --task task.json score --text "some text to score"
```

The baseline cache lives in `~/.config/mecha-nudge/cache/` and is shared across every
install path, so you never recompute it per project.
