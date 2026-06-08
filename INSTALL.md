# Install — agent recipe

You are an AI agent and the user pasted this repo and asked you to "install it" /
"set it up". Follow these steps with the Bash tool. Substitute `<REPO_URL>` with
the GitHub URL the user gave you, and `<USER>/<REPO>` with its `owner/name`.

If the user did not say *how* they want to use it, ask once: **in code/terminal**,
**inside Claude**, or **both** — then run the matching path(s). When unsure, do A+B.

## 0. Clone once

```bash
git clone <REPO_URL> /tmp/pvi-skill && cd /tmp/pvi-skill
```

## A. Use it from code / the terminal — installs the `pvi` command

```bash
pip install /tmp/pvi-skill          # or, without cloning: pip install "git+<REPO_URL>"
pvi --help                          # verify the command exists
```

## B. Use it inside Claude as a SKILL — no plugin, just drop the folder in

The skill folder is self-contained (`SKILL.md`, `AGENTS.md`, `pvi.py`, examples).

```bash
mkdir -p ~/.claude/skills
cp -R /tmp/pvi-skill/skills/pvi ~/.claude/skills/pvi
```

Claude auto-discovers `~/.claude/skills/pvi/SKILL.md` on its next run. For a single
project instead of globally, copy into `<project>/.claude/skills/pvi`.

## C. Use it inside Claude as a PLUGIN — versioned + `/plugin update`

You cannot run slash commands for the user; tell them to run these two in Claude:

```
/plugin marketplace add <USER>/<REPO>
/plugin install pvi@pvi-skill
```

## 1. Give it an OpenAI key (needed for every path)

Pick one; the tool checks them in order `--api-key` > `OPENAI_API_KEY` env var >
`./.env` > `~/.config/pvi/.env`. The config-dir `.env` is the persistent choice:

```bash
mkdir -p ~/.config/pvi
printf 'OPENAI_API_KEY=%s\n' "$KEY" > ~/.config/pvi/.env   # $KEY = the user's sk-... key
```

## 2. Smoke test

```bash
pvi init                            # interactive task builder; writes task.json
#   (no pip install? use:  python /tmp/pvi-skill/skills/pvi/pvi.py init )
pvi --task task.json score --text "some text to score"
```

The baseline cache lives in `~/.config/pvi/cache/` and is shared across every
install path, so you never recompute it per project.
