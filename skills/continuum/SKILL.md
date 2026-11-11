---
name: continuum
description: Use when the user asks why a turn went to a cheaper or larger model, wants to mark an answer good or bad, wants to keep a session on their own model, or when a Task looks like one answered before. Explains how to read Continuum's right-sizing records and how to use the continuum_recall tool.
---

# Continuum: Hermes sizes itself to each Task

Continuum is a plugin that chooses, for each turn, the cheapest of the user's own models that its local records
support. It runs before you see the turn. You do not choose the model; you help the user read what happened and you
can look up how similar Tasks were answered before.

## When to use the `continuum_recall` tool

Call `continuum_recall` with the Task in plain words when:

- the user asks for something that looks like an earlier Task ("do the same as last time", a recurring report, the
  same kind of calculation with new numbers);
- you are about to do a long piece of work and a recorded answer may save steps;
- the user asks "have we done this before?".

The tool returns the closest recorded Tasks with their answers, their recorded outcome and the tools they used. It
reads local records only, and only those of the same user on the same platform as the running turn. An empty result
means that no record can be shown: there is none, the stored text was removed, or Continuum is off or only observes.
Do not guess one.

How to use what it returns:

1. Prefer records with a good outcome. A record marked not good tells you what to avoid.
2. Check that the record really matches the new Task. Similar wording with different numbers needs a new
   calculation, not a copied answer.
3. Say that you reused an earlier answer when you do.

## Commands the user can run

Tell the user about these when they fit. They are slash commands for the user, not tools for you.

| The user wants | Command |
| :--- | :--- |
| To see what Continuum has done so far | `/continuum` |
| To know why the last turn went to that model | `/continuum why` |
| To mark the last answer good or bad | `/continuum good` or `/continuum bad` |
| To keep this session on their own model | `/continuum pin` (and `/continuum unpin`) |
| To try a cheaper model again after the provider refused it | `/continuum unblock` |
| To remove the stored Task and answer text | `/continuum forget-text` |
| To rewrite the record file after text was removed | `hermes continuum compact` (shell) |
| To check the policy on held-out records now | `/continuum audit` |
| To share what was learned without sharing text | `/continuum export` |

## What to tell the user

- A mark with `/continuum good` or `/continuum bad` is the strongest signal Continuum gets. A mark counts for or
  against the model that answered that turn, on similar Tasks. A `bad` mark on a cheaper model's answer can move
  such Tasks back to the user's own model; `/continuum why` shows what the records say now.
- Continuum moves a kind of Task to a cheaper model only on marks (or a check that a caller reports). With no
  recent mark from the user it makes no trial turns; a kind that marks already moved can stay on the cheaper model.
  A PASS from the user's own model's check and a thank-you count as weak evidence; a FAIL counts in full. If the
  user asks why nothing moves, explain this and suggest marking a few answers.
- If the user says an answer from a cheaper model was wrong, suggest `/continuum bad`, then answer the Task again
  with care.
- Continuum changes a request (the model, the tool list, an added past answer) only in `mode: auto` and only when
  the user granted `llm.model_override`. Without it Continuum only records. With `mode: off` it records nothing.
- A pinned session stays on the user's own model, unchanged, until `/continuum unpin`.
- If `/continuum` shows a blocked model, the provider refused it. The user fixes the model name or access, then runs
  `/continuum unblock`. The user also runs `/continuum unblock` once after a change to another provider.
- Stored Task and answer text is removed after 90 days by default (setting `text_days`).
- `unblock`, `audit`, `tune`, `forget-text`, `compact`, `export` and `import` act on the whole record file. They run only for
  the local user: in the Hermes terminal chat, the TUI, the desktop app, an editor session over ACP, or from the
  shell as `hermes continuum <command>`. On a chat platform they answer with a note and change nothing; tell the
  user to run the shell command on the machine that runs Hermes. On a chat platform `/continuum` shows
  the mode, model names, counts, policy values and short notes (an unread `mode` value, a missing consent, that
  Continuum is off for this run or started a new record file), and no provider name, block reason or file name.
- `forget-text` does not touch files that Continuum set aside as unreadable or the file of the experimental
  engines; it names them, and the user deletes them.
- The records stay on the user's machine. The export holds fingerprints and outcomes, no text.

## What not to do

- Do not tell the user that a cheaper model was "as good" for a Task unless the records say so; `/continuum why`
  shows the evidence.
- Do not paste a recalled answer without checking it against the new Task.
