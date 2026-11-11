# Continuum on/off benchmark inside Hermes

This folder holds the harness that measures Continuum inside real Hermes sessions. It answers one question: when
Continuum is on, how much work moves to a cheaper model, and what happens to the share of right answers?

The harness is test code. It drives Hermes from outside (it imports Hermes's agent class, which is not part of
the public plugin API, so a new Hermes release can break the harness without touching the plugin). Hermes never
loads this folder. It is not a simulation of Hermes. Each Task is one new Hermes session: the harness builds Hermes's own agent
(`AIAgent`, the class Hermes's batch runner uses), sends the Task as the user message and reads the reply. When an
arm loads Continuum, Hermes finds and loads the plugin from that arm's own Hermes home, as it does for a user.

## What is in this folder

| File | Purpose |
| :--- | :--- |
| `bench_hermes.py` | Runs one arm: one Hermes session for each Task, in the order of the items file |
| `bench_spec.json` | The settings of the benchmark: endpoint, the two models, context size, tools, the arms |
| `bench_meter/` | A small Hermes plugin, loaded in every arm, that writes one line for each model call (model, tokens, seconds; no text) |
| `analyze_bench.py` | Reads a run and writes `RESULTS.md` and `results.json` |
| `items/hermes-v1.jsonl` | The 240 Tasks with their answer key |
| `items/NOTICE.md` | Sources and licences of the Tasks |

## The arms

| Arm | Continuum | What it stands for |
| :--- | :--- | :--- |
| `A-own-model` | not loaded | Hermes with your own (larger) model for every Task. The base line. |
| `B-small-model` | not loaded | Hermes with the cheaper model for every Task. Shows what you lose if you only switch models. |
| `C-marked` | on, default settings | A user who marks every answer. After each turn the harness runs the plugin's own `/continuum good` or `/continuum bad` from the answer key. |
| `D-self-checked` | on, checks on | Nobody marks anything. Your own model checks each turn that the cheaper model answered (one PASS or FAIL question). The answer key is used only to score the run afterwards. |
| `E-marked-fast` | on, `explore_share` 0.3 | As `C-marked` with more trial turns, so the records fill faster. |

Arms C, D and E start with an empty record file. Continuum learns during the run, so the order of the Tasks matters
and is fixed: the order of the items file.

## How to run it

You need a checkout of Hermes Agent (`>=0.23.0`), a Python that can run it, and an OpenAI-compatible endpoint that
serves two models: your own model and a cheaper one. Put the endpoint and the two model ids in `bench_spec.json`, or
pass them on the command line.

```text
export HERMES_AGENT_DIR=/path/to/hermes-agent

python bench_hermes.py --run my-run --arm A-own-model   --items items/hermes-v1.jsonl
python bench_hermes.py --run my-run --arm B-small-model --items items/hermes-v1.jsonl
python bench_hermes.py --run my-run --arm C-marked      --items items/hermes-v1.jsonl
python bench_hermes.py --run my-run --arm D-self-checked --items items/hermes-v1.jsonl

python analyze_bench.py --run my-run
```

Options of `bench_hermes.py`: `--limit N` (the first N Tasks only, for a trial), `--spec FILE`, `--endpoint URL`,
`--reference MODEL`, `--small MODEL`. A run can be stopped and started again; Tasks that already have a record are
skipped.

Each arm writes into `runs/<run>/<arm>/`:

- `home/`: the arm's own Hermes home (its `config.yaml` and the plugin's data folder). Your own Hermes home is not
  used.
- `records.jsonl`: one line for each Task: the reply, the parsed answer, right or wrong, seconds.
- `meter.jsonl`: one line for each model call: model, token counts, seconds.
- `spec.json`: the settings the arm ran with.

## What is counted, and how

- **Right answers.** Each Task asks for a number or for the letter of an option. The prompt asks for a last line
  `Answer: <...>`. The harness reads that line (or the end of the reply when the line is missing) and compares it
  with the answer key. A reply with no readable answer counts as wrong.
- **Model calls, tokens and seconds.** These come from `bench_meter`, which uses Hermes's own hooks. It is loaded in
  every arm, so arms with and without Continuum are measured in the same way. Check questions that Continuum sends
  to your own model are counted as calls of your own model.
- **Share on the cheaper model.** A Task counts as moved when the cheaper model answered it and the turn did not
  return to your own model.
- **Cost.** `analyze_bench.py` reports tokens for each model. The one-number cost line weights cheaper-model tokens
  by `--price-ratio` (default 0.2). This ratio is an assumption that you set; it is not a measured price.
- **Differences.** Each arm is compared with `A-own-model` Task by Task (paired), with a 95% interval from 2,000
  resamples of the Tasks.
- **The check against the answer key.** For `D-self-checked`, the analysis also reports how often the PASS or FAIL
  verdict of your own model agreed with the answer key.

## Conditions to keep in mind

- The Tasks are short questions with one right answer (arithmetic, counting, dates, object tracking, word
  problems). They show how the loop behaves when outcomes can be scored. They are not a sample of everyday agent
  work, and a result here does not predict the saving on your own sessions.
- Only the `todo` toolset is on, so no model-written code and no terminal command runs during the benchmark.
- A result belongs to the two models, the endpoint and the settings in the run's `spec.json`. Other models give
  other numbers.
- Continuum starts from an empty record file in each arm. The first Tasks of every kind go to your own model, so the
  share moved in 240 Tasks is lower than in long use.
- The harness makes no network call of its own. Hermes calls only the endpoint in the spec.

## The Tasks

`items/hermes-v1.jsonl` holds 240 Tasks, 40 of each kind, drawn from public sets. See `items/NOTICE.md` for the
sources and their licences.

| Kind | Source |
| :--- | :--- |
| `arithmetic-expression` | BIG-Bench Hard, task `multistep_arithmetic_two` |
| `object-counting` | BIG-Bench Hard, task `object_counting` |
| `date-choice` | BIG-Bench Hard, task `date_understanding` |
| `tracking-choice` | BIG-Bench Hard, task `tracking_shuffled_objects_three_objects` |
| `word-problem-multi-step` | GSM8K, test split |
| `word-problem-one-step` | SVAMP |
