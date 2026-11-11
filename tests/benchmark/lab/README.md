# Replays of recorded benchmark runs (`policy_lab.py`)

A replay feeds the Tasks of a finished run, in a shuffled order per seed, to the plugin's own engine and looks up the
recorded answer of the model the engine chose (from the run's no-plugin arms A and B). No model is called. A replay
is not a live run: it reuses one recorded answer per Task and model.

| File | What | Plugin state |
| :--- | :--- | :--- |
| `two-phase-v1-grid-1.json`, `everyday-v1-grid-2.json` | Every turn marked from the answer key; a grid of chooser settings | as of November 10, afternoon |
| `*-C-before.json`, `*-C-after.json` | Every turn marked; default and `explore_share: 0.3` | before and after the November 10 change below |
| `*-D-before.json`, `*-D-after.json` | Nobody marks; the user's own model's PASS or FAIL drawn at the rates measured in the live D arms; `check_share` 1.0 and 0.3 | before and after the change |

The change (November 10, evening): a PASS from the user's own model and a thank-you count as weak evidence; a cheaper
model takes a kind of Task only on strong outcomes; trial turns wait until the user has marked one of their recent
turns. The "before" files ran the plugin as it was just before that change ("../continuum" at that time, or a copy of
it); the "after" files ran the changed plugin ("scratch-copy/..." was a copy of it, identical in engine and config).

Grid files list the settings; each result file has the seeds, the phases and the per-kind table.

To run the tool from the plugin package: `python3 tests/benchmark/policy_lab.py --run RUN --items items/FILE
--spec bench_spec_everyday.json --grid lab/grid-D.json --feedback check --plugin . --out lab/x.json` from a folder
that holds `runs/RUN/A-own-model/` and `runs/RUN/B-small-model/` (each with `records.jsonl` and `meter.jsonl`); the
package keeps those two files of each arm under `tests/benchmark/RUN/`, so copy or link them as `runs/RUN/`.
