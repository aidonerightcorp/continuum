# Continuum in Hermes: run `two-phase-v1`

## Conditions

- Tasks: 480 public checkable Tasks in one fixed mixed order, each one a new Hermes session. Kinds: arithmetic-expression (80), date-choice (80), object-counting (80), tracking-choice (80), word-problem-multi-step (80), word-problem-one-step (80).
- Learning phase (Tasks 1 to 240): 240 public Tasks. Continuum starts with an empty record file.
- Test phase (Tasks 241 to 480): 240 other public Tasks of the same six kinds. Continuum keeps the records of the learning phase. This phase carries the headline figures.
- Hermes: its own agent loop (`AIAgent`, as in Hermes's batch runner), Hermes's own system prompt, toolsets: none (see the limits below), at most 6 model calls a Task.
- Models: the user's own model is `coder-7b-bench`, the small model is `meta-3b-bench`; both are open-weight models on one 12 GB graphics card on one machine, with a real context window of 16384 tokens and temperature 0.2. Largest prompt in the run: 2020 tokens.
- Every Task ends with the same request in every arm: "Work through it step by step. Then end your reply with one last line in this form: Answer: <the number, or the letter of the option>".
- Scoring: the last `Answer:` line of the reply against the answer key of the public set.
- Cost: seconds of model time on that card and tokens, from Hermes's own per-call hooks, the same in every arm. Checks by the user's own model are counted in the arm that makes them.
- One pass, one order, one seed. Intervals: 95%, from resampling the Tasks.

## Arms

- **A-own-model**: Hermes with the user's own (larger) model for every Task. Continuum is not loaded.
- **B-small-model**: Hermes with the small model for every Task. Continuum is not loaded.
- **C-marked**: Continuum on, default settings. After every turn the harness marks the answer good or bad with the plugin's own /continuum command, from the answer key. This stands for a user who marks every turn.
- **D-self-checked**: Continuum on, nobody marks anything. The user's own model checks every turn that the small model answered (one PASS or FAIL question). The answer key is used only to score the run afterwards.
- **E-marked-fast**: As C, with a larger share of trial turns (explore_share 0.3), so the store fills faster.

## Learning phase

| Arm | Right | Answered by the small model | Model seconds | Tokens on the user's model | Tokens on the small model | Checks |
|---|---|---|---|---|---|---|
| A-own-model | 64.6% (58.3% to 70.4%) | 0.0% | 609 | 191,238 | 0 | 0 |
| B-small-model | 28.3% (22.5% to 34.2%) | 100.0% | 1199 | 0 | 208,241 | 0 |
| C-marked | 59.2% (53.3% to 65.4%) | 12.9% | 588 | 163,719 | 26,620 | 0 |
| D-self-checked | 57.1% (50.8% to 63.7%) | 17.9% | 701 | 197,433 | 36,584 | 44 |
| E-marked-fast | 61.3% (55.0% to 67.5%) | 12.9% | 813 | 165,265 | 29,802 | 0 |

Against **A-own-model** on the same Tasks:

| Arm | Right, difference in points | Model seconds | Tokens on the user's model | Price at 0.2 to 1 (an assumed price ratio) |
|---|---|---|---|---|
| B-small-model | -36.2 (-43.8 to -29.2) | 197% of A-own-model | 0% of A-own-model | 22% of A-own-model |
| C-marked | -5.4 (-12.9 to +2.1) | 97% of A-own-model | 86% of A-own-model | 88% of A-own-model |
| D-self-checked | -7.5 (-14.6 to -0.4) | 115% of A-own-model | 103% of A-own-model | 107% of A-own-model |
| E-marked-fast | -3.3 (-10.4 to +3.3) | 134% of A-own-model | 86% of A-own-model | 90% of A-own-model |

## Test phase

| Arm | Right | Answered by the small model | Model seconds | Tokens on the user's model | Tokens on the small model | Checks |
|---|---|---|---|---|---|---|
| A-own-model | 61.3% (55.0% to 67.5%) | 0.0% | 628 | 190,280 | 0 | 0 |
| B-small-model | 27.5% (22.1% to 33.3%) | 100.0% | 1195 | 0 | 209,736 | 0 |
| C-marked | 55.4% (49.2% to 61.7%) | 2.1% | 605 | 185,713 | 4,596 | 0 |
| D-self-checked | 47.1% (40.8% to 53.8%) | 42.1% | 797 | 207,903 | 83,474 | 107 |
| E-marked-fast | 60.4% (54.6% to 66.7%) | 1.7% | 591 | 185,841 | 3,435 | 0 |

Against **A-own-model** on the same Tasks:

| Arm | Right, difference in points | Model seconds | Tokens on the user's model | Price at 0.2 to 1 (an assumed price ratio) |
|---|---|---|---|---|
| B-small-model | -33.8 (-41.2 to -25.8) | 190% of A-own-model | 0% of A-own-model | 22% of A-own-model |
| C-marked | -5.8 (-13.3 to +1.7) | 96% of A-own-model | 98% of A-own-model | 98% of A-own-model |
| D-self-checked | -14.2 (-21.7 to -6.7) | 127% of A-own-model | 109% of A-own-model | 118% of A-own-model |
| E-marked-fast | -0.8 (-7.5 to +5.8) | 94% of A-own-model | 98% of A-own-model | 98% of A-own-model |

## By kind of Task (Test phase)

| Arm | arithmetic-expression | date-choice | object-counting | tracking-choice | word-problem-multi-step | word-problem-one-step |
|---|---|---|---|---|---|---|
| A-own-model | 42.5%, small 0.0% | 50.0%, small 0.0% | 62.5%, small 0.0% | 50.0%, small 0.0% | 80.0%, small 0.0% | 82.5%, small 0.0% |
| B-small-model | 7.5%, small 100.0% | 27.5%, small 100.0% | 10.0%, small 100.0% | 17.5%, small 100.0% | 57.5%, small 100.0% | 45.0%, small 100.0% |
| C-marked | 50.0%, small 0.0% | 35.0%, small 10.0% | 55.0%, small 0.0% | 35.0%, small 0.0% | 75.0%, small 2.5% | 82.5%, small 0.0% |
| D-self-checked | 17.5%, small 77.5% | 30.0%, small 67.5% | 40.0%, small 47.5% | 40.0%, small 12.5% | 75.0%, small 22.5% | 80.0%, small 25.0% |
| E-marked-fast | 40.0%, small 0.0% | 42.5%, small 5.0% | 55.0%, small 0.0% | 52.5%, small 2.5% | 87.5%, small 2.5% | 85.0%, small 0.0% |

## What Continuum did

- **C-marked**: choices reference 444, explore 36; outcome sources explicit 480; strictness now preset "balanced".
- **D-self-checked**: choices reference 329, exploit 103, explore 41, reference-check 7; outcome sources session_end 329, reference_check 151; strictness now preset "shape-only".
  - Checks by the user's own model: 151. They agreed with the answer key on 65 (43.0%). Of the 108 wrong answers checked, the check passed 86.
- **E-marked-fast**: choices reference 445, explore 35; outcome sources explicit 480; strictness now preset "sketch-first".

## Faults

Tasks that ended with a harness error (counted as wrong): A-own-model 0, B-small-model 0, C-marked 0, D-self-checked 0, E-marked-fast 0.

## Limits

- One pass in one order with one seed on one machine. A second pass can differ by a few points; read the intervals.
- No tools. Hermes refuses tool use on a local endpoint with a window under 64,000 tokens, and this card cannot hold both models at that size. The Hermes config declares 64000 tokens; the models really serve 16384. No prompt in the run came near that.
- Two small open-weight models (4-bit). The gap between a user's own model and a cheaper one is different on other models, and so is the share of Tasks that can move.
- Every Task is a new session with one turn. Outcome signals that need a second message (a thank-you, a correction) do not occur here.
- In the marked arms the harness marks every answer from the answer key. That stands for a user who marks every turn; few users do. The self-checked arm uses no answer key while it runs.
- The price column uses an assumed price ratio of 0.2 to 1 between the two models. The measured units are seconds and tokens.
- Seconds are model seconds on a shared 12 GB card, where the two models take turns in memory. A model switch costs load time here that a hosted provider does not have.
- Plugin code: a frozen copy taken 2026-11-10 14:40:14 EST (42 files, mark `b9cb91cb`, 201 passed; full record in `results.json`). Later changes to the plugin are not in this run.
