# Continuum in Hermes: run `everyday-v1`

## Conditions

- Tasks: 480 checkable Tasks (half are four easy kinds written by `make_everyday_items.py` from a fixed seed, half are four harder kinds from public sets) in one fixed mixed order, each one a new Hermes session. Kinds: category-choice (60), date-choice (60), largest-number (60), sum-of-two (60), tracking-choice (60), unit-conversion (60), word-problem-multi-step (60), word-problem-one-step (60).
- Learning phase (Tasks 1 to 240): 240 Tasks, 30 of each of eight kinds. Continuum starts with an empty record file and is never told a kind.
- Test phase (Tasks 241 to 480): 240 other Tasks of the same eight kinds. Continuum keeps the records of the learning phase. This phase carries the headline figures.
- Hermes: its own agent loop (`AIAgent`, as in Hermes's batch runner), Hermes's own system prompt, toolsets: none (see the limits below), at most 6 model calls a Task.
- Models: the user's own model is `coder-7b-bench`, the small model is `meta-3b-bench`; both are open-weight models on one 12 GB graphics card on one machine, with a real context window of 16384 tokens and temperature 0.2. Largest prompt in the run: 2866 tokens.
- Every Task ends with the same request in every arm: "Work through it step by step. Then end your reply with one last line in this form: Answer: <the number, or the letter of the option>".
- Scoring: the last `Answer:` line of the reply against the answer key (from the generating code for the easy kinds, from the public set for the others).
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
| A-own-model | 84.2% (79.6% to 88.8%) | 0.0% | 538 | 182,883 | 0 | 0 |
| B-small-model | 45.8% (39.6% to 52.1%) | 100.0% | 948 | 0 | 199,106 | 0 |
| C-marked | 77.1% (71.7% to 82.1%) | 7.5% | 518 | 167,813 | 14,093 | 0 |
| D-self-checked | 62.1% (55.8% to 68.3%) | 52.9% | 1175 | 203,356 | 101,025 | 133 |
| E-marked-fast | 74.2% (68.8% to 79.6%) | 16.2% | 737 | 151,969 | 37,862 | 0 |

Against **A-own-model** on the same Tasks:

| Arm | Right, difference in points | Model seconds | Tokens on the user's model | Price at 0.2 to 1 (an assumed price ratio) |
|---|---|---|---|---|
| B-small-model | -38.3 (-45.4 to -31.2) | 176% of A-own-model | 0% of A-own-model | 22% of A-own-model |
| C-marked | -7.1 (-13.3 to -1.2) | 96% of A-own-model | 92% of A-own-model | 93% of A-own-model |
| D-self-checked | -22.1 (-28.7 to -15.8) | 219% of A-own-model | 111% of A-own-model | 122% of A-own-model |
| E-marked-fast | -10.0 (-16.7 to -3.3) | 137% of A-own-model | 83% of A-own-model | 87% of A-own-model |

## Test phase

| Arm | Right | Answered by the small model | Model seconds | Tokens on the user's model | Tokens on the small model | Checks |
|---|---|---|---|---|---|---|
| A-own-model | 80.0% (74.6% to 85.0%) | 0.0% | 538 | 182,189 | 0 | 0 |
| B-small-model | 48.3% (42.1% to 54.6%) | 100.0% | 812 | 0 | 191,442 | 0 |
| C-marked | 77.5% (72.1% to 82.9%) | 2.5% | 566 | 178,439 | 6,413 | 0 |
| D-self-checked | 59.2% (52.9% to 65.4%) | 75.4% | 1130 | 215,124 | 143,523 | 194 |
| E-marked-fast | 73.8% (68.3% to 79.2%) | 1.2% | 500 | 179,854 | 2,501 | 0 |

Against **A-own-model** on the same Tasks:

| Arm | Right, difference in points | Model seconds | Tokens on the user's model | Price at 0.2 to 1 (an assumed price ratio) |
|---|---|---|---|---|
| B-small-model | -31.7 (-39.2 to -24.2) | 151% of A-own-model | 0% of A-own-model | 21% of A-own-model |
| C-marked | -2.5 (-8.3 to +3.8) | 105% of A-own-model | 98% of A-own-model | 99% of A-own-model |
| D-self-checked | -20.8 (-28.3 to -13.8) | 210% of A-own-model | 118% of A-own-model | 134% of A-own-model |
| E-marked-fast | -6.2 (-12.1 to -0.4) | 93% of A-own-model | 99% of A-own-model | 99% of A-own-model |

## By kind of Task (Test phase)

| Arm | category-choice | date-choice | largest-number | sum-of-two | tracking-choice | unit-conversion | word-problem-multi-step | word-problem-one-step |
|---|---|---|---|---|---|---|---|---|
| A-own-model | 86.7%, small 0.0% | 50.0%, small 0.0% | 86.7%, small 0.0% | 83.3%, small 0.0% | 66.7%, small 0.0% | 100.0%, small 0.0% | 80.0%, small 0.0% | 86.7%, small 0.0% |
| B-small-model | 96.7%, small 100.0% | 36.7%, small 100.0% | 16.7%, small 100.0% | 56.7%, small 100.0% | 26.7%, small 100.0% | 60.0%, small 100.0% | 50.0%, small 100.0% | 43.3%, small 100.0% |
| C-marked | 93.3%, small 0.0% | 43.3%, small 3.3% | 90.0%, small 0.0% | 76.7%, small 0.0% | 56.7%, small 6.7% | 100.0%, small 0.0% | 83.3%, small 3.3% | 76.7%, small 6.7% |
| D-self-checked | 90.0%, small 90.0% | 50.0%, small 76.7% | 56.7%, small 90.0% | 50.0%, small 96.7% | 23.3%, small 80.0% | 63.3%, small 56.7% | 66.7%, small 60.0% | 73.3%, small 53.3% |
| E-marked-fast | 90.0%, small 0.0% | 36.7%, small 3.3% | 90.0%, small 0.0% | 80.0%, small 0.0% | 43.3%, small 3.3% | 100.0%, small 0.0% | 70.0%, small 3.3% | 80.0%, small 0.0% |

## What Continuum did

- **C-marked**: choices reference 456, explore 24; outcome sources explicit 480; strictness now preset "words-only".
- **D-self-checked**: choices exploit 293, reference 153, reference-check 19, explore 15; outcome sources reference_check 327, session_end 153; strictness now preset "sketch-first".
  - Checks by the user's own model: 327. They agreed with the answer key on 184 (56.3%). Of the 162 wrong answers checked, the check passed 138.
- **E-marked-fast**: choices reference 438, explore 42; outcome sources explicit 480; strictness now preset "balanced".

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
