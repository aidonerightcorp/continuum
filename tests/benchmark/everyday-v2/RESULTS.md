# Continuum in Hermes: run `everyday-v2`

## Conditions

- Tasks: 480 checkable Tasks (half are four easy kinds written by `make_everyday_v2_items.py` from a fixed seed, half are four harder kinds from public sets) in one fixed mixed order, each one a new Hermes session. Kinds: category-choice (60), date-choice (60), largest-number (60), sum-of-two (60), tracking-choice (60), unit-conversion (60), word-problem-multi-step (60), word-problem-one-step (60).
- Learning phase (Tasks 1 to 240): 240 Tasks, 30 of each of eight kinds. Continuum starts with an empty record file and is never told a kind.
- Test phase (Tasks 241 to 480): 240 other Tasks of the same eight kinds. Continuum keeps the records of the learning phase. This phase carries the headline figures.
- Hermes: its own agent loop (`AIAgent`, as in Hermes's batch runner), Hermes's own system prompt, toolsets: none (see the limits below), at most 6 model calls a Task.
- Models: the user's own model is `coder-7b-bench`, the small model is `meta-3b-bench`; both are open-weight models on one 12 GB graphics card on one machine, with a real context window of 16384 tokens and temperature 0.2. Largest prompt in the run: 738 tokens.
- Every Task ends with the same request in every arm: "Work through it step by step. Then end your reply with one last line in this form: Answer: <the number, or the letter of the option>".
- Scoring: the last `Answer:` line of the reply against the answer key (from the generating code for the easy kinds, from the public set for the others).
- Cost: seconds of model time on that card and tokens, from Hermes's own per-call hooks, the same in every arm. Checks by the user's own model are counted in the arm that makes them.
- One pass, one order, one seed. Intervals: 95%, from resampling the Tasks.

## Arms

- **A-own-model**: Hermes with the user's own (larger) model for every Task. Continuum is not loaded.
- **A2-own-model-repeat**: A second pass of arm A: the same model, the same Tasks, no plugin. The difference between A and A2 is the run-to-run noise of one model at temperature 0.2; a plugin arm is read against it.
- **B-small-model**: Hermes with the small model for every Task. Continuum is not loaded.
- **C-marked**: Continuum on, default settings. After every turn the harness marks the answer good or bad with the plugin's own /continuum command, from the answer key. This stands for a user who marks every turn.
- **D-self-checked**: Continuum on, nobody marks anything. The user's own model checks every turn that the small model answered (one PASS or FAIL question). The answer key is used only to score the run afterwards.

## Learning phase

| Arm | Right | Answered by the small model | Model seconds | Tokens on the user's model | Tokens on the small model | Checks |
|---|---|---|---|---|---|---|
| A-own-model | 77.9% (72.5% to 82.9%) | 0.0% | 618 | 184,170 | 0 | 0 |
| A2-own-model-repeat | 81.2% (76.2% to 85.8%) | 0.0% | 605 | 182,962 | 0 | 0 |
| B-small-model | 52.5% (46.2% to 58.8%) | 100.0% | 592 | 0 | 193,740 | 0 |
| C-marked | 79.2% (73.8% to 84.2%) | 7.1% | 636 | 169,147 | 13,730 | 0 |
| D-self-checked | 81.2% (76.2% to 86.2%) | 0.0% | 579 | 182,286 | 0 | 0 |

Against **A-own-model** on the same Tasks:

| Arm | Right, difference in points | Model seconds | Tokens on the user's model | Price at 0.2 to 1 (an assumed price ratio) |
|---|---|---|---|---|
| A2-own-model-repeat | +3.3 (-2.1 to +9.2) | 98% of A-own-model | 99% of A-own-model | 99% of A-own-model |
| B-small-model | -25.4 (-32.5 to -18.3) | 96% of A-own-model | 0% of A-own-model | 21% of A-own-model |
| C-marked | +1.2 (-4.6 to +7.1) | 103% of A-own-model | 92% of A-own-model | 93% of A-own-model |
| D-self-checked | +3.3 (-2.1 to +9.2) | 94% of A-own-model | 99% of A-own-model | 99% of A-own-model |

## Test phase

| Arm | Right | Answered by the small model | Model seconds | Tokens on the user's model | Tokens on the small model | Checks |
|---|---|---|---|---|---|---|
| A-own-model | 80.4% (75.4% to 85.0%) | 0.0% | 624 | 184,198 | 0 | 0 |
| A2-own-model-repeat | 80.4% (75.0% to 85.0%) | 0.0% | 589 | 185,468 | 0 | 0 |
| B-small-model | 47.5% (41.2% to 53.8%) | 100.0% | 463 | 0 | 191,636 | 0 |
| C-marked | 78.3% (72.9% to 83.3%) | 4.2% | 642 | 175,599 | 8,602 | 0 |
| D-self-checked | 78.8% (73.3% to 83.8%) | 0.0% | 617 | 184,764 | 0 | 0 |

Against **A-own-model** on the same Tasks:

| Arm | Right, difference in points | Model seconds | Tokens on the user's model | Price at 0.2 to 1 (an assumed price ratio) |
|---|---|---|---|---|
| A2-own-model-repeat | +0.0 (-5.8 to +5.8) | 94% of A-own-model | 101% of A-own-model | 101% of A-own-model |
| B-small-model | -32.9 (-40.0 to -26.2) | 74% of A-own-model | 0% of A-own-model | 21% of A-own-model |
| C-marked | -2.1 (-7.9 to +4.2) | 103% of A-own-model | 95% of A-own-model | 96% of A-own-model |
| D-self-checked | -1.7 (-7.1 to +3.3) | 99% of A-own-model | 100% of A-own-model | 100% of A-own-model |

## By kind of Task (Test phase)

| Arm | category-choice | date-choice | largest-number | sum-of-two | tracking-choice | unit-conversion | word-problem-multi-step | word-problem-one-step |
|---|---|---|---|---|---|---|---|---|
| A-own-model | 96.7%, small 0.0% | 53.3%, small 0.0% | 93.3%, small 0.0% | 90.0%, small 0.0% | 46.7%, small 0.0% | 100.0%, small 0.0% | 73.3%, small 0.0% | 90.0%, small 0.0% |
| A2-own-model-repeat | 93.3%, small 0.0% | 56.7%, small 0.0% | 96.7%, small 0.0% | 73.3%, small 0.0% | 50.0%, small 0.0% | 100.0%, small 0.0% | 86.7%, small 0.0% | 86.7%, small 0.0% |
| B-small-model | 93.3%, small 100.0% | 30.0%, small 100.0% | 43.3%, small 100.0% | 30.0%, small 100.0% | 43.3%, small 100.0% | 53.3%, small 100.0% | 33.3%, small 100.0% | 53.3%, small 100.0% |
| C-marked | 100.0%, small 6.7% | 43.3%, small 6.7% | 96.7%, small 0.0% | 80.0%, small 6.7% | 50.0%, small 10.0% | 96.7%, small 0.0% | 70.0%, small 3.3% | 90.0%, small 0.0% |
| D-self-checked | 96.7%, small 0.0% | 43.3%, small 0.0% | 90.0%, small 0.0% | 86.7%, small 0.0% | 53.3%, small 0.0% | 96.7%, small 0.0% | 73.3%, small 0.0% | 90.0%, small 0.0% |

## What Continuum did

- **C-marked**: choices reference 453, explore 27; outcome sources explicit 480; strictness now preset "shape-first".
- **D-self-checked**: choices reference 480; outcome sources session_end 480; strictness now .

## Faults

Tasks that ended with a harness error (counted as wrong): A-own-model 0, A2-own-model-repeat 0, B-small-model 0, C-marked 0, D-self-checked 0.

## Limits

- One pass in one order with one seed on one machine. A second pass can differ by a few points; read the intervals.
- No tools. Hermes refuses tool use on a local endpoint with a window under 64,000 tokens, and this card cannot hold both models at that size. The Hermes config declares 64000 tokens; the models really serve 16384. No prompt in the run came near that.
- Two small open-weight models (4-bit). The gap between a user's own model and a cheaper one is different on other models, and so is the share of Tasks that can move.
- Every Task is a new session with one turn. Outcome signals that need a second message (a thank-you, a correction) do not occur here.
- In the marked arms the harness marks every answer from the answer key. That stands for a user who marks every turn; few users do. The self-checked arm uses no answer key while it runs.
- The price column uses an assumed price ratio of 0.2 to 1 between the two models. The measured units are seconds and tokens.
- Seconds are model seconds on a shared 12 GB card, where the two models take turns in memory. A model switch costs load time here that a hosted provider does not have.
- Plugin code: a frozen copy taken 2026-11-10 22:19:27 EST (42 files, mark `0460b6e8`, 244 passed; full record in `results.json`). Later changes to the plugin are not in this run.
