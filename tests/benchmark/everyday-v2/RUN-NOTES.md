# everyday-v2: notes on the run

- Purpose: confirm the change of November 10, evening (a PASS from the user's own model and a thank-you count as weak
  evidence; only strong outcomes such as marks move a kind of Task; trial turns wait for a recent mark) on 480 Tasks
  that no earlier run, replay or tuning read (`items/everyday-v2-two-phase.jsonl`, written by
  `make_everyday_v2_items.py`; overlap with everyday-v1 and two-phase-v1: 0).
- Arms A, A2 and B do not load the plugin. A2 repeats A on the same Tasks to measure how much one model varies
  between two passes at temperature 0.2.
- Arms C and D ran the plugin copy frozen at 22:19:27 EST (`plugin-snapshot.json`, content mark 0460b6e8…, 244 tests).
- After that copy, three more independent reviews led to further changes (a thank-you weighs 0.3; trial turns are
  switched off separately from comparison turns; forget-text paths; the audit rates both models on marks; the status
  counts a saving only on cheaper turns marked good). None of them changes what happens in this run's sessions: one
  Task per session, no thank-you messages, no forget-text, no imported records, every turn marked (C) or none (D).
  The release copy was therefore not run again on these Tasks.
- The local model server held other models between the arms (aliases for a film were made at about 22:40 and
  23:05); no arm ran while another process used the graphics card.
- In the copy published with the plugin (`tests/benchmark/everyday-v2/`), one reply of the small model that wrote out
  a made-up home-folder path has that path replaced by `[path removed]`; nothing else in the records is changed.
- A and A2 (the same model, the same Tasks, no plugin) scored 380 and 388 of 480 and 193 of 240 each in the test
  phase, but gave a different right-or-wrong result on 102 Tasks: 47 right only in A, 55 right only in A2 (counted
  from the two `records.jsonl` files).
