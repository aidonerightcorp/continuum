# Run notes: everyday-v1

- 3:31 to 3:40 PM Eastern, Nov 10: the first start of this run did nothing. The wait loop in `run_everyday.sh`
  matched the text of its own parent command line and never ended. No Task was run and no record was written.
  The earlier script is kept as `run_everyday.sh.before-1540`.
- 3:40 PM: started again without that loop. Arm order: C-marked, A-own-model, B-small-model, D-self-checked,
  E-marked-fast. The arms do not depend on each other (each has its own Hermes home and record file).
- Plugin: the frozen copy of run two-phase-v1 (see `plugin-snapshot.json`). Code changes after that copy are not
  in this run.
