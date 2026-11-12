# Decisions (shared-interface changes need an entry here + owner stop)

## PROPOSED (not applied): benchmark result optional fields for T2/T3

- Context: OPTILOOP_TESTS.md T2 (energy cross-check) and T3 (compile break-even)
  require reporting values that have no key in shared interface 3.
- Proposal: allow OPTIONAL keys on benchmark result items: `recompilations`
  (int), `break_even_docs` (float|null), `static_cache_used` (str|null),
  `energy_poll_joules_per_doc` (float|null), `energy_crosscheck_ok` (bool|null).
  Absent = not measured. All existing keys unchanged; readers must tolerate
  absence.
- Status: awaiting owner decision. Workers: print these in console output and
  report them back; do NOT add JSON keys until approved.
