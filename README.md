# Five audit gates for interpretability claims in pulse-wave models

Executable material for the paper *Five Audit Gates for Interpretability Claims in
Pulse-Wave Models — A Reproducible Protocol Applied to a Cuffless Blood-Pressure
Pipeline*.

The claim the paper makes about its own arithmetic is narrow and checkable: every number
it prints was produced by a named analysis, and a number that was typed by hand stops
the build. This repository is what makes that claim runnable. Point the trace at the
manuscript and it either finds an entry for each value or fails.

## What is here

```
registry/registry.json    one entry per value printed in the paper, with its provenance
tables/gates.csv          Table I, five columns
tables/gates_full.csv     the full nine column table (Table V of the supplement)
tables/threshold_timing.csv  when each threshold was fixed, and how that was classified
checks/                   five checks, each one rule per line, non-zero exit on failure
lean/Audit.lean           the gate logic and the threshold arithmetic, machine checked
figures/                  the figures of the paper
```

## Quick start

```
python run_checks.py                    # everything below, in order
```

The individual checks, each of which can be run on its own.

```
python checks/check_registry.py         # schema, ids, count, precision, value ranges
python checks/check_tables.py           # tables against the registry and against each other
python checks/check_hashes.py           # nothing changed since the hash was recorded
python checks/poison_test.py            # the checks still catch a planted error
python checks/check_numbers.py path/to/manuscript.md
```

No third party packages are needed, only Python 3.9 or newer. `make check` and
`make demo` are thin wrappers for the same commands.

The formal file needs Lean 4. If `lean` is not on `PATH`, pass it explicitly.

```
lean lean/Audit.lean
LEAN=/path/to/lean python run_checks.py
```

It prints the five rows of Table I when it type checks, so a pass leaves visible output
rather than only a zero exit code.

## The five gates

| Gate | Question | Timing | Verdict | Claim removed |
|---|---|---|---|---|
| G1 | Can every value be recomputed from a hashed file? | P | PASS | none |
| G2 | Is the truth independent of the model that produced the features? | P | FAIL | independent validation |
| G3 | Is the quantity reliable at the unit at which it is reported? | P | PARTIAL | single-beat usability |
| G4 | Does the test reject at its nominal rate under the null? | H | FAIL | the family-ordering claim |
| G5 | Does the reported unit survive reseeding of the model? | R | SPLIT | feature-level attribution |

Timing is P for pre-specified, R for registered before the analysis it judges, and H for
post hoc. A split is not a pass: it means one unit failed and another survived, so the
claim is reportable only at the unit that survived. `tables/threshold_timing.csv` gives
the criterion, the threshold, the date it was fixed, and the protocol file that records
it for each gate.

## The number registry

`registry/registry.json` holds 174 entries. Each one carries a value, the unit it is
measured in, and the analysis that produced it. The provenance strings are kept in the
language of the audit log, which is Chinese for most entries.

The trace is the load-bearing part. `checks/check_numbers.py` reads a manuscript, drops
citation keys, URLs, section and figure numbers, and years, then requires every remaining
number to match a registry entry at the precision printed in the text. Run against the
manuscript of the paper it reports 188 tokens and no unregistered value.

A value may be registered twice when the same number is reported in two places, and a
sign may be typeset away from the digits, so the match is on the magnitude as well as on
the signed value.

## Poison tests

`checks/poison_test.py` works on a throwaway copy of the repository and plants two
errors.

| Test | Planted error | Must be caught by |
|---|---|---|
| A | a registered value is rewritten, 0.366 becomes 0.466 | the manuscript trace |
| B | a table cell is edited after its hash was recorded | the hash check |

If either mutation passes, the corresponding check is not load bearing and the test
fails. This is the same pair the paper describes for gate 1, in executable form.

## Formal checks

`lean/Audit.lean` restates in Lean 4 the two things a reader is entitled to take on
trust from prose, and does not.

1. The combination rule. Survival is the absence of a failing verdict, so one failure
   removes the claim whatever the other four gates returned. Table I is encoded as data,
   and the file proves that it holds two failures and that the pipeline as a whole does
   not survive.
2. The arithmetic behind each verdict. Every threshold comparison is a theorem over
   integers, carried in thousandths so that nothing is decided by floating point. The
   scan that gate 3 was put through is proved invariant in the direction that matters: a
   quantity below the tightest bound is below every looser bound, so the verdict cannot
   be an artefact of where the scan started.

## Scope and limits

What this repository does not contain, and why.

- **Waveforms.** PWDB is public but large, VitalDB requires registration, and MIMIC-III
  requires credentialed access through PhysioNet. None is redistributed here.
- **Intermediate quantities.** The build registry used during the study holds 200
  expressions over 48 hashed source files. This repository ships the manuscript-facing
  subset, one entry per printed value. The remaining entries are intermediate to a
  reported value and are computed from the waveform files above, so they cannot be
  reproduced without them. The two agree on every value that appears in the paper, which
  is what the trace checks.
- **Model training.** Feature extraction, the three architectures, and the sparse
  autoencoder are not included. The numbers they produced are.

The audit is a procedure, not a certificate. A passing gate says the claim was stated at
a unit where it can be checked and that the threshold that decides it was fixed in
advance. It says nothing about whether the claim is true.

## Citation

```
@software{five_audit_gates_2026,
  title  = {Five audit gates for interpretability claims in pulse-wave models},
  author = {TODO},
  year   = {2026},
  url    = {https://github.com/ccartermices/five-audit-gates}
}
```

## License

MIT, see `LICENSE`.
