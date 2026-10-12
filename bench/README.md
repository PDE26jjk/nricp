# Benchmark: wrapping the four acceptance pairs

Fits a decimated template onto a cross-topology target and reports the quality
metrics the acceptance gates are written in.

## What a case is

All of it is pinned in `config/bench_config.json`, which every run records:

- the template is decimated by farthest-point grouping (~4000 vertices);
- the target is decimated with its **own** grouping (~3500 vertices), so the
  solver never sees the shared indexing that scoring uses;
- ground truth per template vertex comes from the shared triangulation of the
  high-resolution pair, which is the only reason exact correspondence error is
  measurable at all;
- 25 landmarks are spread by farthest-point sampling over the template, with
  target positions read from that ground truth (the analogue of hand-placed
  control points);
- the rigid initialization is a closed-form Kabsch fit over those landmarks.
  It replaces the earlier surface-ICP start: the library's surface sampling
  cannot be seeded, so an ICP start made prepared cases *non-reproducible*
  (fixture hashes differed between identical runs), and it costs a sampling
  pass for no measured quality gain.

## Dataset

Not distributed here. `prepare_data.py` verifies a local copy against
`config/data_manifest.json` (size and sha256) and copies it into `bench/data/`,
which is untracked:

```powershell
python bench\prepare_data.py --source <directory with the high-res pairs>
python bench\prepare_data.py --source <dir> --write-manifest   # refresh hashes
```

Without a source it prints exactly which files are required and exits; the
runners refuse to start rather than failing obscurely.

## Running

```powershell
python bench\run_bench.py --backend reference --methods rigid,amberg,sumner
python bench\run_bench.py --pairs victoria0->victoria2 --methods sumner --tag smoke
python bench\golden_regen.py --methods amberg,sumner
```

Outputs land in `bench/out/` (results table, run metadata with the full config
and versions, per-vertex error maps) and `bench/fixtures/` (golden fixtures plus
an index of content hashes). The run exits non-zero when any gate is violated
and names the pair and the gate.

**Cost:** a full four-pair pass takes minutes per backend method on CPU (the
reference solvers are single-threaded), and regenerating the fixtures takes
about the same again. In this workspace both are run only on request.

## Metrics and gates

| Metric | Meaning | Gate |
|---|---|---|
| `corr_mean` | mean distance from each fitted vertex to its ground-truth position | ≤ 1.05 × reference |
| `corr_p95` | p95 of the same | ≤ 1.10 × reference |
| `corr_mean_fraction_of_diagonal` | the same as `corr_mean`, relative to the target's bounding-box diagonal | reported (target 1%) |
| `stretch_p95` | p95 of relative template edge-length change | ≤ reference + 0.05 |
| `foldover` | fraction of faces whose normal flipped against the rigidly aligned template | ≤ reference + 0.01 |
| `chamfer_A2B` | mean distance from fitted vertices to the target surface | reported |

Acceptance is anchored on the reference baseline in
`config/reference_baseline.json`, which `update_reference_baseline.py` derives
from a reference run: per pair, the bar is the best correspondence error the
reference reached there, with the other metrics taken from that same method.
Absolute targets (1% of the diagonal, stretched edges, no flipped faces) are
recorded in the config and reported, but they do not gate: the reference itself
reaches the 1% target on two of the four pairs only, and never reaches zero
flipped faces.

## Reference baseline (pinned configuration)

Measured with the reference backend, landmark initialization, 25 landmarks:

| pair | method | corr_mean | % diag | stretch p95 | foldover | chamfer |
|---|---|---|---|---|---|---|
| cat0→cat3 | amberg | 3.12 | 1.35 | 0.38 | 0.010 | 0.45 |
| cat0→cat3 | sumner | 2.07 | 0.90 | 0.36 | 0.004 | 0.40 |
| david0→david1 | amberg | 4.21 | 2.34 | 0.46 | 0.076 | 0.70 |
| david0→david1 | sumner | 3.05 | 1.70 | 0.49 | 0.082 | 0.54 |
| michael0→michael2 | amberg | 6.10 | 3.63 | 0.56 | 0.042 | 0.92 |
| michael0→michael2 | sumner | 3.22 | 1.92 | 0.45 | 0.126 | 0.66 |
| victoria0→victoria2 | amberg | 5.11 | 2.51 | 0.56 | 0.061 | 0.82 |
| victoria0→victoria2 | sumner | 2.05 | 1.01 | 0.34 | 0.056 | 0.46 |

What this table means for the gates:

1. The absolute correctness target (≤ 2.1, ≈1% of the diagonal) is reached by
   the reference on cat0→cat3 and victoria0→victoria2 only; david and michael sit
   at 1.7–1.9% of the diagonal. Those published numbers came from a single
   victoria run, and they are not a universal property of the reference.
2. `foldover = 0` is not reached by the reference on any pair; it ranges from
   0.4% to 12.6% of faces.

Hence the relative gates above. A candidate for the product pipeline is to
reduce the flip rate in a finishing pass (smoothing, normal re-orientation) and
measure the gate after it; that would be a separate, measurable improvement over
this baseline.

## Native backend

`--backend module` runs this project's solver through the same cases, gates, and
fixtures. It arrives with route A; until then the flag reports that clearly.
