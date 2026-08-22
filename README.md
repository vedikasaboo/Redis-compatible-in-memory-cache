# Evict

A Redis-compatible in-memory cache built from scratch in Python with
`asyncio` — RESP over TCP, event-loop concurrency, a consistent-hash
cluster, crash recovery via write-ahead logging, and a learned eviction
policy benchmarked honestly against the classical ones.

Real `redis-cli` and `redis-benchmark` drive it unmodified.

---

## Architecture

```
                    ┌─────────────────────────────────────────┐
                    │            ClusterClient                │
                    │  hash(key) -> ring -> owning node       │
                    │  one connection per node, no proxy      │
                    └───────┬──────────┬──────────┬───────────┘
                            │          │          │
                  RESP/TCP  │          │          │
              ┌─────────────┘          │          └─────────────┐
              ▼                        ▼                        ▼
    ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
    │     node-0       │     │     node-1       │     │     node-2       │
    │ ┌──────────────┐ │     │                  │     │                  │
    │ │ asyncio loop │ │     │      (same)      │     │      (same)      │
    │ │ 1 coroutine  │ │     │                  │     │                  │
    │ │ per conn     │ │     └──────────────────┘     └──────────────────┘
    │ └──────┬───────┘ │
    │        ▼         │      Per node, the request path:
    │ ┌──────────────┐ │
    │ │ RESP parse   │ │        read() ──► parse ──► stage to WAL
    │ │ (framing)    │ │                                 │
    │ └──────┬───────┘ │                                 ▼
    │        ▼         │                            apply to store
    │ ┌──────────────┐ │                                 │
    │ │  dispatch    │ │                                 ▼
    │ └──────┬───────┘ │                        await durability
    │        ▼         │                                 │
    │ ┌──────────────┐ │                                 ▼
    │ │    store     │ │                          encode ──► write()
    │ │  LRU / LFU   │ │
    │ │  / learned   │ │      Background coroutines:
    │ └──────┬───────┘ │        · WAL commit loop (fsync in executor)
    │        ▼         │        · snapshot writer (incremental, yields)
    │ ┌──────────────┐ │        · active TTL expiry sweep
    │ │ WAL + snapshot│ │
    │ └──────────────┘ │
    └──────────────────┘
```

Four independent pieces, each with its own document:

| Piece | Where | Notes |
|---|---|---|
| RESP protocol + framing | `cache/resp.py` | [NETWORKING.md](NETWORKING.md) §1–2 |
| Event-loop concurrency | `cache/server.py` | [DESIGN.md](DESIGN.md) |
| Consistent-hash cluster | `cache/hashring.py`, `cache/cluster.py` | [NETWORKING.md](NETWORKING.md) §4 |
| WAL + snapshots | `cache/wal.py`, `cache/snapshot.py` | [NETWORKING.md](NETWORKING.md) §5 |
| Eviction + ML | `cache/eviction.py`, `cache/ml_policy.py` | below |

---

## Headline results

All measured on this machine (M-series MacBook Air, macOS 14, Python 3.11).

### Throughput

Measured with `redis-benchmark -n 10000`, the real Redis client:

| | SET | GET |
|---|---:|---:|
| **This cache** (Python, single-threaded) | ~100,000/s | ~120,000/s |
| Real Redis 8.8.0 (C), same machine | 215,517/s | 221,239/s |
| Real Redis, pipelined `-P 16` | 1,562,500/s | 2,127,660/s |

Within ~2× of C Redis from pure Python, single-threaded. Run-to-run
variance is a few percent, so these are rounded rather than reported to the
digit. The pipelining row is why the parser had to handle many commands per
read in the first place.

### Durability modes (8 concurrent clients)

| mode | writes/sec |
|---|---:|
| `always`, no group commit | 9,195 |
| `always`, **group commit** | **22,535** (2.5×) |
| `everysec` | 59,771 |

At one client the group-commit speedup is 1.0× — nothing to batch, which is
the control that shows the 2.5× is real batching rather than noise.

**200/200 `kill -9` crash-recovery cycles pass** (100 per mode, 453,075
acknowledged writes verified). `always` guarantees every acknowledged write
survives the kill; `everysec` guarantees the log is never corrupt and
recovery yields a clean prefix of the write sequence — the two claims are
asserted separately, because they are genuinely different guarantees.

### Consistent hashing (200,000 keys, 4 nodes)

| | max/min load | keys moved when a 5th node joins |
|---|---:|---:|
| 1 virtual node | 4.30× | — |
| **150 virtual nodes** | **1.14×** | **22.6%** (1/N ideal = 20%) |
| modulo hashing | 1.00× | ~80% |

![key distribution](results/key_distribution.png)

### Eviction policies — hit rate (200k accesses, 1 MB budget, 5 repeats)

Every sampling policy gets the **same 64 candidates per eviction**, and every
policy gets the same **byte** budget (so a policy with fatter per-key
metadata gets proportionally fewer keys).

| policy | cyclic | mixed | scan | zipfian |
|---|---:|---:|---:|---:|
| **learned (ML)** | **30.61%** | 77.18% | **23.41%** | 89.02% |
| sampled-lfu | 0.00% | **77.73%** | 0.00% | **89.45%** |
| sampled-lru | 0.00% | 73.96% | 0.00% | 88.31% |
| true-lru | 0.00% | 73.53% | 0.00% | 87.99% |
| random | 6.96% | 71.82% | 2.39% | 86.22% |

![eviction hit rate](results/eviction_hit_rate.png)

**The learned policy wins two of four, and loses two.** All four gaps are
statistically significant over 5 repeats.

Where it wins, it wins completely. On `cyclic` and `scan` every classical
policy scores **exactly 0.00%** — each key is evicted just before its next
use, so recency and frequency are not merely unhelpful but actively
misleading. The model reaches **23.41% on `scan` against a theoretical
ceiling of 23.4%** (capacity ÷ key range), and 30.61% on `cyclic` against a
31.2% ceiling. It has essentially found the optimal strategy on both.

Where it loses, it loses narrowly: 0.55 points on `mixed`, 0.44 on
`zipfian` — the two workloads that most resemble real traffic. On those, a
plain frequency counter is simply hard to beat, and the model's noisier
estimate of the same signal costs it.

#### An earlier version of this table was not a fair fight

It gave the learned policy 16 candidates per eviction while the classical
policies got 5, justified by the model's scoring being a cheap batched
prediction. That reasoning is true but not exclusive — scoring a candidate
for LFU is two attribute reads, cheaper still. Measured, **LFU gains just as
much from a wider sample as the model does** (`mixed`: 76.32% at 5
candidates → 77.72% at 64). Correcting it flipped `mixed` from a claimed win
into a loss. The table above is the corrected one.

#### Things tried that did not work

- **A ranking objective.** The policy never consumes a predicted distance as
  a number; it only asks which of 64 candidates is worst. Training
  LightGBM's LambdaRank to optimise that ordering directly is the
  theoretically right move, and it did not help: `mixed` 77.37% vs 77.71%
  for plain regression, `zipfian` 88.98% vs 89.00%. Available behind
  `--objective rank`, left off by default.
- **A single model across all workloads.** Markedly worse — "older means
  further from reuse" is right on Zipfian traffic and exactly backwards on a
  cycle. Models are trained per workload.

#### What did work

1. **Cheaper features (40 → 24 bytes/key).** The first version stored five
   fields and lost workloads it should have won, purely by running an 11%
   smaller cache at the same byte budget. A running gap-sum and previous gap
   collapsed into one EWMA; value size was dropped entirely (zero feature
   importance, and free to derive with `len()`). `age_over_gap` — how
   overdue a key is in units of its own rhythm — is derived, so it costs
   nothing, and it is what lets one model read opposite regimes correctly.
2. **On-policy retraining.** Examples are collected once with LRU shaping
   the resident set, then re-collected with the round-one model driving
   eviction, and the model refit on those. A model trained only on LRU's
   resident sets is being asked, at serving time, about keys LRU would never
   have kept.
3. **A wider candidate sample.** 16 → 64 lifted `cyclic` from 28.8% to
   30.6% and `scan` from 21.6% to 23.4%, for only 25% more inference time,
   because per-call overhead dominates a batch this small. It helps the
   classical policies too, which is exactly why they now get it as well.

#### Cost

At equal sample size, on an eviction-heavy workload:

| policy | µs per operation |
|---|---:|
| true-lru | 0.83 |
| sampled-lru | 17.58 |
| sampled-lfu | 19.50 |
| learned | 87.84 |

So the model is ~4.5× the cost of sampled-LFU and ~100× true-LRU. Inference
runs **only at eviction** — never on `GET`/`SET`, and a test asserts that
reads and non-evicting writes trigger zero predictions. Only the first 50
trees are evaluated: hit rate is measurably flat from 25 trees to the full
276 while latency falls 2.8×, since ranking 64 candidates does not need the
regression accuracy the extra trees buy.

**The honest summary:** on traffic that looks real, a well-tuned LFU beats
this model, and costs a fraction as much. The model earns its place only
where access patterns defeat recency and frequency outright — but there it
is not a marginal improvement, it is the difference between a cache that
works and one returning 0%.

## Running it

```bash
# setup (once)
python3.11 -m venv venv && source venv/bin/activate
pip install numpy lightgbm xxhash pytest matplotlib scikit-learn
brew install redis libomp          # redis-cli/benchmark; libomp for lightgbm

# single node — then talk to it with real redis-cli
python -m cache.server --port 6390 --capacity 10000
redis-cli -p 6390 SET foo bar
redis-cli -p 6390 GET foo
redis-cli -p 6390 INFO

# with persistence
python -m cache.server --port 6390 --data-dir ./data --fsync everysec

# with the learned eviction policy
python -m cache.server --port 6390 --eviction ml --model models/mixed.txt

# 3-node cluster
python tools/launch_cluster.py --nodes 3 --data-root ./data

# the full demo (cluster, load, kill -9, WAL recovery) — narrated
./demo.sh

# tests
pytest -q

# reproduce every benchmark
./validate.sh                                   # redis-cli + redis-benchmark
python tools/bench_fsync.py                     # durability modes
python tools/crash_test.py --iterations 100     # kill -9 recovery
python tools/plot_distribution.py               # hash ring balance
python tools/run_experiments.py                 # traces, training, eviction study
```

Every generator is seeded and every result file is written with the config
that produced it, so the numbers above are reproducible.

---

## Limitations

Stated plainly, because most of them are deliberate scope decisions rather
than oversights.

- **Memory is bounded by key count, not bytes.** `--capacity` counts keys.
  The eviction *benchmark* converts a byte budget into a per-policy key
  count, but the server itself does not track real memory. A workload with
  wildly varying value sizes would blow past any intended footprint.
- **No replication.** Each shard is a single point of failure for its own
  keys. A dead shard errors cleanly and recovers from its own WAL, but until
  it comes back, its slice of the keyspace is unavailable. No failover.
- **Static cluster membership.** Adding a node changes where keys are looked
  up, not where data lives — there is no live migration, so a join
  effectively invalidates the keys that moved. `hashring` supports
  `add_node`/`remove_node`; wiring that to online resharding is the missing
  piece.
- **Cache warmth is lost on recovery.** The WAL restores *data*, but recency
  and frequency statistics start from scratch, so a recovered node evicts
  badly until its access history rebuilds.
- **Single-machine benchmarks.** Every number here was measured with client
  and server on one host over loopback. Real network latency, packet loss,
  and NIC limits are entirely absent, which flatters the throughput figures
  and completely hides partition behaviour.
- **A small command set.** `GET`, `SET` (with `EX`/`PX`/`EXAT`/`PXAT`),
  `DEL`, `EXISTS`, `TTL`, `PTTL`, `PING`, `INFO`. No hashes, lists, sets,
  transactions, pub/sub, or `SCAN`.
- **The learned policy is trained per workload, offline.** Each model is
  fitted to one traffic shape, in two passes, against a trace generated from
  the same distribution it is later evaluated on. Real traffic shifts; there
  is no online adaptation and no drift detection, so a deployed model would
  silently decay as the workload moved away from what it was fitted to.
  Detecting that is the obvious next piece of work.
- **`__len__` counts expired-but-uncollected keys.** Matching Redis's
  `DBSIZE`; use `GET`/`TTL` for expiry-accurate answers.
- **No auth, no TLS.** Bind to localhost.

---

## Documents

- [DESIGN.md](DESIGN.md) — where the request path yields, and why the store
  needs no locks
- [NETWORKING.md](NETWORKING.md) — mapping from project parts to networking
  concepts
- [NOTES.md](NOTES.md) — Phase 0 exploration: real RESP bytes captured off
  the wire, and the baseline Redis benchmarks
