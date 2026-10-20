"""Unit tests for the guard: feature extraction, verdict hysteresis, rate limiting."""
import time

from cache.guard import (CLASSES, FEATURE_NAMES, RELEASE_WINDOWS, ConnStats, Guard,
                         RuleDetector, feature_row)


def test_feature_row_matches_names():
    row = feature_row(cmds=10, gets=8, sets=2, misses=4, keyed=10, distinct=5, top_key=3,
                      errors=1, auth_failures=0, bytes_in=400, pending_bytes=0, stalled_windows=0)
    f = dict(zip(FEATURE_NAMES, row))
    assert len(row) == len(FEATURE_NAMES)
    assert f["rate"] == 10 and f["get_ratio"] == 0.8 and f["miss_ratio"] == 0.5
    assert f["distinct_ratio"] == 0.5 and f["top_key_share"] == 0.3 and f["bytes_per_cmd"] == 40


def test_empty_window_is_all_zeros_not_a_division_error():
    s = ConnStats(1, ("127.0.0.1", 1))
    assert s.close_window() == [0.0] * len(FEATURE_NAMES)


def test_conn_stats_counts_misses_and_distinct_keys():
    s = ConnStats(1, None)
    for i in range(10):
        s.record(b"GET", f"k{i % 4}".encode(), miss=i < 3, error=False)
    f = dict(zip(FEATURE_NAMES, s.close_window()))
    assert f["miss_ratio"] == 0.3
    assert f["distinct_ratio"] == 0.4  # 4 distinct keys over 10 keyed commands
    assert f["top_key_share"] == 0.3   # k0 seen 3 times


def test_stalled_windows_count_only_trickles():
    s = ConnStats(1, None)
    s.pending_bytes = 30
    for expected in (1, 2, 3):
        s.bytes_in = 1
        assert dict(zip(FEATURE_NAMES, s.close_window()))["stalled_windows"] == expected
    s.bytes_in = 100_000  # a big upload in progress is not a stall
    assert dict(zip(FEATURE_NAMES, s.close_window()))["stalled_windows"] == 0


def test_rules_separate_penetration_from_a_warmer():
    pen = feature_row(cmds=500, gets=500, sets=0, misses=500, keyed=500, distinct=500, top_key=1,
                      errors=0, auth_failures=0, bytes_in=20000, pending_bytes=0, stalled_windows=0)
    # A warm-up job misses as much, but it SETs each key it misses, so every
    # key appears twice and distinct_ratio is 0.5.
    warm = feature_row(cmds=1000, gets=500, sets=500, misses=500, keyed=1000, distinct=500, top_key=2,
                       errors=0, auth_failures=0, bytes_in=40000, pending_bytes=0, stalled_windows=0)
    labels = [lbl for lbl, _ in RuleDetector().classify([pen, warm])]
    assert labels == ["penetration", "normal"]


def _tick(g, c):
    """One window in which the connection did something (empty windows are skipped)."""
    c.record(b"GET", b"k", miss=True, error=False)
    g.tick()


class _Fixed:
    """A detector that returns whatever label it is told to."""
    name = "fixed"

    def __init__(self):
        self.label = "normal"

    def classify(self, rows):
        return [(self.label, 0.99) for _ in rows]


def test_verdict_needs_release_windows_of_normal_to_lift():
    g = Guard("rules")
    det = _Fixed()
    g._rules = det
    c = g.register(None)
    det.label = "penetration"
    _tick(g, c)
    assert c.verdict == "penetration"
    det.label = "normal"
    for _ in range(RELEASE_WINDOWS - 1):
        _tick(g, c)
        assert c.verdict == "penetration"
    _tick(g, c)
    assert c.verdict == "normal"


def test_empty_window_is_not_sent_to_the_detector():
    g = Guard("rules")
    det = _Fixed()
    det.label = "slowloris"
    g._rules = det
    c = g.register(None)
    g.tick()  # no traffic at all: no evidence, so no verdict
    assert c.verdict == "normal"


def test_limited_client_gets_about_its_rate():
    g = Guard("rules")
    det = _Fixed()
    g._rules = det
    c = g.register(None)
    det.label = "penetration"
    _tick(g, c)
    allowed = 0
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.0:
        allowed += g.admit(c) is None
    assert 10 <= allowed <= 30  # MITIGATION allows 20/s for penetration


def test_refused_commands_do_not_make_an_attacker_look_calm():
    # A throttled penetration attacker: 20 GETs ran (all misses), 780 refused.
    s = ConnStats(1, None)
    for i in range(20):
        s.record(b"GET", f"u{i}".encode(), miss=True, error=False)
    for i in range(780):
        s.record_refused(f"v{i}".encode())
    f = dict(zip(FEATURE_NAMES, s.close_window()))
    assert f["rate"] == 800           # what it is attempting
    assert f["miss_ratio"] == 1.0     # judged only on what ran
    assert f["error_ratio"] == 0.0    # refusals are not its errors
    assert f["distinct_ratio"] == 1.0
    assert RuleDetector().classify([s.last_features])[0][0] == "penetration"


def test_one_failed_auth_is_never_acted_on():
    from cache.guard import apply_policy
    row = feature_row(cmds=1, gets=0, sets=0, misses=0, keyed=0, distinct=0, top_key=0, errors=1,
                      auth_failures=1, bytes_in=40, pending_bytes=0, stalled_windows=0, auth_failures_total=1)
    assert apply_policy("brute_force", 0.999, row)[0] == "normal"
    row3 = feature_row(cmds=1, gets=0, sets=0, misses=0, keyed=0, distinct=0, top_key=0, errors=1,
                       auth_failures=1, bytes_in=40, pending_bytes=0, stalled_windows=0, auth_failures_total=3)
    assert apply_policy("brute_force", 0.999, row3)[0] == "brute_force"


def test_observe_mode_never_refuses():
    g = Guard("observe")
    c = g.register(None)
    c.verdict = "penetration"
    assert g.admit(c) is None


def test_classes_start_with_normal():
    # train_guard and the evaluation treat class index 0 as "not an attack".
    assert CLASSES[0] == "normal"
