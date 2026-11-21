import numpy as np
import pytest

from fogrl.config import ScenarioSpec, load_requests, load_scenarios
from fogrl.traffic import (
    generate_stream,
    generate_stream_until,
    load_trace_counts,
    make_synthetic_trace,
)

REQUESTS = load_requests("configs/requests.yaml")
SCENARIOS = load_scenarios("configs/scenarios.yaml")


def rate_in(stream, start, end):
    mask = (stream.times_s >= start) & (stream.times_s < end)
    return mask.sum() / (end - start)


def test_same_seed_same_stream_different_seed_differs():
    spec = SCENARIOS["spikes"]
    a = generate_stream(spec, 50.0, 500, REQUESTS, seed=3)
    b = generate_stream(spec, 50.0, 500, REQUESTS, seed=3)
    c = generate_stream(spec, 50.0, 500, REQUESTS, seed=4)
    assert np.array_equal(a.times_s, b.times_s)
    assert np.array_equal(a.type_index, b.type_index)
    assert np.array_equal(a.work_ms, b.work_ms)
    assert not np.array_equal(a.times_s, c.times_s)


@pytest.mark.parametrize("name", ["low", "medium", "high", "spikes", "diurnal", "flash_crowd"])
def test_times_increase_and_mean_rate_matches(name):
    spec = SCENARIOS[name]
    base = 40.0
    stream = generate_stream(spec, base, 20000, REQUESTS, seed=1)
    assert np.all(np.diff(stream.times_s) >= 0)
    mean_rate = len(stream) / stream.times_s[-1]
    if name == "flash_crowd":
        assert mean_rate > base  # la pointe ajoute du trafic
    else:
        assert mean_rate == pytest.approx(base, rel=0.12)


def test_flash_crowd_peak_is_about_factor_times_base():
    spec = SCENARIOS["flash_crowd"]  # start 5 s, ramp 3 s, hold 6 s, factor 5
    stream = generate_stream(spec, 40.0, 6000, REQUESTS, seed=2)
    before = rate_in(stream, 0.0, 5.0)
    peak = rate_in(stream, 9.0, 13.0)  # plein plateau
    assert before == pytest.approx(40.0, rel=0.3)
    assert peak == pytest.approx(200.0, rel=0.25)


def test_spikes_alternate_between_two_rates():
    spec = SCENARIOS["spikes"]
    stream = generate_stream(spec, 40.0, 20000, REQUESTS, seed=5)
    counts, _ = np.histogram(stream.times_s, bins=np.arange(0, stream.times_s[-1], 1.0))
    # debits par seconde : deux regimes (autour de 20 et 80 req/s), donc forte dispersion
    assert counts.std() > 15


def test_request_mix_matches_weights_and_sizes_are_positive():
    stream = generate_stream(SCENARIOS["medium"], 50.0, 30000, REQUESTS, seed=7)
    share = {n: np.mean(stream.type_index == i) for i, n in enumerate(stream.type_names)}
    assert share["light"] == pytest.approx(0.6, abs=0.02)
    assert share["medium"] == pytest.approx(0.3, abs=0.02)
    assert share["heavy"] == pytest.approx(0.1, abs=0.02)
    assert np.all(stream.work_ms > 0)
    heavy = stream.work_ms[stream.type_index == stream.type_names.index("heavy")]
    assert heavy.mean() == pytest.approx(150.0, rel=0.05)


def test_stream_until_is_prefix_of_longer_stream():
    spec = SCENARIOS["medium"]
    short = generate_stream_until(spec, 30.0, 10.0, REQUESTS, seed=9)
    assert short.times_s[-1] <= 10.0 and len(short) > 150
    longer = generate_stream(spec, 30.0, len(short) + 50, REQUESTS, seed=9)
    assert np.array_equal(short.times_s, longer.times_s[: len(short)])


def test_trace_replay_scales_to_requested_rate(tmp_path):
    trace = make_synthetic_trace(tmp_path / "trace.csv", n_functions=6, minutes=300, seed=1)
    counts = load_trace_counts(trace)
    assert len(counts) == 300 and counts.sum() > 0
    spec = ScenarioSpec("t", "trace", 1.0, {"trace_file": str(trace), "bin_s": 0.5})
    stream = generate_stream(spec, 40.0, 6000, REQUESTS, seed=3)
    assert len(stream) / stream.times_s[-1] == pytest.approx(40.0, rel=0.25)
    same = generate_stream(spec, 40.0, 6000, REQUESTS, seed=3)
    assert np.array_equal(stream.times_s, same.times_s)


def test_committed_sample_trace_is_readable():
    counts = load_trace_counts("data/traces/azure_sample_synthetic.csv")
    assert len(counts) == 1440


def test_invalid_inputs():
    with pytest.raises(ValueError):
        generate_stream(SCENARIOS["low"], 0.0, 10, REQUESTS, seed=0)
    with pytest.raises(ValueError):
        ScenarioSpec("x", "unknown", 1.0)
