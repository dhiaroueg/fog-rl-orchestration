import numpy as np
import pytest

from fogrl.config import RewardConfig
from fogrl.reward import compute_reward

CFG = RewardConfig(w1=1.0, w2=0.5, w3=0.3, w4=2.0, w5=5.0, t_ref_ms=200.0, e_ref_j=1.0, c_ref=1.0)


def reward(latency, deadline=300.0, energy=0.0, cost=0.0, rejected=False, cfg=CFG):
    return compute_reward(cfg, latency, deadline, energy, cost, rejected)


def test_formula_matches_cahier():
    # latence 100 ms dans les delais, 2 J, cout 1 : -(100/200 + 0.5*2 + 0.3*1)
    assert reward(100.0, energy=2.0, cost=1.0) == pytest.approx(-(0.5 + 1.0 + 0.3))
    # echeance manquee : +w4
    assert reward(400.0) == pytest.approx(-(400 / 200 + 2.0))


def test_each_term_can_be_disabled_by_weight():
    latency_only = RewardConfig(w1=1.0, w2=0.0, w3=0.0, w4=0.0, w5=0.0)
    assert compute_reward(latency_only, 100.0, 50.0, 99.0, 99.0, False) == pytest.approx(-0.5)


def test_reward_decreases_with_latency_then_is_capped():
    values = [reward(l) for l in (10, 100, 300, 600, 900)]
    assert values == sorted(values, reverse=True)
    assert reward(900.0) == reward(10_000.0)  # plafond = 3 x echeance


@pytest.mark.parametrize("deadline", [100.0, 300.0, 1000.0])
def test_rejection_never_beats_any_answer_to_the_same_request(deadline):
    rejected = reward(0.0, deadline=deadline, rejected=True)
    for latency in np.linspace(0, 20 * deadline, 60):
        for energy, cost in [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0)]:
            assert rejected <= reward(latency, deadline, energy, cost) + 1e-9


def test_invalid_weights_rejected():
    with pytest.raises(ValueError):
        RewardConfig(w1=-1.0)
    with pytest.raises(ValueError):
        RewardConfig(t_ref_ms=0.0)
