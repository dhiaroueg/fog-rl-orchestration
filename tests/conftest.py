"""Fixtures partagees par les tests."""

import pytest


@pytest.fixture()
def tiny_env(tmp_path):
    """Configuration a requetes tres courtes et a faible debit : les tests restent rapides."""

    (tmp_path / "requests.yaml").write_text(
        "requests:\n"
        "  light: {work_ms: 2, deadline_ms: 100, weight: 0.6}\n"
        "  medium: {work_ms: 3, deadline_ms: 300, weight: 0.3}\n"
        "  heavy: {work_ms: 5, deadline_ms: 1000, weight: 0.1}\n"
    )
    (tmp_path / "scenarios.yaml").write_text("scenarios:\n  tiny: {type: poisson, load: 0.02}\n")
    env = tmp_path / "env.yaml"
    env.write_text(
        f"nodes_file: configs/nodes/pool6.yaml\nrequests_file: {tmp_path / 'requests.yaml'}\n"
        f"scenarios_file: {tmp_path / 'scenarios.yaml'}\nconfiguration: n4\nlayout: padded\n"
        "scenarios: [tiny]\nepisode_requests: 40\nnormalization: {arrival_rate_rps: 200}\n"
        "reward: {w1: 1.0, w2: 0.5, w3: 0.3, w4: 2.0, w5: 5.0}\n"
    )
    return str(env)
