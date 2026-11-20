import csv

from infra.measure_service import write_run_measurements


def test_write_run_measurements_records_all_fog_nodes(tmp_path):
    results = [{
        "request_id": "r-1",
        "status": "completed",
        "selected_node": "fogA",
        "observed_at": "2026-11-20T21:00:00+00:00",
        "observed_states": {
            "fogA": {
                "queue_ms": 4.0,
                "cpu_load": 0.25,
                "net_rtt_ms": 12.0,
                "battery": 0.9,
                "depleted": False,
                "in_flight": 1,
                "waiting": 0,
            },
            "fogB": {
                "queue_ms": 8.0,
                "cpu_load": 0.5,
                "net_rtt_ms": 20.0,
                "battery": 1.0,
                "depleted": False,
                "in_flight": 0,
                "waiting": 1,
            },
        },
        "state_errors": {"fogC": "connection refused"},
        "worker_result": {"wait_ms": 3.0, "service_ms": 5.0, "type": "light"},
        "elapsed_ms": 25.0,
        "deadline_ms": 100.0,
        "deadline_met": True,
    }]
    arrivals = [{
        "request_id": "r-1",
        "type": "light",
        "t_arrival_ms": 0.0,
        "deadline_ms": 100.0,
    }]

    output = write_run_measurements(
        results,
        arrivals,
        tmp_path / "measurements.csv",
        scenario="flash_crowd",
    )

    with output.open(newline="", encoding="utf-8") as measurement_file:
        rows = list(csv.DictReader(measurement_file))

    assert [row["node"] for row in rows] == ["fogA", "fogB", "fogC"]
    assert {row["scenario"] for row in rows} == {"flash_crowd"}
    assert rows[0]["dispatch_status"] == "selected"
    assert rows[0]["service_time_ms"] == "5.0"
    assert rows[0]["cpu_pct"] == "25.0"
    assert rows[1]["dispatch_status"] == "not_selected"
    assert rows[1]["service_time_ms"] == ""
    assert rows[2]["node_status"] == "unavailable"
    assert rows[2]["error"] == "connection refused"
