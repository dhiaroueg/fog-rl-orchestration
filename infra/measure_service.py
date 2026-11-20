from __future__ import annotations

import csv
import datetime
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx

FOG_NODES = ("fogA", "fogB", "fogC")
RUN_MEASUREMENT_FIELDS = (
    "scenario",
    "timestamp",
    "request_id",
    "simulated_arrival_ms",
    "node",
    "request_type",
    "request_status",
    "node_status",
    "dispatch_status",
    "queue_ms",
    "cpu_pct",
    "net_rtt_ms",
    "battery_fraction",
    "depleted",
    "in_flight",
    "waiting",
    "worker_wait_ms",
    "service_time_ms",
    "elapsed_ms",
    "deadline_ms",
    "deadline_met",
    "error",
)


def write_run_measurements(
    results: Sequence[Mapping[str, Any]],
    arrivals: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    scenario: str,
) -> Path:
    """Écrit les états des trois nœuds Fog observés pendant une exécution."""
    requests_by_id = {str(row["request_id"]): row for row in arrivals}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=RUN_MEASUREMENT_FIELDS)
        writer.writeheader()
        for result in results:
            request_id = str(result["request_id"])
            request = requests_by_id.get(request_id, {})
            observed_states = result.get("observed_states", {})
            state_errors = result.get("state_errors", {})
            dispatch_failures = result.get("dispatch_failures", {})
            selected_node = result.get("selected_node")
            worker_result = result.get("worker_result", {})

            for node in FOG_NODES:
                state = observed_states.get(node, {})
                state_error = state_errors.get(node, "")
                dispatch_error = dispatch_failures.get(node, "")
                if state_error:
                    node_status = "unavailable"
                elif state.get("depleted", False):
                    node_status = "depleted"
                elif state:
                    node_status = "available"
                else:
                    node_status = "unknown"

                is_selected = node == selected_node
                cpu_load = state.get("cpu_load")
                writer.writerow({
                    "scenario": scenario,
                    "timestamp": result.get("observed_at", ""),
                    "request_id": request_id,
                    "simulated_arrival_ms": request.get("t_arrival_ms", ""),
                    "node": node,
                    "request_type": request.get("type", worker_result.get("type", "")),
                    "request_status": result.get("status", ""),
                    "node_status": node_status,
                    "dispatch_status": (
                        "selected" if is_selected
                        else "failed" if dispatch_error
                        else "not_selected"
                    ),
                    "queue_ms": state.get("queue_ms", ""),
                    "cpu_pct": round(float(cpu_load) * 100, 3) if cpu_load is not None else "",
                    "net_rtt_ms": state.get("net_rtt_ms", ""),
                    "battery_fraction": state.get("battery", ""),
                    "depleted": state.get("depleted", ""),
                    "in_flight": state.get("in_flight", ""),
                    "waiting": state.get("waiting", ""),
                    "worker_wait_ms": worker_result.get("wait_ms", "") if is_selected else "",
                    "service_time_ms": worker_result.get("service_ms", "") if is_selected else "",
                    "elapsed_ms": result.get("elapsed_ms", ""),
                    "deadline_ms": request.get("deadline_ms", result.get("deadline_ms", "")),
                    "deadline_met": result.get("deadline_met", ""),
                    "error": state_error or dispatch_error,
                })
    return output


def measure_service(vm: str, nodes: Mapping[str, str]) -> None:
    """Mesure le temps de service des nœuds, indépendamment de run.py."""
    types = ("light", "medium", "heavy")
    sample_count, warmup_count = 30, 3
    rows = []
    with httpx.Client(timeout=120, limits=httpx.Limits(max_connections=50)) as client:
        for node, port in nodes.items():
            url = f"http://100.83.84.3:{port}"
            concurrency = client.get(url + "/state").json()["concurrency"]
            for task_type in types:
                for mode, parallelism, load_level in (
                    ("seul", 1, 0.0),
                    ("plein", concurrency, 1.0),
                ):
                    if mode == "plein" and concurrency == 1:
                        continue
                    values = []
                    for index in range(sample_count + warmup_count):
                        with ThreadPoolExecutor(parallelism) as executor:
                            futures = [
                                executor.submit(
                                    client.post,
                                    url + "/process",
                                    json={"type": task_type, "size": 1.0},
                                )
                                for _ in range(parallelism)
                            ]
                            responses = [future.result() for future in futures]
                        if index >= warmup_count:
                            values.extend(
                                response.json()["service_ms"]
                                for response in responses
                                if response.status_code == 200
                            )
                        time.sleep(0.3)
                    values.sort()
                    median = statistics.median(values)
                    percentile_95 = values[max(0, int(0.95 * len(values)) - 1)]
                    print(
                        f"{node:6} {task_type:7} {mode:6} "
                        f"mediane {median:7.1f} ms | p95 {percentile_95:7.1f} | "
                        f"min {values[0]:7.1f} | max {values[-1]:7.1f} | n={len(values)}"
                    )
                    now = datetime.datetime.now().isoformat(timespec="seconds")
                    rows.extend(
                        [node, task_type, load_level, value, "", "", now]
                        for value in values
                    )

    output = Path(f"docs/mesures_service_{vm}.csv")
    with output.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.writer(output_file)
        writer.writerow([
            "node", "req_type", "load_level", "service_time_ms",
            "net_rtt_ms", "cpu_pct", "timestamp",
        ])
        writer.writerows(rows)
    print("ecrit", output)


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: python infra/measure_service.py <vm> <node>=<port> [...]")
    vm = sys.argv[1]
    nodes = {}
    for argument in sys.argv[2:]:
        node, separator, port = argument.partition("=")
        if not separator or not node or not port:
            raise SystemExit(f"Argument de nœud invalide: {argument!r}; format attendu node=port")
        nodes[node] = port
    measure_service(vm, nodes)


if __name__ == "__main__":
    main()