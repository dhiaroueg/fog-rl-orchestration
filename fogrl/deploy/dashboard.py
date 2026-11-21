"""Tableau de bord Grafana du prototype (metriques de l'orchestrateur et des workers)."""

from __future__ import annotations

from typing import Optional

DATASOURCE = {"type": "prometheus", "uid": "prometheus"}
WINDOW = "30s"


def _target(expr: str, legend: str = "", ref: str = "A") -> dict:
    return {"datasource": DATASOURCE, "expr": expr, "legendFormat": legend, "refId": ref}


def _panel(
    panel_id: int,
    title: str,
    targets: list[dict],
    x: int,
    y: int,
    w: int = 12,
    h: int = 8,
    unit: str = "short",
    kind: str = "timeseries",
    stack: bool = False,
    minimum: Optional[float] = None,
    maximum: Optional[float] = None,
    thresholds: Optional[list[tuple[Optional[float], str]]] = None,
    description: str = "",
) -> dict:
    defaults: dict = {"unit": unit, "custom": {"lineWidth": 1, "fillOpacity": 20 if stack else 0}}
    if stack:
        defaults["custom"]["stacking"] = {"mode": "normal", "group": "A"}
    if minimum is not None:
        defaults["min"] = minimum
    if maximum is not None:
        defaults["max"] = maximum
    if thresholds:
        defaults["thresholds"] = {
            "mode": "absolute",
            "steps": [{"color": color, "value": value} for value, color in thresholds],
        }
    return {
        "id": panel_id,
        "type": kind,
        "title": title,
        "description": description,
        "datasource": DATASOURCE,
        "targets": targets,
        "gridPos": {"x": x, "y": y, "w": w, "h": h},
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {"legend": {"displayMode": "list", "placement": "bottom"}, "tooltip": {"mode": "multi"}},
    }


def build_dashboard() -> dict:
    r = WINDOW
    panels = [
        _panel(1, "Latence p95 par politique", [_target(
            f"histogram_quantile(0.95, sum by (le, policy) (rate(fog_route_latency_seconds_bucket[{r}])))", "{{policy}}")],
            0, 0, unit="s", description="Duree d'envoi au noeud a sa reponse (reseau + attente + service)."),
        _panel(2, "Latence moyenne par politique", [_target(
            f"sum by (policy) (rate(fog_route_latency_seconds_sum[{r}])) / sum by (policy) (rate(fog_route_latency_seconds_count[{r}]))",
            "{{policy}}")], 12, 0, unit="s"),
        _panel(3, "Echeances manquees (rejets inclus)", [_target(
            f"sum by (policy) (rate(fog_route_deadline_missed_total[{r}])) / sum by (policy) (rate(fog_route_requests_total[{r}]))",
            "{{policy}}")], 0, 8, unit="percentunit", minimum=0, maximum=1),
        _panel(4, "Debit entrant (req/s)", [_target("fog_orchestrator_arrival_rate_rps", "debit")], 12, 8, unit="reqps"),
        _panel(5, "Repartition des requetes par noeud", [_target(
            f'sum by (node) (rate(fog_route_requests_total{{status="ok"}}[{r}]))', "{{node}}")],
            0, 16, unit="reqps", stack=True),
        _panel(6, "Requetes rejetees par noeud", [_target(
            f'sum by (node) (rate(fog_route_requests_total{{status="rejected"}}[{r}]))', "{{node}}")],
            12, 16, unit="reqps", stack=True),
        _panel(7, "File d'attente estimee par noeud", [_target("fog_monitor_queue_ms", "{{node}}")], 0, 24, unit="ms"),
        _panel(8, "Charge CPU par noeud", [_target("fog_node_cpu_load", "{{node}}")], 12, 24,
               unit="percentunit", minimum=0, maximum=1),
        _panel(9, "Batterie virtuelle", [_target("fog_node_battery_fraction", "{{node}}")], 0, 32,
               unit="percentunit", minimum=0, maximum=1,
               thresholds=[(None, "red"), (0.25, "orange"), (0.5, "green")]),
        _panel(10, "Puissance virtuelle par noeud (W)", [_target(
            f"sum by (node) (rate(fog_node_energy_joules_total[{r}]))", "{{node}}")],
            12, 32, unit="watt", stack=True,
            description="Modele lineaire P = Pidle + (Pmax - Pidle) x U : hypothese, pas une mesure physique."),
        _panel(11, "Cout cloud (par seconde)", [_target(
            f"sum by (policy) (rate(fog_route_cost_total[{r}]))", "{{policy}}")], 0, 40),
        _panel(12, "Temps de decision p99 (budget 20 ms)", [_target(
            f"histogram_quantile(0.99, sum by (le, policy) (rate(fog_decision_seconds_bucket[{r}])))", "{{policy}}")],
            12, 40, unit="s", thresholds=[(None, "green"), (0.02, "red")]),
        _panel(13, "Decisions confiees au repli (par motif)", [_target(
            f"sum by (reason) (rate(fog_decision_fallback_total[{r}]))", "{{reason}}")], 0, 48, unit="ops"),
        _panel(14, "Noeuds disponibles (vue de l'orchestrateur)", [_target("fog_monitor_node_up", "{{node}}")],
               12, 48, minimum=0, maximum=1, kind="state-timeline"),
        _panel(15, "Latence reseau mesuree (ms)", [_target("fog_monitor_rtt_ms", "{{node}}")], 0, 56, unit="ms"),
        _panel(16, "Politique active", [_target("fog_policy_info", "{{policy}} ({{kind}})")], 12, 56, w=6, h=4,
               kind="stat"),
        _panel(17, "Disjoncteur de politique (1 = ouvert)", [_target("fog_policy_breaker_open", "disjoncteur")],
               18, 56, w=6, h=4, kind="stat", minimum=0, maximum=1),
    ]
    return {
        "uid": "fog-rl",
        "title": "Orchestration fog par RL",
        "tags": ["fog", "rl"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "5s",
        "time": {"from": "now-15m", "to": "now"},
        "panels": panels,
    }
