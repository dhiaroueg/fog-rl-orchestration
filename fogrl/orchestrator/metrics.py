"""Metriques Prometheus de l'orchestrateur (registre propre a chaque instance)."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily

LATENCY_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0)
DECISION_BUCKETS = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.02, 0.05, 0.1)


class _StateCollector:
    """Jauges lues au moment du scrape : vue des noeuds, debit entrant, politique active."""

    def __init__(self, router) -> None:
        self.router = router

    def collect(self):
        router = self.router
        rate = GaugeMetricFamily("fog_orchestrator_arrival_rate_rps", "Debit recent des requetes (req/s)")
        rate.add_metric([], router.rate.current(router.clock()))
        yield rate
        info = GaugeMetricFamily("fog_policy_info", "Politique active (valeur 1)", labels=["policy", "kind"])
        info.add_metric([router.engine.policy.name, "rl" if router.engine.policy.needs_budget else "baseline"], 1.0)
        yield info
        breaker = GaugeMetricFamily("fog_policy_breaker_open", "1 si la politique est suspendue (repli permanent)")
        breaker.add_metric([], 1.0 if router.engine.breaker_open else 0.0)
        yield breaker
        families = {
            "up": GaugeMetricFamily("fog_monitor_node_up", "Noeud disponible selon l'orchestrateur", labels=["node"]),
            "rtt": GaugeMetricFamily("fog_monitor_rtt_ms", "Latence reseau mesuree (EMA, ms)", labels=["node"]),
            "queue": GaugeMetricFamily("fog_monitor_queue_ms", "Attente estimee (ms), derniere mesure", labels=["node"]),
            "cpu": GaugeMetricFamily("fog_monitor_cpu_load", "Charge CPU, derniere mesure", labels=["node"]),
            "battery": GaugeMetricFamily("fog_monitor_battery_fraction", "Batterie virtuelle", labels=["node"]),
            "in_flight": GaugeMetricFamily("fog_monitor_in_flight", "Requetes envoyees et non terminees", labels=["node"]),
        }
        for node_id, view in router.monitor.views.items():
            families["up"].add_metric([node_id], 1.0 if router.monitor.is_available(node_id) else 0.0)
            families["rtt"].add_metric([node_id], view.rtt_ms)
            families["queue"].add_metric([node_id], view.queue_ms)
            families["cpu"].add_metric([node_id], view.cpu_load)
            families["in_flight"].add_metric([node_id], router.in_flight.get(node_id, 0))
            if view.battery_fraction is not None:
                families["battery"].add_metric([node_id], view.battery_fraction)
        yield from families.values()


class RouterMetrics:
    def __init__(self, router) -> None:
        self.registry = CollectorRegistry()
        self.registry.register(_StateCollector(router))
        self.requests = Counter(
            "fog_route_requests_total", "Requetes routees", ["policy", "node", "type", "status"], registry=self.registry
        )
        self.deadline_missed = Counter(
            "fog_route_deadline_missed_total",
            "Requetes rejetees ou terminees apres leur echeance",
            ["policy", "type"],
            registry=self.registry,
        )
        self.latency = Histogram(
            "fog_route_latency_seconds",
            "Latence d'envoi a la reponse du noeud",
            ["policy", "type"],
            buckets=LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.decision = Histogram(
            "fog_decision_seconds",
            "Temps de decision de la politique",
            ["policy"],
            buckets=DECISION_BUCKETS,
            registry=self.registry,
        )
        self.fallbacks = Counter(
            "fog_decision_fallback_total",
            "Decisions confiees a la regle de repli",
            ["policy", "reason"],
            registry=self.registry,
        )
        self.energy = Counter(
            "fog_route_energy_joules_total", "Energie dynamique virtuelle des requetes", ["policy", "node"], registry=self.registry
        )
        self.cost = Counter(
            "fog_route_cost_total", "Cout cloud cumule", ["policy", "node"], registry=self.registry
        )

    def observe(self, policy: str, node: str, req_type: str, result: dict) -> None:
        status = "rejected" if result["rejected"] else "ok"
        self.requests.labels(policy, node, req_type, status).inc()
        self.decision.labels(policy).observe(result["decision_ms"] / 1000.0)
        if result["fallback"]:
            self.fallbacks.labels(policy, result["fallback_reason"]).inc()
        if not result["deadline_met"]:
            self.deadline_missed.labels(policy, req_type).inc()
        if not result["rejected"]:
            self.latency.labels(policy, req_type).observe(result["latency_ms"] / 1000.0)
            self.energy.labels(policy, node).inc(result["energy_j"])
            if result["cost"]:
                self.cost.labels(policy, node).inc(result["cost"])
