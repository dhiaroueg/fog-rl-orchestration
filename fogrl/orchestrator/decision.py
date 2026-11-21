"""Moteur de decision : budget de temps, repli least-connections et disjoncteur de politique (EF-07).

Une decision RL tourne dans un thread avec un delai maximal (``budget_ms``, 20 ms). Si la
politique depasse le delai, leve une exception, renvoie une action invalide ou un noeud
indisponible, la requete est routee par ``least-connections`` (la regle de repli du cahier)
et la decision est marquee ``fallback``. Apres ``failure_threshold`` echecs consecutifs, la
politique est suspendue ``cooldown_s`` secondes (disjoncteur) pour ne pas faire attendre
chaque requete 20 ms.
"""

from __future__ import annotations

import asyncio
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..policies.base import Policy
from ..policies.baselines import LeastConnectionsPolicy
from ..state_spec import DecisionContext, NodeState, Normalization, RequestState, pad_nodes


def warm_up_policy(
    policy: Policy,
    slot_ids: Sequence[str],
    normalization: Normalization,
    nodes: Sequence[NodeState],
    n: int = 5,
) -> None:
    """Quelques decisions a blanc : la premiere inference torch depasserait sinon le budget de 20 ms."""

    request = RequestState("medium", 40.0, 300.0)
    padded = pad_nodes(nodes, slot_ids, normalization)
    for _ in range(max(n, 0)):
        policy.select(padded, request, DecisionContext(0.0, 10.0))
    policy.reset(None)


class NoNodeAvailable(RuntimeError):
    """Aucun noeud ne peut recevoir la requete."""


@dataclass(frozen=True)
class Decision:
    index: int  # emplacement (dans slot_ids) du noeud choisi
    node_id: str
    fallback: bool
    reason: str  # "" si la politique a decide ; sinon timeout | error:<Type> | invalid_action | breaker_open
    decision_ms: float
    policy: str


class DecisionEngine:
    def __init__(
        self,
        policy: Policy,
        slot_ids: Sequence[str],
        normalization: Normalization,
        *,
        budget_ms: float = 20.0,
        failure_threshold: int = 20,
        cooldown_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.policy = policy
        self.slot_ids = tuple(slot_ids)
        self.normalization = normalization
        self.budget_s = budget_ms / 1000.0
        self.failure_threshold = failure_threshold
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.fallback_policy = LeastConnectionsPolicy()
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="decide")
        self._consecutive_failures = 0
        self._open_until = 0.0
        self.decisions = 0
        self.fallbacks: Counter = Counter()
        self.fault: Optional[tuple[str, float]] = None  # panne injectee de l'agent : ("error", 0) | ("slow", delai_s)

    # -------------------------------------------------------------- gestion

    def swap(self, policy: Policy, slot_ids: Sequence[str]) -> None:
        """Remplace la politique a chaud (EF-08) : l'echange est une simple affectation.

        Une nouvelle politique est saine : la panne injectee eventuelle est levee.
        """

        self.slot_ids = tuple(slot_ids)
        self.policy = policy
        self._consecutive_failures = 0
        self._open_until = 0.0
        self.fault = None

    def reset(self, seed: Optional[int] = None) -> None:
        self.policy.reset(seed)
        self._consecutive_failures = 0
        self._open_until = 0.0
        self.decisions = 0
        self.fallbacks.clear()
        self.fault = None

    def set_fault(self, mode: str, delay_ms: float = 0.0) -> None:
        """Simule la defaillance de l'agent (test EF-07 : « suppression du modele pendant un run »).

        ``error`` : la politique leve une exception a chaque decision ; ``slow`` : elle met ``delay_ms`` ms a repondre
        (au-dela du budget, la decision se replie) ; ``none`` leve la panne. Le repli est alors visible dans les
        metriques ``fog_decision_fallback_total`` et dans le journal (``fallback``, ``fallback_reason``).
        """

        if mode == "none":
            self.fault = None
        elif mode == "error":
            self.fault = ("error", 0.0)
        elif mode == "slow":
            if delay_ms <= 0:
                raise ValueError("delay_ms doit etre > 0 pour le mode slow")
            self.fault = ("slow", delay_ms / 1000.0)
        else:
            raise ValueError(f"Mode de panne inconnu : {mode!r} (none, error ou slow)")
        self._consecutive_failures = 0
        self._open_until = 0.0

    def _select(self, policy: Policy, padded, request: RequestState, ctx: DecisionContext) -> int:
        """Appel de la politique, avec la panne injectee eventuelle (execute dans le thread de decision)."""

        fault = self.fault
        if fault is not None:
            if fault[0] == "slow":
                time.sleep(fault[1])
            else:
                raise RuntimeError("agent indisponible (panne injectee)")
        return policy.select(padded, request, ctx)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    @property
    def breaker_open(self) -> bool:
        return self.clock() < self._open_until

    def warm_up(self, nodes: Sequence[NodeState], n: int = 5) -> None:
        warm_up_policy(self.policy, self.slot_ids, self.normalization, nodes, n)

    # ------------------------------------------------------------- decision

    async def decide(
        self, nodes: Sequence[NodeState], request: RequestState, ctx: DecisionContext
    ) -> Decision:
        started = time.perf_counter()
        padded = pad_nodes(nodes, self.slot_ids, self.normalization)
        policy = self.policy
        index: Optional[int] = None
        reason = ""
        if self.breaker_open:
            reason = "breaker_open"
        else:
            try:
                if policy.needs_budget or self.fault is not None:
                    future = asyncio.get_running_loop().run_in_executor(
                        self._executor, self._select, policy, padded, request, ctx
                    )
                    chosen = await asyncio.wait_for(future, timeout=self.budget_s)
                else:
                    chosen = policy.select(padded, request, ctx)
                chosen = int(chosen)
                if 0 <= chosen < len(padded) and padded[chosen].available:
                    index = chosen
                else:
                    reason = "invalid_action"
            except asyncio.TimeoutError:
                reason = "timeout"
            except Exception as exc:  # mesure corrompue, modele defaillant... : on se replie
                reason = f"error:{type(exc).__name__}"

        if index is None:
            if reason != "breaker_open":
                self._register_failure()
            try:
                index = self.fallback_policy.select(padded, request, ctx)
            except ValueError as exc:
                raise NoNodeAvailable(str(exc)) from exc
            self.fallbacks[reason] += 1
        else:
            self._consecutive_failures = 0
        self.decisions += 1
        return Decision(
            index=index,
            node_id=padded[index].node_id,
            fallback=bool(reason),
            reason=reason,
            decision_ms=(time.perf_counter() - started) * 1000.0,
            policy=policy.name,
        )

    def _register_failure(self) -> None:
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._open_until = self.clock() + self.cooldown_s
            self._consecutive_failures = 0
