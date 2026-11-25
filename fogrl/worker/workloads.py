"""Charges de travail des requetes : legere, moyenne et lourde (cahier, section 6.1).

Deux modes :
    - ``burn`` (defaut, dev et CI) : occupe le CPU pendant ``work_ms`` de temps
      CPU. Sous une limite Docker ``--cpus`` le temps reel s'allonge (throttling
      CFS) comme pour un vrai calcul ;
    - ``real`` (deploiement) : vrais calculs numpy (agregation de capteurs,
      traitement d'image) et inference ONNX MobileNetV2 pour la requete lourde
      si ``ONNX_MODEL`` est fourni (sinon repli sur un calcul matriciel).
      Le nombre d'iterations est calibre au demarrage pour que ``work_ms``
      corresponde a la duree sur un coeur libre ; les durees reelles sont ensuite
      mesurees par la calibration (``fogrl.calibration``).

Avec ``emulate_speed`` le travail est divise par la vitesse du profil du noeud :
utile pour simuler l'heterogeneite quand les machines sont identiques et sans
limites Docker (poste de developpement, CI).

Executeurs : en production, chaque « serveur » du noeud est un PROCESSUS
(``executor="process"``). Des threads Python se partageraient le GIL et un noeud
a 2 serveurs n'aurait en pratique qu'un coeur. Les tests utilisent des threads.
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def burn_cpu_ms(ms: float) -> None:
    """Occupe le CPU ``ms`` millisecondes de temps CPU du thread courant."""

    if ms <= 0:
        return
    end = time.thread_time() + ms / 1000.0
    x = 1.0
    while time.thread_time() < end:
        x = x * 1.0000001 + 1e-9


class Workloads:
    def __init__(
        self,
        mode: str = "burn",
        speed: float = 1.0,
        emulate_speed: bool = False,
        onnx_path: Optional[str] = None,
    ) -> None:
        if mode not in ("burn", "real"):
            raise ValueError("mode doit etre 'burn' ou 'real'")
        if speed <= 0:
            raise ValueError("speed doit etre > 0")
        self.mode = mode
        self.speed = speed
        self.emulate_speed = emulate_speed
        self._session = None
        self._input_name = ""
        self._kernels_by_type: dict = {}
        self._kernel_ms: dict[str, float] = {}
        if mode == "real":
            self._load_onnx(onnx_path)
            self._kernels_by_type = self._build_kernels()
            self._calibrate_kernels()

    # -- initialisation ------------------------------------------------------

    def _load_onnx(self, onnx_path: Optional[str]) -> None:
        if not onnx_path or not Path(onnx_path).exists():
            log.warning("Modele ONNX absent : la requete lourde utilise un calcul matriciel de repli")
            return
        try:
            import onnxruntime as ort

            options = ort.SessionOptions()
            options.intra_op_num_threads = 1  # un serveur = un coeur
            self._session = ort.InferenceSession(
                onnx_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as exc:  # pragma: no cover - depend de l'installation
            log.warning("ONNX indisponible (%s) : repli sur un calcul matriciel", exc)
            self._session = None

    @staticmethod
    def _build_kernels() -> dict:
        rng = np.random.default_rng(0)
        sensors = rng.normal(size=2000)
        image = rng.random((96, 96))
        matrix = rng.random((96, 96))

        def light() -> None:  # agregation de capteurs
            sensors.mean(), sensors.std(), np.percentile(sensors, 95)

        def medium() -> None:  # traitement d'image : flou par sommes cumulees + gradient
            c = image.cumsum(0).cumsum(1)
            blurred = c[8:, 8:] - c[:-8, 8:] - c[8:, :-8] + c[:-8, :-8]
            np.hypot(*np.gradient(blurred))

        def heavy() -> None:  # repli sans ONNX : produit matriciel
            matrix @ matrix

        return {"light": light, "medium": medium, "heavy": heavy}

    def _calibrate_kernels(self) -> None:
        for req_type, kernel in self._kernels_by_type.items():
            kernel()  # echauffement
            start = time.thread_time()
            for _ in range(20):
                kernel()
            self._kernel_ms[req_type] = max((time.thread_time() - start) / 20 * 1000.0, 1e-4)

    # -- execution -------------------------------------------------------------

    def effective_ms(self, work_ms: float) -> float:
        return work_ms / self.speed if self.emulate_speed else work_ms

    def run(self, req_type: str, work_ms: float) -> None:
        """Execute la requete (bloquant : a lancer dans un thread ou un processus)."""

        ms = self.effective_ms(work_ms)
        if self.mode == "burn":
            burn_cpu_ms(ms)
            return
        if req_type == "heavy" and self._session is not None:
            x = np.random.default_rng().random((1, 3, 224, 224), dtype=np.float32)
            self._session.run(None, {self._input_name: x})
            return
        kernel = self._kernels_by_type[req_type]
        for _ in range(max(1, round(ms / self._kernel_ms[req_type]))):
            kernel()


# --------------------------------------------------------------------- executeurs

_CHILD: Optional[Workloads] = None


def _init_child(mode: str, speed: float, emulate_speed: bool, onnx_path: Optional[str]) -> None:
    global _CHILD
    _CHILD = Workloads(mode, speed, emulate_speed, onnx_path)


def _run_child(req_type: str, work_ms: float) -> None:
    assert _CHILD is not None, "processus enfant non initialise"
    _CHILD.run(req_type, work_ms)


def _noop() -> None:
    time.sleep(0.05)  # garde le processus occupe pour que les autres demarrent


class WorkloadExecutor:
    """Execute les requetes : ``servers`` processus (production) ou threads (tests)."""

    def __init__(
        self,
        servers: int,
        mode: str = "burn",
        speed: float = 1.0,
        emulate_speed: bool = False,
        onnx_path: Optional[str] = None,
        kind: str = "process",
    ) -> None:
        if kind not in ("process", "thread"):
            raise ValueError("executor doit etre 'process' ou 'thread'")
        self.kind = kind
        self.servers = servers
        self._pool: Executor
        if kind == "process":
            self._pool = ProcessPoolExecutor(
                max_workers=servers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_child,
                initargs=(mode, speed, emulate_speed, onnx_path),
            )
            self._submit = lambda req_type, work_ms: self._pool.submit(_run_child, req_type, work_ms)
        else:
            workloads = Workloads(mode, speed, emulate_speed, onnx_path)
            self._pool = ThreadPoolExecutor(max_workers=servers, thread_name_prefix="serve")
            self._submit = lambda req_type, work_ms: self._pool.submit(workloads.run, req_type, work_ms)

    def warm_up(self) -> None:
        """Demarre tous les processus (sinon la premiere requete paie le demarrage)."""

        if self.kind == "process":
            futures = [self._pool.submit(_noop) for _ in range(self.servers)]
            for future in futures:
                future.result(timeout=120)

    def submit(self, req_type: str, work_ms: float):
        return self._submit(req_type, work_ms)

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
