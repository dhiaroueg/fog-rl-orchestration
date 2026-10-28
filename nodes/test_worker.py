import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import pytest

HERE = Path(__file__).parent

# Tous les parametres sont fixes ici : les variables laissees dans votre terminal ne comptent pas.
BASE = {
    'NODE_NAME': 'testnode', 'CONCURRENCY': 1, 'SPEED_FACTOR': 1.0, 'MAX_QUEUE': 20,
    'P_IDLE': 2.7, 'P_MAX': 6.4, 'BATTERY': 'false',
    'BATTERY_CAPACITY_J': 200, 'RECHARGE_W': 1.0, 'BATTERY_RESTART': 0.5,
}

# Un seul client pour tous les tests : creer un client a chaque requete ralentit et desordonne l'envoi.
CLIENT = httpx.Client(timeout=60, limits=httpx.Limits(max_connections=100))


@pytest.fixture(scope='module', autouse=True)
def _close_client():
    yield
    CLIENT.close()


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def node():
    procs = []

    def start(**overrides):
        port = free_port()
        env = os.environ.copy()
        env.update({k: str(v) for k, v in {**BASE, **overrides}.items()})
        p = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'worker:app', '--port', str(port), '--log-level', 'warning'],
            cwd=HERE, env=env)
        procs.append(p)
        url = f'http://127.0.0.1:{port}'
        for _ in range(100):
            try:
                CLIENT.get(url + '/state', timeout=1)
                return url
            except Exception:
                time.sleep(0.1)
        raise RuntimeError('le noeud ne demarre pas')

    yield start
    for p in procs:
        p.terminate()
        p.wait(timeout=10)


def post(url, typ, size=1.0):
    return CLIENT.post(url + '/process', json={'type': typ, 'size': size})


def state(url):
    return CLIENT.get(url + '/state', timeout=5).json()


def fire(url, types, size=1.0, stagger=0.02):
    """Envoie les requetes dans l'ordre, espacees de `stagger` s, et attend toutes les reponses."""
    with ThreadPoolExecutor(len(types)) as ex:
        futs = []
        for t in types:
            futs.append(ex.submit(post, url, t, size))
            time.sleep(stagger)
        return [f.result() for f in futs]


def test_state_format(node):
    s = state(node())
    for key in ['node', 'queue_ms', 'cpu_load', 'in_flight', 'waiting', 'max_queue',
                'concurrency', 'battery', 'depleted', 'energy_j']:
        assert key in s
    assert s['node'] == 'testnode'
    assert s['queue_ms'] == 0 and s['in_flight'] == 0


def test_metrics_exposed(node):
    url = node()
    text = CLIENT.get(url + '/metrics', timeout=5).text
    assert 'node_queue_ms' in text and 'node_battery' in text


def test_unknown_type_is_400(node):
    assert post(node(), 'foo').status_code == 400


def test_fifo_order(node):
    url = node(CONCURRENCY=1)
    rs = fire(url, ['heavy'] * 4, size=3, stagger=0.05)       # 450 ms par requete
    assert all(r.status_code == 200 for r in rs)
    waits = [r.json()['wait_ms'] for r in rs]
    assert waits == sorted(waits)          # ordre d'arrivee respecte
    assert waits[0] < 50 and waits[2] > 600


def test_reject_when_queue_full(node):
    url = node(CONCURRENCY=1, MAX_QUEUE=2)
    rs = fire(url, ['heavy'] * 6, size=3, stagger=0.02)       # les 6 arrivent en 100 ms, avant la fin de la 1re
    codes = [r.status_code for r in rs]
    assert codes.count(200) == 3 and codes.count(503) == 3


def test_state_responsive_during_compute(node):
    url = node()
    with ThreadPoolExecutor(1) as ex:
        f = ex.submit(post, url, 'heavy', 3)                   # 450 ms de calcul
        time.sleep(0.1)
        t = time.perf_counter()
        s = state(url)
        dt = time.perf_counter() - t
        f.result()
    assert dt < 0.25
    assert s['in_flight'] == 1


def test_queue_ms_tracks_and_returns_to_zero(node):
    url = node(CONCURRENCY=1)
    with ThreadPoolExecutor(3) as ex:
        futs = []
        for _ in range(3):
            futs.append(ex.submit(post, url, 'heavy'))
            time.sleep(0.02)
        time.sleep(0.05)
        mid = state(url)
        for f in futs:
            f.result()
    assert mid['queue_ms'] >= 300
    time.sleep(0.3)
    end = state(url)
    assert end['queue_ms'] == 0 and end['in_flight'] == 0 and end['waiting'] == 0


def test_concurrency_two(node):
    url = node(CONCURRENCY=2)
    rs = fire(url, ['heavy'] * 4, size=3, stagger=0.01)
    waits = sorted(r.json()['wait_ms'] for r in rs)
    assert waits[1] < 100                      # deux demarrent ensemble
    assert waits[2] > 300                      # la troisieme attend un emplacement


def test_battery_drains_refuses_and_recovers(node):
    # petite batterie et recharge rapide pour un test court
    url = node(BATTERY='true', BATTERY_CAPACITY_J=4, RECHARGE_W=2.0, MAX_QUEUE=60)
    assert state(url)['battery'] == 1.0 and state(url)['depleted'] is False
    with ThreadPoolExecutor(25) as ex:
        futs = [ex.submit(post, url, 'heavy') for _ in range(25)]
        deadline = time.time() + 15
        while time.time() < deadline and not state(url)['depleted']:
            time.sleep(0.1)
        assert state(url)['depleted'] is True              # epuisee
        assert post(url, 'light').status_code == 503       # refus
        for f in futs:
            f.result()                                     # les requetes acceptees vont au bout
    deadline = time.time() + 15
    while time.time() < deadline and state(url)['depleted']:
        time.sleep(0.2)
    assert state(url)['depleted'] is False                 # reprise
    assert post(url, 'light').status_code == 200


def test_node_without_battery_never_depletes(node):
    url = node(BATTERY='false', CONCURRENCY=2, MAX_QUEUE=60)
    rs = fire(url, ['heavy'] * 10, stagger=0.0)
    assert all(r.status_code == 200 for r in rs)
    s = state(url)
    assert s['battery'] == 1.0 and s['depleted'] is False