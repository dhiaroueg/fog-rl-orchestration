import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import httpx

from state_spec import NodeState, RequestState, encode_state

NODES = {'fogA': 'http://127.0.0.1:8001', 'fogB': 'http://127.0.0.1:8002'}

# lance la rafale en arriere-plan, puis observe l'etat pendant ce temps
burst = subprocess.Popen(
    [sys.executable, str(ROOT / 'nodes' / 'burst.py'), NODES['fogA'], '15', 'heavy'],
    stdout=subprocess.DEVNULL)
time.sleep(0.3)

for _ in range(16):
    nodes = []
    for name, url in NODES.items():
        s = httpx.get(url + '/state', timeout=2).json()
        nodes.append(NodeState(name, s['queue_ms'], s['cpu_load'], 0.0, s['battery'], not s['depleted']))
    obs, mask = encode_state(nodes, RequestState('heavy', 150.0, 1000.0), 10.0)
    print('fogA', obs[:4].round(2), '| fogB', obs[4:8].round(2), '| masque', mask)
    time.sleep(0.5)

burst.wait()
