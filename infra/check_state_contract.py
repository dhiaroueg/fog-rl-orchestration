import httpx

from state_spec import NodeState, RequestState, encode_state, observation_dim

NODES = {'fogA': 'http://127.0.0.1:8001', 'fogB': 'http://127.0.0.1:8002'}

nodes = []
for name, url in NODES.items():
    s = httpx.get(url + '/state', timeout=2).json()
    nodes.append(NodeState(
        node_id=name,
        queue_ms=s['queue_ms'],
        cpu_load=s['cpu_load'],
        net_rtt_ms=0.0,                      # mesuree par l'orchestrateur plus tard
        battery_fraction=s['battery'],
        available=not s['depleted'],
    ))

obs, mask = encode_state(nodes, RequestState('heavy', 150.0, 1000.0), arrival_rate_rps=10.0)
print('forme :', obs.shape, '| attendu :', observation_dim(len(nodes)))
print('min/max :', obs.min(), obs.max())
print('masque :', mask)
print(obs.round(3))