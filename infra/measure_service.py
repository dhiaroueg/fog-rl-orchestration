# import csv
# import datetime
# import statistics
# import time

# import httpx

# NODES = {'fogA': 'http://127.0.0.1:8001', 'fogB': 'http://127.0.0.1:8002'}   # tunnel SSH ouvert
# TYPES = ['light', 'medium', 'heavy']
# N, WARMUP = 30, 3

# rows = []
# with httpx.Client(timeout=60) as c:
#     for node, url in NODES.items():
#         for typ in TYPES:
#             vals = []
#             for i in range(N + WARMUP):
#                 r = c.post(url + '/process', json={'type': typ, 'size': 1.0})
#                 if i >= WARMUP:
#                     vals.append(r.json()['service_ms'])
#                 time.sleep(0.2)                     # laisse le nœud redevenir libre
#             vals.sort()
#             med, p95 = statistics.median(vals), vals[int(0.95 * len(vals)) - 1]
#             print(f'{node:6} {typ:7} mediane {med:7.1f} ms | p95 {p95:7.1f} ms | min {vals[0]:7.1f} | max {vals[-1]:7.1f}')
#             for v in vals:
#                 rows.append([node, typ, 0.0, v, '', '', datetime.datetime.now().isoformat(timespec='seconds')])

# with open('docs/mesures_service_vm1.csv', 'w', newline='', encoding='utf-8') as f:
#     w = csv.writer(f)
#     w.writerow(['node', 'req_type', 'load_level', 'service_time_ms', 'net_rtt_ms', 'cpu_pct', 'timestamp'])
#     w.writerows(rows)
# print('ecrit docs/mesures_service_vm1.csv')


import csv
import datetime
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

# usage : python infra\measure_service.py vm2 fogC=8003
vm = sys.argv[1]
nodes = dict(a.split('=') for a in sys.argv[2:])
TYPES = ['light', 'medium', 'heavy']
N, WARMUP = 30, 3

rows = []
with httpx.Client(timeout=120, limits=httpx.Limits(max_connections=50)) as c:
    for node, port in nodes.items():
        url = f'http://127.0.0.1:{port}'
        conc = c.get(url + '/state').json()['concurrency']
        for typ in TYPES:
            for mode, k, level in (('seul', 1, 0.0), ('plein', conc, 1.0)):
                if mode == 'plein' and conc == 1:
                    continue                      # rien a mesurer en plus avec une concurrence de 1
                vals = []
                for i in range(N + WARMUP):
                    with ThreadPoolExecutor(k) as ex:
                        futs = [ex.submit(c.post, url + '/process', json={'type': typ, 'size': 1.0}) for _ in range(k)]
                        rs = [f.result() for f in futs]
                    if i >= WARMUP:
                        vals += [r.json()['service_ms'] for r in rs if r.status_code == 200]
                    time.sleep(0.3)               # laisse le noeud redevenir libre
                vals.sort()
                med = statistics.median(vals)
                p95 = vals[max(0, int(0.95 * len(vals)) - 1)]
                print(f'{node:6} {typ:7} {mode:6} mediane {med:7.1f} ms | p95 {p95:7.1f} | min {vals[0]:7.1f} | max {vals[-1]:7.1f} | n={len(vals)}')
                now = datetime.datetime.now().isoformat(timespec='seconds')
                rows += [[node, typ, level, v, '', '', now] for v in vals]

out = f'docs/mesures_service_{vm}.csv'
with open(out, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['node', 'req_type', 'load_level', 'service_time_ms', 'net_rtt_ms', 'cpu_pct', 'timestamp'])
    w.writerows(rows)
print('ecrit', out)