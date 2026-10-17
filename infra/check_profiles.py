import sys
import yaml

cfg = yaml.safe_load(open('experiments/configs/profiles.yaml', encoding='utf-8'))
ok = True

print('=== Budget par VM ===')
for vm, v in cfg['vms'].items():
    nodes = [n for n, p in cfg['nodes'].items() if p['vm'] == vm]
    cpu = sum(cfg['nodes'][n]['cpus'] for n in nodes)
    mem = sum(cfg['nodes'][n]['mem_mb'] for n in nodes)
    cpu_max = v['vcpus'] - v['reserved_cpus']
    mem_max = v['ram_mb'] - v['reserved_mem_mb']
    good = cpu <= cpu_max and mem <= mem_max
    ok = ok and good
    print(f"{vm}: {nodes} CPU {cpu}/{cpu_max}  RAM {mem}/{mem_max} Mo  -> {'OK' if good else 'DEPASSE'}")

print()
print('=== Temps de service theorique (ms), modele a remplacer par la calibration ===')
print('formule : travail / speed / min(1, cpus / requetes_en_cours)')
for name, p in cfg['nodes'].items():
    row = []
    for t, r in cfg['request_types'].items():
        alone = r['work_ms'] / p['speed'] / min(1, p['cpus'] / 1)
        full = r['work_ms'] / p['speed'] / min(1, p['cpus'] / p['concurrency'])
        row.append(f"{t}: {alone:.0f} seul / {full:.0f} plein")
    print(f"{name}: " + ' | '.join(row))

print()
missing = [n for n, p in cfg['nodes'].items() if p.get('cost_per_req', 1) is None]
if missing:
    print('A definir : cost_per_req pour', missing)

sys.exit(0 if ok else 1)