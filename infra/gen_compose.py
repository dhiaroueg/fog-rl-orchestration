import sys

import yaml

# un port d'ecoute sur la VM par noeud
PORTS = {'fogA': 8001, 'fogB': 8002, 'fogC': 8003, 'cloud': 8004, 'fogD': 8005, 'fogE': 8006}
# valeurs facultatives du profil -> variable d'environnement du worker
OPTIONAL = {'max_queue': 'MAX_QUEUE', 'battery_capacity_j': 'BATTERY_CAPACITY_J',
            'recharge_w': 'RECHARGE_W', 'battery_restart': 'BATTERY_RESTART'}

# Lire les arguments et le profil
vm = sys.argv[1]
path = sys.argv[2] if len(sys.argv) > 2 else 'experiments/configs/profiles.yaml'
cfg = yaml.safe_load(open(path, encoding='utf-8'))

# Parcourir les nœuds et choisir ceux de la VM
services = {}
for name, p in cfg['nodes'].items():
    if p['vm'] != vm:
        continue
    # Construire les variables d'environnement
    env = {
        'NODE_NAME': name,
        'CONCURRENCY': p['concurrency'],
        'SPEED_FACTOR': p['speed'],
        'P_IDLE': p.get('p_idle', 0),
        'P_MAX': p.get('p_max', 0),
        'BATTERY': 'true' if p.get('battery') else 'false',
        'NETEM_DELAY_MS': p['netem_delay_ms'],
        'NETEM_JITTER_MS': p['netem_jitter_ms'],
    }
    for key, var in OPTIONAL.items():
        if key in p:
            env[var] = p[key]
    # Décrire le conteneur
    services[name] = {
        'image': 'fog-worker',
        'container_name': name,
        'init': True,
        'cpus': p['cpus'],
        'mem_limit': f"{p['mem_mb']}m",
        'cap_add': ['NET_ADMIN'],
        'environment': {k: str(v) for k, v in env.items()},
        'ports': [f'{PORTS[name]}:8000'],
    }

if not services:
    sys.exit(f'aucun noeud pour {vm} dans {path}')

out = f'infra/{vm}.yml'
with open(out, 'w', encoding='utf-8') as f:
    f.write(f'# Genere par infra/gen_compose.py depuis {path} : ne pas modifier a la main.\n')
    yaml.safe_dump({'services': services}, f, sort_keys=False)
print('ecrit', out, '->', list(services))