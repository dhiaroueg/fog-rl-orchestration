# chaque nœud Fog exécute ce programme pour exposer son état et ses métriques,
# afin que l’orchestrateur puisse savoir comment il se comporte.
import os
import asyncio
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Response, HTTPException
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, generate_latest

from pydantic import BaseModel

##### Params du noeud (viennent de profiles.yaml via les var d'env) #####
NODE_NAME = os.getenv('NODE_NAME', 'fogA')
CONCURRENCY = int(os.getenv('CONCURRENCY', '1'))
SPEED_FACTOR = float(os.getenv('SPEED_FACTOR', '1.0'))
MAX_QUEUE = int(os.getenv('MAX_QUEUE', '20'))
P_IDLE = float(os.getenv('P_IDLE', '2.7'))
P_MAX = float(os.getenv('P_MAX', '6.4'))
HAS_BATTERY = os.getenv('BATTERY', 'false').lower() == 'true'

BATTERY_CAPACITY_J = float(os.getenv('BATTERY_CAPACITY_J', '200'))   # energie utilisable au-dela du repos
RECHARGE_W = float(os.getenv('RECHARGE_W', '1.0'))                   # recharge lente
BATTERY_RESTART = float(os.getenv('BATTERY_RESTART', '0.5'))         # niveau de reprise apres epuisement

WORK_MS = {'light': 5.0, 'medium': 40.0, 'heavy': 150.0}   #Combien de temps de CPU demande chaque type de requête
SAMPLE_S = 0.1    # periode d'echantillonnage pour que le nœud « regarde » sa propre charge
ALPHA = 0.3   # lissage exponentiel de cpu_load

##### Etat interne du noeud (rempli par les prochaines sous-etapes) #####
st = {
    'waiting': 0,        # requetes admises, pas encore en cours de traitement
    'in_flight': 0,      # requetes en cours de traitement
    'pending_ms': 0.0,   # travail restant estime des requetes non terminees
    'cpu_load': 0.0,     # in_flight / CONCURRENCY, lisse
    'battery': 1.0,      # 1.0 si secteur ou batterie pleine
    'battery_j': BATTERY_CAPACITY_J,  # energie restante en joules
    'depleted': False,   # True si la batterie est epuisee
    'energy_j': 0.0,     # energie cumulee en joules
}

pool: Optional[ProcessPoolExecutor] = None # ProcessPoolExecutor permet d'exécuter le calcul CPU dans un processus séparé.
sem = asyncio.Semaphore(CONCURRENCY)  #Il limite le nombre de requêtes qui passent simultanément (par exemple CONCURRENCY=2) dans selon fifo .

##### Prometheus metrics #####
registry = CollectorRegistry()

g_queue = Gauge( # une valeur numerique qui peut monter et descendre, pour suivre l’état du nœud.
    'node_queue_ms', # le nom de la métrique.
    'Travail en attente (ms)', # sa description
    ['node'], #une étiquette qui identifie le nœud
    registry=registry # le registre dans lequel la métrique est enregistré
)
g_load = Gauge('node_cpu_load', 'Charge (fraction de concurrence occupee)', ['node'], registry=registry)
g_inflight = Gauge('node_in_flight', 'Requetes en cours', ['node'], registry=registry)
g_battery = Gauge('node_battery', 'Niveau de batterie (0 a 1)', ['node'], registry=registry)
g_energy = Gauge('node_energy_joules', 'Energie cumulee (J)', ['node'], registry=registry)
c_requests = Counter('node_requests_total', 'Requetes par issue', ['node', 'status'], registry=registry)

def burn(ms: float) -> None:
    """Cette fonction exécute une boucle qui consomme du temps CPU pendant une durée cible au lieu de rester attendre avec sleep()."""
    """C’est ce qui permet à --cpus de ralentir réellement le calcul"""
    end = time.process_time() + ms / 1000.0
    while time.process_time() < end:
        pass

async def sampler():
    """Met a jour cpu_load (lisse) et l'energie, toutes les SAMPLE_S secondes."""
    last = time.perf_counter() # enregistre le temps actuel avec un compteur précis

    while True:
        await asyncio.sleep(SAMPLE_S) 
        now = time.perf_counter() #Calculer le temps écoulé
        dt, last = now - last, now
        inst = min(1.0, st['in_flight'] / CONCURRENCY) #Estimer l'occupation du worker
        st['cpu_load'] = ALPHA * inst + (1 - ALPHA) * st['cpu_load'] # Mettre à jour cpu_load
        power = P_IDLE + (P_MAX - P_IDLE) * inst # Estimer la puissance 
        st['energy_j'] += power * dt # Estimerl'énergie
        if HAS_BATTERY:
            drain = (power - P_IDLE) * dt          # seul le surplus au-dela du repos vide la batterie (énergie retirée de la batterie)
            gain = RECHARGE_W * dt #énergie ajoutée par la recharge
            st['battery_j'] = min(BATTERY_CAPACITY_J, max(0.0, st['battery_j'] - drain + gain))
            st['battery'] = st['battery_j'] / BATTERY_CAPACITY_J
            if st['battery'] <= 0.0:
                st['depleted'] = True
            elif st['depleted'] and st['battery'] >= BATTERY_RESTART:
                st['depleted'] = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pool
    pool = ProcessPoolExecutor(max_workers=CONCURRENCY)
    loop = asyncio.get_running_loop()
    # echauffement : demarre les processus avant la premiere vraie requete
    await asyncio.gather(*[loop.run_in_executor(pool, burn, 1.0) for _ in range(CONCURRENCY)])
    task = asyncio.create_task(sampler())
    yield
    task.cancel()
    pool.shutdown(wait=False, cancel_futures=True)

app = FastAPI(lifespan=lifespan)

class Req(BaseModel):
    type: str                          # light, medium ou heavy
    size: float = 1.0                  # multiplicateur du travail
    request_id: Optional[str] = None   # pour la decoupe en morceaux, plus tard
    chunk_id: Optional[int] = None

def snapshot():
    return {
        'node': NODE_NAME,
        'queue_ms': round(st['pending_ms'], 2),
        'cpu_load': round(st['cpu_load'], 3),
        'in_flight': st['in_flight'],
        'waiting': st['waiting'],
        'max_queue': MAX_QUEUE,
        'concurrency': CONCURRENCY,
        'battery': round(st['battery'], 3),
        'depleted': st['depleted'],
        'energy_j': round(st['energy_j'], 3),
    }

@app.get('/state')
async def state():
    return snapshot()

@app.get('/metrics')
async def metrics():
    g_queue.labels(NODE_NAME).set(st['pending_ms']) 
    # Pour le nœud;associe la valeur de pending_ms à la métrique node_queue_ms portant l’étiquette node="NODE_NAME".
    g_load.labels(NODE_NAME).set(st['cpu_load'])
    g_inflight.labels(NODE_NAME).set(st['in_flight'])
    g_battery.labels(NODE_NAME).set(st['battery'])
    g_energy.labels(NODE_NAME).set(st['energy_j'])
    return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

@app.post('/process')
# vérifie si le nœud peut l'accepter, attend si nécessaire, exécute le travail et renvoie le résultat.
async def process(req: Req):
    if req.type not in WORK_MS: #Vérifier le type de la requête
        raise HTTPException(400, f'type inconnu : {req.type}')
     #Vérifier si le worker peut accepter la requête
    if st['waiting'] >= MAX_QUEUE or (HAS_BATTERY and st['depleted']):
        c_requests.labels(NODE_NAME, 'rejected').inc()
        raise HTTPException(503, 'file pleine ou batterie epuisee')

    work = WORK_MS[req.type] * req.size / SPEED_FACTOR
    t_arrival = time.perf_counter() # mémorise le moment où le worker commence à gérer cette requête
    st['waiting'] += 1 #ajoute une requête au nombre de requêtes en attente
    st['pending_ms'] += work 
    started = False
    try:
        async with sem:      # attend son tour (FIFO)
            st['waiting'] -= 1
            st['in_flight'] += 1
            started = True
            t_start = time.perf_counter()
            await asyncio.get_running_loop().run_in_executor(pool, burn, work)
            t_end = time.perf_counter()
    finally:
        st['pending_ms'] = max(0.0, st['pending_ms'] - work)
        if started:
            st['in_flight'] -= 1
        else:
            st['waiting'] -= 1

    c_requests.labels(NODE_NAME, 'ok').inc()
    return {
        'node': NODE_NAME,
        'type': req.type,
        'work_ms': round(work, 2),
        'wait_ms': round((t_start - t_arrival) * 1000, 2),
        'service_ms': round((t_end - t_start) * 1000, 2),
        'request_id': req.request_id,
        'chunk_id': req.chunk_id,
    }