import asyncio
import sys
import time

import httpx

URL = sys.argv[1] if len(sys.argv) > 1 else 'http://127.0.0.1:8001'
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
TYPE = sys.argv[3] if len(sys.argv) > 3 else 'heavy'
TYPES = ['light', 'medium', 'heavy']


def type_for(i):
    return TYPES[i % 3] if TYPE == 'mixed' else TYPE


async def one(client, i):
    typ = type_for(i)
    t = time.perf_counter()
    r = await client.post(f'{URL}/process', json={'type': typ, 'size': 1.0})
    total = (time.perf_counter() - t) * 1000
    body = r.json() if r.status_code == 200 else {}
    return i, typ, r.status_code, round(total), body.get('wait_ms'), body.get('service_ms')


async def watch(client, stop):
    while not stop.is_set():
        s = (await client.get(f'{URL}/state')).json()
        print('   state:', 'queue_ms', s['queue_ms'], '| in_flight', s['in_flight'],
              '| waiting', s['waiting'], '| battery', s['battery'], '| depleted', s['depleted'])
        await asyncio.sleep(0.5)


async def main():
    async with httpx.AsyncClient(timeout=60) as client:
        stop = asyncio.Event()
        watcher = asyncio.create_task(watch(client, stop))
        results = await asyncio.gather(*[one(client, i) for i in range(N)])
        stop.set()
        await watcher
        print('id  type  statut  total_ms  wait_ms  service_ms')
        for r in sorted(results, key=lambda x: x[0]):
            print(*r)


asyncio.run(main())