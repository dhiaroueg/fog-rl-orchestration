import numpy as np
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.env_checker import check_env
from toy_env import ToyFogEnv

check_env(ToyFogEnv())
print("check_env OK")


def evaluate(policy, seeds=range(1000, 1030)):
    lat, acts = [], []
    for s in seeds:
        env = ToyFogEnv()
        obs, _ = env.reset(seed=s)
        done, i = False, 0
        while not done:
            a = policy(obs, i)
            obs, _, _, done, info = env.step(a)
            lat.append(info["resp"])
            acts.append(a)
            i += 1
    return round(float(np.mean(lat)), 1), round(float(np.mean(acts)), 2)


results = {}
for Algo in (PPO, DQN, A2C):
    model = Algo("MlpPolicy", ToyFogEnv(), seed=0, verbose=0)
    model.learn(100_000)
    results[Algo.__name__] = evaluate(
        lambda o, i, m=model: int(m.predict(o, deterministic=True)[0])
    )

results["Round-robin"] = evaluate(lambda o, i: i % 2)
results["Toujours rapide"] = evaluate(lambda o, i: 0)
results["Aleatoire"] = evaluate(lambda o, i: np.random.randint(2))

print("\nPolitique        (latence moyenne ms, part envoyee au noeud lent)")
for name, r in results.items():
    print(f"{name:<16} {r}")