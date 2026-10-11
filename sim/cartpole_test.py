from stable_baselines3 import PPO, DQN

for Algo in (PPO, DQN):
    m = Algo("MlpPolicy", "CartPole-v1", seed=0, verbose=0)
    m.learn(50_000)
    print(Algo.__name__, "OK")
    