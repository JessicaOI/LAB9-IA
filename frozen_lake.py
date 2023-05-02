import warnings
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    import gym
    from gym.envs.toy_text.frozen_lake import generate_random_map

    # Crear un entorno FrozenLake 4x4 predeterminado
    env1 = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)

    # Crear un entorno FrozenLake 8x8 aleatorio
    random_map = generate_random_map(size=8)
    env2 = gym.make('FrozenLake-v1', desc=random_map)

    def play_frozen_lake(env):
        env.reset()
        env.render()
        done = False
        while not done:
            action = int(input("Ingrese la acción (0: Izquierda, 1: Abajo, 2: Derecha, 3: Arriba): "))
            obs, reward, done, _, info = env.step(action)  # Añade un guion bajo (_) aquí
            env.render()
            if done:
                if reward:
                    print("¡Has ganado!")
                else:
                    print("Has caído en un agujero.")
                env.reset()

    # Jugar en el entorno 4x4
    play_frozen_lake(env1)

    # Jugar en el entorno 8x8 aleatorio
    play_frozen_lake(env2)
