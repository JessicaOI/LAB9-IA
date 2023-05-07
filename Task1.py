import gym
from gym.envs.toy_text.frozen_lake import generate_random_map
import numpy as np
import random

iterations = 300
stepsIteration = []
iterations_since_success = 0
wins = 0


def create_custom_env(slippery=True):
    custom_map = generate_random_map(size=4)
    return gym.make("FrozenLake-v1", render_mode="human", desc=custom_map, is_slippery=slippery)

env = create_custom_env(slippery=True)
q_table = np.zeros([env.observation_space.n, env.action_space.n])


#El factor de aprendizaje (alpha) y el factor de descuento (gamma) determinan cómo el agente actualiza la tabla q
alpha = 0.8
gamma = 0.95

#El valor de epsilon determina la probabilidad de que el agente tome una acción aleatoria
epsilon = 1.0
#Entre mas itere llegara a un epsilon mas bajo para reducir su aleatoridad
min_epsilon = 0.01
decay_epsilon = 0.995

for n in range(iterations):
    print(f"Iteración no. {n + 1}")
    iterations_since_success += 1

    state = env.reset()[0]


    for i in range(100):
        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        step_result = env.step(action)

        next_state = step_result[0]
        reward = step_result[1]
        done = step_result[2]

        # print("state:", state)
        # print("next_state:", next_state)
        # print("action:", action)


        #Aquí, el agente actualiza su conocimiento en la tabla q utilizando la ecuación de actualización de Q-learning. La actualización considera tanto las recompensas inmediatas como las futuras.
        q_table[state, action] = q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]) - q_table[state, action])

        state = next_state
        env.render()

        print(f"Movimiento numero: {i + 1}")

        if reward > 0:
            print("Ganó\n")
            wins += 1
            stepsIteration.append(iterations_since_success)
            iterations_since_success = 0

            # Cambiar de mapa sin reiniciar la q-table
            env = create_custom_env()

            break

        if done:
            print("Fin del juego\n")
            break

        # Imprimir la q-table en cada iteración
    print("Q-table en la iteración actual:")
    print(q_table)

    if epsilon > min_epsilon:
        epsilon *= decay_epsilon

# Imprimir la q-table final
print("\nQ-table final después de todas las iteraciones:")
print(q_table)


print(f"Número de victorias: {wins}")
for x in range(len(stepsIteration)):
    print(f"{x + 1}: {stepsIteration[x]}")

env.close()
