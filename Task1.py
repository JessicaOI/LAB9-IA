import numpy as np
import gym

env = gym.make("FrozenLake-v1", is_slippery=True)

#implementar el algoritmo Q-learning

# Hiperparámetros
alpha = 0.1
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.999
episodes = 5000

# Inicializar la tabla Q
q_table = np.zeros([env.observation_space.n, env.action_space.n])

for episode in range(episodes):
    state = env.reset()
    done = False
    
    while not done:
        # Selección de acción (política epsilon-greedy)
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Tomar acción y observar resultado
        result = env.step(action)
        next_state, reward, done, _ = result[0], result[1], result[2], result[3]

        # Convertir state y action a enteros
        state = int(state)
        action = int(action)

        # Actualizar la tabla Q
        q_value = q_table[state, action]
        next_q_value = np.max(q_table[next_state])
        new_q_value = q_value + alpha * (reward + gamma * next_q_value - q_value)
        q_table[state, action] = new_q_value

        # Actualizar el estado
        state = next_state

    # Reducción de epsilon para exploración y explotación equilibradas
    epsilon *= epsilon_decay




#probar el agente entrenado

state = env.reset()
done = False
env.render()

while not done:
    action = np.argmax(q_table[state])
    state, _, done, _ = env.step(action)
    env.render()
