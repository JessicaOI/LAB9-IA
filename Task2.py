import gym
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import check_is_fitted
from collections import deque
import random
import time

N = 10  # Asegúrate de reemplazar 10 con el tamaño máximo de un estado en tu problema
step_counter = 0

def flatten_observation(observation):
    if isinstance(observation, tuple) and len(observation) > 0:
        observation = observation[0]  # Extraer la matriz NumPy de la observación
    return np.array(observation).flatten()[:10]

# Parámetros del entorno
env = gym.make('Boxing-v0', render_mode='human')
obs = env.reset()

# Hiperparámetros del agente
epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.99
gamma = 0.99
alpha = 0.1
max_memory = 50000
batch_size = 32

# Inicializar el modelo de RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, max_depth=5, n_jobs=-1)
memory = deque(maxlen=max_memory)

# Lista para almacenar los pasos de cada episodio
episode_steps = []

# Función de actualización del modelo
def update_model(memory, model):
    if len(memory) < batch_size:
        return

    samples = random.sample(memory, batch_size)
    X = np.zeros((batch_size, 10))  # Cambia el número de columnas a 2
    y = []

    for idx, (state, action, reward, next_state, done) in enumerate(samples):
        if done:
            target = reward
        else:
            try:
                check_is_fitted(model)  # Verificar si el modelo ha sido ajustado
                target = reward + gamma * np.max(model.predict([next_state])[0])
            except:
                target = 0  # Usar un valor predeterminado (puede ajustarlo según sus necesidades)

        try:
            current_state_prediction = model.predict([state])[0]
        except:
            current_state_prediction = np.zeros(env.action_space.n)
        current_state_prediction[action] = target

        X[idx] = state
        y.append(current_state_prediction)

    model.fit(X, y)


# Bucle principal para iterar sobre el entorno
while True:
    # Incrementar el contador de pasos
    step_counter += 1
    # Renderizar el entorno en modo 'human'
    env.render()

    # Agregar una pausa para dar tiempo al entorno de renderizar
    time.sleep(0.01)

    # Aplanar la observación
    flat_obs = flatten_observation(obs)
    
    # Verificar la forma de la observación aplanada
    if flat_obs.shape != (10,):
        continue

    # Tomar una acción con un valor aleatorio de 'epsilon-greedy'
    if np.random.rand() < epsilon or len(memory) < batch_size:
        action = env.action_space.sample()
    else:
        action = np.argmax(model.predict([flat_obs])[0])

    # Ejecutar la acción y obtener la observación, recompensa, estado "done" e información
    next_obs, reward, done, info, _ = env.step(action)
    

    # Aplanar la próxima observación
    flat_next_obs = flatten_observation(next_obs)

    # Almacenar el paso en la lista de pasos del episodio
    episode_steps.append((flat_obs, action, reward, flat_next_obs, done))

    # Actualizar la memoria y el modelo solo después de un cierto número de pasos
    memory.append((flat_obs, action, reward, flat_next_obs, done))
    if step_counter % 100 == 0:
        update_model(memory, model)

    # Actualizar la observación actual
    obs = next_obs

    # Verificar si el episodio ha terminado
    if done:
        # Imprimir el número de pasos del episodio
        print(f"Episode finished after {len(episode_steps)} steps")

        # Actualizar el valor de 'epsilon'
        epsilon = max(epsilon * epsilon_decay, epsilon_min)

        # Reiniciar la lista de pasos del episodio
        episode_steps = []

        # Reiniciar el entorno y obtener la observación inicial
        obs = env.reset()