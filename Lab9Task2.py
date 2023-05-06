import gym

# create the Atari Boxing environment
env = gym.make('Boxing-v0', render_mode='human')

# reset the environment and get the initial observation
observation = env.reset()

# run a loop to interact with the environment
done = False
while True:
    # render the environment
    env.render()

    # choose an action (random in this case)
    action = env.action_space.sample()

    # take a step in the environment with the chosen action
    obs= env.step(action)
    reward = env.step(action)
    done = env.step(action)
    info = env.step(action)

# close the environment
env.close()
