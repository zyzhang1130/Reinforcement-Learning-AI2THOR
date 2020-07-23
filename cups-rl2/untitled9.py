from gym_ai2thor.envs.ai2thor_env import AI2ThorEnv
import ai2thor.controller
controller = ai2thor.controller.Controller()
multi_agent_event = controller.step(dict(action='Initialize', gridSize=0.25, agentCount=2))
N_EPISODES = 20
env = AI2ThorEnv()
max_episode_length = env.task.max_episode_length
for episode in range(N_EPISODES):
    state = env.reset()
    for step_num in range(max_episode_length):
        action = env.action_space.sample()
        controller.step(action, agentId=0)
        controller.step(action, agentId=1)
