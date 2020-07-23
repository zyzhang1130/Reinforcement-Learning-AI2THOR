from ai2thor.controller import Controller
import random

# initialize controller with 2 agents
controller = Controller(scene='FloorPlan28', agentCount=2)

# define all the actions
actions = ['MoveAhead', 'RotateRight', 'RotateLeft']

# get random actions
action0 = random.choice(actions)
action1 = random.choice(actions)

# step with random actions
controller.step(action=action0, agentId=0)
controller.step(action=action1, agentId=1)