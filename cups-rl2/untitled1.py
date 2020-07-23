import ai2thor.controller
controller = ai2thor.controller.Controller()
controller.start()

# agentCount specifies the number of agents in a scene
multi_agent_event = controller.step(dict(action='Initialize', gridSize=0.25, agentCount=2))

# print out agentIds
for e in multi_agent_event.events:
    print(e.metadata['agentId'])

# move the second agent ahead, agents are 0-indexed
multi_agent_event = controller.step(dict(action='MoveAhead', agentId=1)) 