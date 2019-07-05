import sys
sys.path.append('./')

from intrabuildingtransport.mansion.utils import MansionAttribute, MansionState
from intrabuildingtransport.mansion.utils import ElevatorAction
from intrabuildingtransport.env import IntraBuildingEnv

env = IntraBuildingEnv("config.ini")
env.seed(1998)
#iteration = env.iterations
step = env.reset()
action = [ElevatorAction(-1, 1) for i in range(4)]
for i in range(100):
    next_state, reward, _, _ = env.step(action)

assert isinstance(env.attribute, MansionAttribute)
assert isinstance(env.state, MansionState)
print(env.statistics)
env.log_debug("This is a debug log")
env.log_notice("This is a notice log")
