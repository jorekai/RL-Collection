from collections import namedtuple

# we define a experience tuple for easy convention of MDP notation
# sometimes in literature people use the equivalent word observation for state
# state == observation
Experience = namedtuple("Experience", "state action reward next_state done")
