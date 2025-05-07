"""
Transitions are probabilistic
Defensive:
    Will go hunt when scared
"""

# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from contest.capture_agents import CaptureAgent
from contest.capture import GameState


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='NullAgent', second='NullAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class NullAgent(CaptureAgent):

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state: GameState):
        super().register_initial_state(game_state)

    def choose_action(self, game_state):
        return "Stop"

    
