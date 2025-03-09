# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util
import numpy as np
from utils import create_graph, all_pairs_first_actions
import time

from contest.capture_agents import CaptureAgent
from contest.capture import GameState
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='RunningMudkipsOffensiveAgent', second='DefensiveReflexAgent', num_training=0):
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

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = - \
            len(food_list)  # self.getScore(successor)

        # Compute distance to the nearest food

        if len(food_list) > 0:  # This should always be True,  but better safe than sorry
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food)
                               for food in food_list])
            features['distance_to_food'] = min_distance
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # Computes whether we're on defense (1) or offense (0)
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.get_agent_state(i)
                   for i in self.get_opponents(successor)]
        invaders = [
            a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.get_maze_distance(
                my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(
            self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}


"""
TODO:
    - Entire defensive agent
    - Choose energy over food if close
    - Run back condition on risk - num carrying and previous risk
    - Share vision between agents
"""


class RunningMudkipsOffensiveAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # food proximity to enemy probable risk, higher means cares more about food proximity
        self.ALPHA = 0.2

    def register_initial_state(self, game_state):
        super().register_initial_state(game_state)
        self.width, self.height = game_state.data.layout.width, game_state.data.layout.height
        self.graph = create_graph(game_state)
        self.nodes = list(self.graph.keys())
        self.shortest_actions = all_pairs_first_actions(self.graph)

    def choose_action(self, game_state: GameState):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)
        """
        Approach:
            - Get probabilities of location of enemy agents
            - Mask the probabilities where there is no food
            - Find least probable location of enemy agent and go there using the shortest path

        Additions:
            - We can go to nearby food even if agent is not too far
        """
        loc = game_state.get_agent_state(self.index).get_position()
        enemy_dist = self.__get_enemy_location_distribution(loc, game_state)

        min_risk_food, risk = self.get_best_node(enemy_dist, self.ALPHA)
        print(f'Agent: {loc}, Min Risk Food: {min_risk_food}, Risk: {risk}')
        return self.shortest_actions[loc, min_risk_food][1]

    def get_best_node(self, nodes_dict, alpha):
        max_possible_distance = self.width + self.height

        def calculate_score(item):
            coords, (risk, distance) = item
            normalized_distance = distance / max_possible_distance
            return (alpha * normalized_distance) + ((1-alpha) * risk)

        best_item = min(nodes_dict.items(), key=calculate_score)
        best_coords = best_item[0]
        best_score = calculate_score(best_item)

        return best_coords, best_score

    def __get_enemy_location_distribution(self, agent_location, game_state: GameState):
        """
        Provides the probability that an enemy is at a food specific location
        Enemy probability is uniform for all nodes within a specific range of the true enemy location

        TODO: Improve this heuristic
        If the enemy location is known, use manual risk probability = 1 - normalized distance from location to food??
        """
        is_red = game_state.is_on_red_team(self.index)
        distances = game_state.get_agent_distances()

        food = game_state.get_blue_food().as_list()  # TODO: Add blue team logic
        true_distances = np.array(
            [self.shortest_actions[agent_location, loc][0] for loc in food])

        enemy_agent_indices = game_state.get_blue_team_indices()  # TODO: Add blue team
        enemy_distances = [distances[enemy_index]
                           for enemy_index in enemy_agent_indices]

        enemy_dist_probs = [np.zeros(len(food)) for _ in range(2)]
        for i in range(2):
            if not game_state.get_agent_state(enemy_agent_indices[i]).is_pacman:
                enemy_dist_probs[i] = np.array([GameState.get_distance_prob(
                    x, enemy_distances[i]) for x in true_distances])

        for i in range(2):
            pos = game_state.get_agent_state(
                enemy_agent_indices[i]).get_position()
            if pos is not None:
                # TODO: Define better risk measure
                food_from_enemy = np.array(
                    [self.shortest_actions[pos, loc][0] for loc in food])
                enemy_dist_probs[i] = 1 - \
                    (food_from_enemy / (self.width + self.height))

        enemy_distribution = enemy_dist_probs[0] + \
            enemy_dist_probs[1] - enemy_dist_probs[0] * enemy_dist_probs[1]

        return {loc: (prob, dist) for loc, prob, dist in zip(food, enemy_distribution, true_distances)}

# for the defensive agent, is there a better logic than simply going behind the enemy?
