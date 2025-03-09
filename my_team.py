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
                first='RunningMudkipsOffensiveAgent', second='RunningMudkipsDefensiveAgent', num_training=0):
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
    - Share vision between agents
    - Offensive agent: 
        - If approaching end of time, just come back
        - Attack larger clusters of food
    - Defensive agent: 
        - Protect larger clusters of food
"""


class RunningMudkipsAgent(CaptureAgent):
    graph = {}
    nodes = []
    shortest_actions = {}

    def __init__(self, index, time_for_computing=0.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state: GameState):
        super().register_initial_state(game_state)
        self.is_red = game_state.is_on_red_team(self.index)
        self.team_idxs = game_state.get_red_team_indices(
        ) if self.is_red else game_state.get_blue_team_indices()
        self.enemy_idxs = game_state.get_blue_team_indices(
        ) if self.is_red else game_state.get_red_team_indices()
        self.width, self.height = game_state.data.layout.width, game_state.data.layout.height

        border_x = self.width // 2 - 1 if self.is_red else self.width // 2
        self.border = [(border_x, y) for y in range(self.height)
                       if not game_state.has_wall(border_x, y)]

        self.total_food = len(game_state.get_red_food().as_list())

        if RunningMudkipsAgent.graph == {}:
            RunningMudkipsAgent.graph = create_graph(game_state)
            RunningMudkipsAgent.nodes = set(RunningMudkipsAgent.graph.keys())
            RunningMudkipsAgent.shortest_actions = all_pairs_first_actions(
                RunningMudkipsAgent.graph)

    def choose_action(self, game_state):
        pass

    def _get_enemy_location_distribution(self, agent_location, game_state: GameState, nodes, enemy_fn):
        """
        Provides the probability that an enemy is at a food specific location
        Enemy probability is uniform for all nodes within a specific range of the true enemy location

        TODO: Improve this heuristic
        If the enemy location is known, use manual risk probability = 1 - normalized distance from location to food??
        """
        distances = game_state.get_agent_distances()

        true_distances = np.array(
            [RunningMudkipsAgent.shortest_actions[agent_location, loc][0] for loc in nodes])

        enemy_distances = [distances[enemy_index]
                           for enemy_index in self.enemy_idxs]

        enemy_dist_probs = [np.zeros(len(nodes)) for _ in range(2)]
        for i in range(2):
            # TODO: Add better logic for being scared
            enemy_agent = game_state.get_agent_state(self.enemy_idxs[i])
            if enemy_fn(enemy_agent):
                enemy_dist_probs[i] = np.array([GameState.get_distance_prob(
                    x, enemy_distances[i]) for x in true_distances])

        for i in range(2):
            pos = game_state.get_agent_position(self.enemy_idxs[i])
            if pos is not None and enemy_fn(enemy_agent):
                # TODO: Define better risk measure
                nodes_from_enemy = np.array(
                    [RunningMudkipsAgent.shortest_actions[pos, loc][0] for loc in nodes])
                enemy_dist_probs[i] = 1 - \
                    (nodes_from_enemy / (self.width + self.height))

        enemy_distribution = enemy_dist_probs[0] + \
            enemy_dist_probs[1] - enemy_dist_probs[0] * enemy_dist_probs[1]

        return {loc: (prob, dist) for loc, prob, dist in zip(nodes, enemy_distribution, true_distances)}

    def _get_best_node(self, nodes_dict, alpha, optim_fn=min):
        max_possible_distance = self.width + self.height

        def calculate_score(item):
            coords, (risk, distance) = item
            normalized_distance = distance / max_possible_distance
            return (alpha * normalized_distance) + ((1-alpha) * risk)

        best_item = optim_fn(nodes_dict.items(), key=calculate_score)
        best_coords = best_item[0]
        best_score = calculate_score(best_item)

        return best_coords, best_score


class RunningMudkipsOffensiveAgent(RunningMudkipsAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # food proximity to enemy probable risk, higher means cares more about food proximity
        self.ALPHA = 0.2
        # border proximity to enemy probable risk, higher means cares more about border proximity
        self.BETA = 0.6
        # capsule proximity to enemy probable risk, higher means cares more about capsule proximity
        self.GAMMA = 0.8
        # threshold to pickup capsule
        self.DELTA = 0.1
        # weights to prioritise num_carrying against food collection risk, higher means cares more about num_carrying
        self.RHO = 0.6
        # threshold to return to border
        self.EPS = 0.3

    def register_initial_state(self, game_state: GameState):
        super().register_initial_state(game_state)

    def choose_action(self, game_state: GameState):
        """
        General definitions:
            - Risk from nodes: Probability distribution of enemy location at nodes: 
                either using noisy distance or true enemy position
        """
        loc = game_state.get_agent_position(self.index)
        agent = game_state.get_agent_state(self.index)
        food = game_state.get_blue_food().as_list(
        ) if self.is_red else game_state.get_red_food().as_list()

        def is_not_scared_ghost(agent):
            return not agent.is_pacman and agent.scared_timer == 0

        # Option 1: Food node picking option
        """
        Considerations:
            - Distance to various food locations
            - Risk from food
        """
        # Treat scared agents as food
        # enemy_agents = [game_state.get_agent_state(
        #     idx) for idx in self.enemy_idxs]
        # enemy_locations = [agent.get_position(
        # ) for agent in enemy_agents if agent.scared_timer > 0 and agent.get_position()]

        enemy_dist_for_food = self._get_enemy_location_distribution(
            loc, game_state, food, is_not_scared_ghost)
        min_risk_food, risk_food = self._get_best_node(
            enemy_dist_for_food, self.ALPHA, min)
        destination = min_risk_food
        # print(f'Agent: {loc}, Min Risk Food: {min_risk_food}, Risk: {risk}')

        if not agent.is_pacman:
            return RunningMudkipsAgent.shortest_actions[loc, destination][1]

        # Option 2: Going back option
        """
        Considerations: 
            - If carrying + returned == total - 2
            - Number of points carrying
            - Risk from food
            - TODO: Proximity to border
            - TODO: Time remaining
        """
        collected_max_points = self.__collected_max_points(game_state)
        enemy_dist_for_border = self._get_enemy_location_distribution(
            loc, game_state, self.border, is_not_scared_ghost)
        min_risk_border, risk_border = self._get_best_node(
            enemy_dist_for_border, self.BETA, min)

        carrying = agent.num_carrying
        normalized_carrying = carrying / (carrying + len(food) - 2)
        return_factor = self.RHO * \
            normalized_carrying + (1-self.RHO) * risk_food

        if collected_max_points or (return_factor > self.EPS and carrying > 0):
            destination = min_risk_border

        # Option 3: Capsule picking option
        """
        Considerations:
            - Proximity to dot
            - Risk from capsule
        """
        capsules = game_state.get_blue_capsules(
        ) if self.is_red else game_state.get_red_capsules()
        if len(capsules) > 0 and not collected_max_points:
            enemy_dist_for_capsule = self._get_enemy_location_distribution(
                loc, game_state, capsules, is_not_scared_ghost)
            min_risk_capsule, risk_capsule = self._get_best_node(
                enemy_dist_for_capsule, self.GAMMA, min)
            if risk_capsule <= self.DELTA:
                destination = min_risk_capsule

        # Option 4: Somthing using the possible actions

        return RunningMudkipsAgent.shortest_actions[loc, destination][1]

    def __collected_max_points(self, game_state: GameState):
        team_agents = [game_state.get_agent_state(
            idx) for idx in self.team_idxs]

        food_collected = team_agents[0].num_carrying + team_agents[0].num_returned + \
            team_agents[1].num_carrying + team_agents[1].num_returned

        return food_collected >= self.total_food - 2


class RunningMudkipsDefensiveAgent(RunningMudkipsAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def register_initial_state(self, game_state: GameState):
        super().register_initial_state(game_state)
        start = 0 if self.is_red else self.width // 2
        end = self.width // 2 if self.is_red else self.width
        self.team_area = [(x, y) for x in range(start, end) for y in range(
            0, self.height) if (x, y) in RunningMudkipsAgent.nodes]
        self.medoid = self._get_medoid(self.team_area)

    def _get_medoid(self, nodes):
        n = len(nodes)
        dist_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i+1, n):
                dist = RunningMudkipsAgent.shortest_actions[nodes[i], nodes[j]][0]
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist

        total_distances = np.sum(dist_matrix, axis=1)
        medoid_idx = np.argmin(total_distances)

        return nodes[medoid_idx]

    def choose_action(self, game_state: GameState):
        """
        Get noisy estimate of of enemy pacman(s), go to most probable location
        """

        def is_pacman(agent):
            return agent.is_pacman

        loc = game_state.get_agent_position(self.index)
        agent = game_state.get_agent_state(self.index)
        food = game_state.get_red_food().as_list(
        ) if self.is_red else game_state.get_blue_food().as_list()

        enemy_agents = [game_state.get_agent_state(
            idx) for idx in self.enemy_idxs]

        # Option 1: If no pacman, go to center of grid
        if not enemy_agents[0].is_pacman and not enemy_agents[1].is_pacman:
            self.medoid = self._get_medoid(food)
            return RunningMudkipsAgent.shortest_actions[loc, self.medoid][1]

        # Option 2: If enemy is visible, go to it
        for i in range(2):
            pos = game_state.get_agent_position(self.enemy_idxs[i])
            if enemy_agents[i].is_pacman and pos:
                return RunningMudkipsAgent.shortest_actions[loc, pos][1]

        # Option 3: Go to highest probability pacman
        enemy_dist = self._get_enemy_location_distribution(
            loc, game_state, self.team_area, is_pacman)
        max_prob_loc, prob = self._get_best_node(
            enemy_dist, 0.5, max)
        destination = max_prob_loc

        # Option 4: Prevent the pacman from getting to an energy dot

        # Option 5: If I am scared, dont go to pacman?

        return RunningMudkipsAgent.shortest_actions[loc, destination][1]
