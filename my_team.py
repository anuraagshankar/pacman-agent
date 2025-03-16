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

import random
import contest.util as util
import numpy as np
from graph_utils import create_graph, all_pairs_first_actions, find_entry_points
import time
from collections import Counter


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
        self.semi_perimeter = self.width + self.height

        border_x = self.width // 2 - 1 if self.is_red else self.width // 2
        self.border = [(border_x, y) for y in range(self.height)
                       if not game_state.has_wall(border_x, y)]

        self.total_food = len(game_state.get_red_food().as_list())
        self.agent_vision = {
            idx: {jdx: None for jdx in self.enemy_idxs} for idx in self.team_idxs}

        if RunningMudkipsAgent.graph == {}:
            RunningMudkipsAgent.graph = create_graph(game_state)
            RunningMudkipsAgent.nodes = set(RunningMudkipsAgent.graph.keys())
            RunningMudkipsAgent.shortest_actions = all_pairs_first_actions(
                RunningMudkipsAgent.graph)

            RunningMudkipsAgent.location_probs = [{} for _ in range(4)]
            for idx in self.enemy_idxs:
                RunningMudkipsAgent.location_probs[idx] = self._reset_distribution(
                    game_state.get_initial_agent_position(idx))

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

        manhattan_distances = np.array(
            [manhattan_distance(agent_location, loc) for loc in nodes])
        true_distances = np.array(
            [RunningMudkipsAgent.shortest_actions[agent_location, loc][0] for loc in nodes])
        enemy_probs = np.array([[RunningMudkipsAgent.location_probs[self.enemy_idxs[i]][loc]
                                 for loc in nodes] for i in range(2)])

        enemy_distances = [distances[enemy_index]
                           for enemy_index in self.enemy_idxs]

        enemy_dist_probs = [np.zeros(len(nodes)) for _ in range(2)]
        for i in range(2):
            # TODO: Add better logic for being scared
            enemy_agent = game_state.get_agent_state(self.enemy_idxs[i])
            if enemy_fn(enemy_agent):
                enemy_dist_probs[i] = np.array([GameState.get_distance_prob(
                    x, enemy_distances[i]) for x in manhattan_distances])
                enemy_dist_probs[i] = enemy_dist_probs[i] * enemy_probs[i]

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

        scored_items = [(coords, calculate_score((coords, values)))
                        for coords, values in nodes_dict.items()]

        min_score = optim_fn(score for _, score in scored_items)
        tied_items = [(coords, score)
                      for coords, score in scored_items if score == min_score]

        best_coords, best_score = random.choice(tied_items)
        return best_coords, best_score

    def _reset_distribution(self, position):
        distribution = {node: 0 for node in RunningMudkipsAgent.graph}
        distribution[position] = 1
        return distribution

    def _reset_enemy_distribution(self, game_state: GameState):
        for idx in self.enemy_idxs:
            pos = game_state.get_agent_position(idx)
            if pos:
                RunningMudkipsAgent.location_probs[idx] = self._reset_distribution(
                    pos)

    def _update_distribution(self):
        for idx in self.enemy_idxs:
            next_distribution = {node: 0 for node in RunningMudkipsAgent.graph}

            for node, prob in RunningMudkipsAgent.location_probs[idx].items():
                if prob == 0:
                    continue

                neighbors = RunningMudkipsAgent.graph[node]
                transition_prob = prob / (len(neighbors) + 1)
                next_distribution[node] += transition_prob

                for neighbor in neighbors:
                    next_distribution[neighbor] += transition_prob

            RunningMudkipsAgent.location_probs[idx] = next_distribution

    def _update_enemy_location(self, game_state: GameState):
        for idx in self.enemy_idxs:
            loc = game_state.get_agent_position(idx)
            self.agent_vision[self.index][idx] = loc


class RunningMudkipsOffensiveAgent(RunningMudkipsAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # food proximity to enemy probable risk, higher means cares more about food proximity
        self.ALPHA = 0.3
        # border proximity to enemy probable risk, higher means cares more about border proximity
        self.BETA = 0.6
        # capsule proximity to enemy probable risk, higher means cares more about capsule proximity
        self.GAMMA = 0.8
        # threshold to pickup capsule
        self.DELTA = 0.1
        # RHO_1: prioritizes carrying amount, RHO_2: food risk, 1-RHO_1-RHO_2: proximity to border
        self.RHO_1, self.RHO_2 = 0.5, 0.3
        # Percentage of food to prioritise before return
        self.ETA = 0.5
        # threshold to return to border
        self.EPS = 0.3
        # threshold that determines that we are being chased
        self.CHASE_THRESHOLD = 1
        # threshold to disregard scared agents
        self.SCARED_THRESHOLD = 5
        # game ending threshold
        self.GAME_ENDING_THRESHOLD = 3

    def register_initial_state(self, game_state: GameState):
        super().register_initial_state(game_state)
        self.CHASE_COUNTER = {idx: 0 for idx in self.enemy_idxs}

    def choose_action(self, game_state: GameState):
        """
        General definitions:
            - Risk from nodes: Probability distribution of enemy location at nodes:
                either using noisy distance or true enemy position
        """
        self._update_enemy_location(game_state)
        self.__update_chase_counter(game_state)
        self._update_distribution()
        self._reset_enemy_distribution(game_state)
        loc = game_state.get_agent_position(self.index)
        agent = game_state.get_agent_state(self.index)
        food = game_state.get_blue_food().as_list(
        ) if self.is_red else game_state.get_red_food().as_list()

        def is_not_scared_ghost(agent):
            return not agent.is_pacman and agent.scared_timer < self.SCARED_THRESHOLD

        # Option 1: Food node picking option

        enemy_dist_for_food = self._get_enemy_location_distribution(
            loc, game_state, food, is_not_scared_ghost)
        min_risk_food, risk_food = self._get_best_node(
            enemy_dist_for_food, self.ALPHA, min)
        destination = min_risk_food

        if not agent.is_pacman or risk_food == 0:
            return random.choice(RunningMudkipsAgent.shortest_actions[loc, destination][1])

        # Option 2: Going back option
        collected_max_points = self.__collected_max_points(game_state)
        is_game_ending = self.__is_game_ending(game_state, agent)
        enemy_dist_for_border = self._get_enemy_location_distribution(
            loc, game_state, self.border, is_not_scared_ghost)
        min_risk_border, risk_border = self._get_best_node(
            enemy_dist_for_border, self.BETA, min)

        border_factor = 1 - \
            (self.__get_min_distance_to_border(loc) / self.semi_perimeter)
        carrying = agent.num_carrying
        normalized_carrying = carrying / (self.ETA * self.total_food)
        return_factor = self.RHO_1 * normalized_carrying + self.RHO_2 * \
            risk_food + (1-self.RHO_1-self.RHO_2) * border_factor

        if collected_max_points or is_game_ending or (return_factor > self.EPS and carrying > 0):
            destination = min_risk_border

        # Option 3: Capsule picking option
        """
        Considerations:
            - Proximity to dot
            - Risk from capsule
        """
        are_agents_scared = self.__are_there_scared_agents(game_state)
        capsules = game_state.get_blue_capsules(
        ) if self.is_red else game_state.get_red_capsules()
        if len(capsules) > 0 and not collected_max_points and not are_agents_scared:
            enemy_dist_for_capsule = self._get_enemy_location_distribution(
                loc, game_state, capsules, is_not_scared_ghost)
            min_risk_capsule, risk_capsule = self._get_best_node(
                enemy_dist_for_capsule, self.GAMMA, min)

            chase_val = np.max(list(self.CHASE_COUNTER.values()))

            if risk_capsule <= self.DELTA or chase_val > self.CHASE_THRESHOLD:
                destination = min_risk_capsule

        # Option 4: Return during game end

        return random.choice(RunningMudkipsAgent.shortest_actions[loc, destination][1])

    def __are_there_scared_agents(self, game_state: GameState):
        for idx in self.enemy_idxs:
            if game_state.get_agent_state(idx).scared_timer > self.SCARED_THRESHOLD:
                return True

        return False

    def __is_game_ending(self, game_state: GameState, agent):
        actions_remaining = game_state.data.timeleft // 4
        border_dist = self.__get_min_distance_to_border(agent.get_position())
        if actions_remaining - border_dist < self.GAME_ENDING_THRESHOLD:
            return True

    def __update_chase_counter(self, game_state: GameState):
        enemy_locs = {idx: game_state.get_agent_position(
            idx) for idx in self.enemy_idxs}
        for idx in self.enemy_idxs:
            if enemy_locs[idx] is None:
                self.CHASE_COUNTER[idx] = 0
            else:
                self.CHASE_COUNTER[idx] += 1

    def __collected_max_points(self, game_state: GameState):
        team_agents = [game_state.get_agent_state(
            idx) for idx in self.team_idxs]

        food_collected = team_agents[0].num_carrying + team_agents[0].num_returned + \
            team_agents[1].num_carrying + team_agents[1].num_returned

        return food_collected >= self.total_food - 2

    def __get_min_distance_to_border(self, loc):
        return np.min([RunningMudkipsAgent.shortest_actions[loc, bor][0]
                       for bor in self.border])


class RunningMudkipsDefensiveAgent(RunningMudkipsAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

        # from given probable opponent positions, picking closest vs highest probable
        # higher alpha prioritises closer nodes
        self.ALPHA = 0.3
        # percentage of food to keep food as the center
        self.BETA = 0.4
        # for how many timesteps to count last eaten food as a valid location
        self.LAST_EATEN_EXPIRY = 10
        # number of times to sample for entry points
        self.ENTRY_POINT_SAMPLES = 20
        # entry point minimum distance threshold
        self.BAD_ENTRY_POINT_THRESHOLD = 5

    def register_initial_state(self, game_state: GameState):
        super().register_initial_state(game_state)
        start = 0 if self.is_red else self.width // 2
        end = self.width // 2 if self.is_red else self.width
        self.team_area = [(x, y) for x in range(start, end) for y in range(
            0, self.height) if (x, y) in RunningMudkipsAgent.nodes]
        self.enemy_area = [
            node for node in RunningMudkipsAgent.nodes if node not in self.team_area]

        self.medoid = self._get_medoid(self.team_area)
        entry_points = self.__calc_entry_points(self.team_area)
        self.bb = entry_points
        enemy_entry_points = self.__calc_entry_points(self.enemy_area)
        self.cc = enemy_entry_points
        self.entry_points = self.__remove_bad_entry_points(
            entry_points, enemy_entry_points)
        self.dd = self.entry_points
        self.cur_entry_point = self.__calc_first_entry_point(
            game_state)  # this is the index

        self.previous_food = self.__get_food(game_state)
        self.current_food = self.__get_food(game_state)
        self.food_eaten = []
        self.time = 0
        self.oa_distribution = {node: 0 for node in RunningMudkipsAgent.graph}

    def __remove_bad_entry_points(self, team_entry_points, enemy_entry_points):
        filtered_entry_points = []
        for ep in team_entry_points:
            min_dist = np.min(
                [RunningMudkipsAgent.shortest_actions[ep, eep][0] for eep in enemy_entry_points])
            if min_dist <= self.BAD_ENTRY_POINT_THRESHOLD:
                filtered_entry_points.append(ep)

        if len(filtered_entry_points) == 0:
            return team_entry_points
        return filtered_entry_points

    def __calc_entry_points(self, area):
        roots = random.sample(area, self.ENTRY_POINT_SAMPLES)
        entry_point_list = [find_entry_points(
            RunningMudkipsAgent.graph, root, area) for root in roots]

        all_entry_points = []
        for s in entry_point_list:
            all_entry_points.extend(s)
        self.aa = all_entry_points

        frequency = Counter(all_entry_points)
        cnt = 0 if max(frequency.values()) == 1 else 1

        return sorted(
            [element for element, count in frequency.items() if count > cnt])

    def __calc_first_entry_point(self, game_state: GameState):
        loc = game_state.get_agent_position(self.index)
        distances = [RunningMudkipsAgent.shortest_actions[loc, ep][0]
                     for ep in self.entry_points]
        return np.argmin(distances)

    def __get_food(self, game_state: GameState):
        food = game_state.get_red_food() if self.is_red else game_state.get_blue_food()
        return set(food.as_list())

    def _get_medoid(self, nodes_src, nodes_dest=None):
        if nodes_dest is None:
            nodes_dest = nodes_src

        n = len(nodes_src)
        m = len(nodes_dest)
        dist_matrix = np.zeros((n, m))

        for i in range(n):
            for j in range(m):
                dist = RunningMudkipsAgent.shortest_actions[nodes_src[i],
                                                            nodes_dest[j]][0]
                dist_matrix[i, j] = dist

        total_distances = np.sum(dist_matrix, axis=1)
        medoid_idx = np.argmin(total_distances)

        return nodes_src[medoid_idx]

    def __update_food_eaten(self):
        eaten_food = self.previous_food - self.current_food
        if len(eaten_food) > 0:
            food = list(eaten_food)[0]
            self.food_eaten.append((food, self.time))
            self.__reset_distribution_for_offensive_opponent(food)

        self.__update_distribution_for_offensive_opponent()

    def choose_action(self, game_state: GameState):
        """
        Get noisy estimate of of enemy pacman(s), go to most probable location
        """
        self.time += 1
        self.current_food = self.__get_food(game_state)
        self.__update_food_eaten()

        self._update_enemy_location(game_state)
        self._reset_enemy_distribution(game_state)

        def is_pacman(agent):
            return agent.is_pacman

        loc = game_state.get_agent_position(self.index)
        agent = game_state.get_agent_state(self.index)
        food = game_state.get_red_food().as_list(
        ) if self.is_red else game_state.get_blue_food().as_list()

        enemy_agents = [game_state.get_agent_state(
            idx) for idx in self.enemy_idxs]

        # Option 1: If no pacman, go to either food or border medoid
        # TODO: Change this to based on map symmetry and food left
        if not enemy_agents[0].is_pacman and not enemy_agents[1].is_pacman:
            if len(food) < self.BETA * self.total_food:
                destination = self._get_medoid(food)
            else:
                # self.medoid = self._get_medoid(self.border)
                if loc == self.entry_points[self.cur_entry_point]:
                    self.cur_entry_point = (
                        self.cur_entry_point + 1) % len(self.entry_points)
                destination = self.entry_points[self.cur_entry_point]
            self.previous_food = self.__get_food(game_state)
            return random.choice(RunningMudkipsAgent.shortest_actions[loc, destination][1])

        # Option 2: If enemy is visible to anyone, go to it
        for i in self.team_idxs:
            for j in range(2):
                pos = self.agent_vision[i][self.enemy_idxs[j]]
                if enemy_agents[j].is_pacman and pos:
                    self.previous_food = self.__get_food(game_state)
                    return random.choice(RunningMudkipsAgent.shortest_actions[loc, pos][1])

        # Option 3: If food was eaten in the previous turn
        # TODO: Add expiry time here
        if len(self.food_eaten) > 0 and self.time - self.food_eaten[-1][1] < self.LAST_EATEN_EXPIRY:
            food, eaten_time = self.food_eaten[-1]
            loc_probs = [(node, self.oa_distribution[node])
                         for node in self.team_area if self.oa_distribution[node] > 0]

            dist = game_state.get_agent_distances()
            e_1, e_2 = dist[self.enemy_idxs[0]], dist[self.enemy_idxs[1]]
            true_dist = manhattan_distance(loc, food)

            noisy_dist = e_1 if abs(
                e_1-true_dist) < abs(e_2-true_dist) else e_2
            enemy_dist = self.__get_updated_oa_dist_with_noise(
                loc_probs, loc, noisy_dist)

        # Option 4: Go to highest probability pacman
        else:
            enemy_dist = self._get_enemy_location_distribution(
                loc, game_state, self.team_area, is_pacman)

        # TODO: Add hyperparameter here
        max_prob_loc, prob = self._get_best_node(
            enemy_dist, self.ALPHA, max)
        destination = max_prob_loc

        self.previous_food = self.__get_food(game_state)
        return random.choice(RunningMudkipsAgent.shortest_actions[loc, destination][1])

    def __get_updated_oa_dist_with_noise(self, loc_probs, loc, noisy_dist):
        # return format { node: (risk, dist) }
        distribution = {}
        for node, prob in loc_probs:
            noise_prob = GameState.get_distance_prob(
                manhattan_distance(loc, node), noisy_dist)
            distribution[node] = (
                prob*noise_prob, RunningMudkipsAgent.shortest_actions[loc, node][0])

        return distribution

    def __reset_distribution_for_offensive_opponent(self, position):
        """
        TODO: Assumption of single offensive agent
        """
        self.oa_distribution = {node: 0 for node in RunningMudkipsAgent.graph}
        self.oa_distribution[position] = 1

    def __update_distribution_for_offensive_opponent(self):
        next_distribution = {node: 0 for node in RunningMudkipsAgent.graph}

        for node, prob in self.oa_distribution.items():
            if prob == 0:
                continue

            neighbors = RunningMudkipsAgent.graph[node]
            transition_prob = prob / (len(neighbors) + 1)
            next_distribution[node] += transition_prob

            for neighbor in neighbors:
                next_distribution[neighbor] += transition_prob

        self.oa_distribution = next_distribution


def manhattan_distance(xy1, xy2):
    """Returns the Manhattan distance between points xy1 and xy2"""
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
