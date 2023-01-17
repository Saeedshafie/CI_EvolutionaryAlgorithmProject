from player import Player
import numpy as np
import random, copy
from config import CONFIG


class Evolution():

    def __init__(self, mode):
        self.mode = mode

    # calculate fitness of players
    def calculate_fitness(self, players, delta_xs):
        for i, p in enumerate(players):
            p.fitness = delta_xs[i]

    def mutate(self, child):

        # child: an object of class `Player`

        
        prob = 0.9
        
        # w1
        noise = np.random.normal(0,0.3,child.nn.w[0].shape)
        rand = random.uniform(0, 1)
        if rand < prob:
            child.nn.w[0] += noise
        # w2 
        noise = np.random.normal(0,0.3,child.nn.w[1].shape)
        rand = random.uniform(0, 1)
        if rand < prob:
            child.nn.w[1] += noise
        # b1
        noise = np.random.normal(0,0.3,child.nn.b[0].shape)
        rand = random.uniform(0, 1)
        if rand < prob:
            child.nn.b[0] += noise
        # b2
        noise = np.random.normal(0,0.3,child.nn.b[1].shape)
        rand = random.uniform(0, 1)
        if rand < prob:
            child.nn.b[1] += noise
        return child

    def generate_new_population(self, num_players, prev_players=None):

        # in first generation, we create random players
        if prev_players is None:
            return [Player(self.mode) for _ in range(num_players)]

        else:
            # Q tournoment
            Q = 10
            parents_list = []
            children = []
            # TODO
            # num_players example: 150
            # prev_players: an array of `Player` objects
            for _ in range(num_players):
                random_players = random.sample(prev_players, Q)
                best_player = max(random_players, key=lambda x: x.fitness)
                parents_list.append(copy.deepcopy(best_player))
            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover
            for _ in range(num_players):
                parents = random.sample(parents_list, 2)
                child = Player('helicopter')
                # w1
                top_half = np.vsplit(parents[0].nn.w[0], 2)
                below_half = np.vsplit(parents[1].nn.w[0], 2)
                child.nn.w[0] = np.concatenate((top_half[0],below_half[1]), axis=0)
                # w2
                top_half = np.hsplit(parents[0].nn.w[1], 2)
                below_half = np.hsplit(parents[1].nn.w[1], 2)
                child.nn.w[1] = np.concatenate((top_half[0],below_half[1]), axis=1)
                # b1
                #print(parents[0].nn.b[0].shape)

                top_half = np.vsplit(parents[0].nn.b[0].reshape(parents[0].nn.b[0].shape[0], 1), 2)
                below_half = np.vsplit(parents[1].nn.b[0].reshape(parents[1].nn.b[0].shape[0], 1), 2)
                child.nn.b[0] = np.concatenate((top_half[0],below_half[1]), axis=0)
                # b2
                #top_half = np.vsplit(parents[0].nn.b[1], 2)
                #below_half = np.vsplit(parents[1].nn.b[1], 2)
                child.nn.b[1] = parents[1].nn.b[1]

                children.append(self.mutate(child))
            new_players = children
            return new_players

            # TODO (additional): a selection method other than `fitness proportionate`
            # TODO (additional): implementing crossover

    # Fitness is Already Calculated, We Have sort and Choose
    def next_population_selection(self, players, num_players):
        # num_players example: 100
        # players: an array of `Player` objects

        # Default Sort and Chose Method, First Implementation
        players.sort(key=lambda x: x.fitness, reverse=True)
        return players[: num_players]
        
        # Extra if Time
        # TODO (additional): a selection method other than `top-k`
        # TODO (additional): plotting

        
