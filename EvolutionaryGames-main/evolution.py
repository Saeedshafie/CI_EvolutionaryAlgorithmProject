from player import Player
import numpy as np
import random, copy
from config import CONFIG
import os
import time

# For Generation of Plot File
#counter = 0

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

        self.save_fitness(players)
        self.save_fitness_average_max_min(players)

        Mode = 1

        # Default Sort and Choose Method, First Implementation
        if Mode == 1:
            players.sort(key=lambda x: x.fitness, reverse=True)
            return players[: num_players]

        # TODO (additional): a selection method other than `top-k`
        elif Mode == 2:
            # First Part : Building the First Ruler with Probability calculations
            total_fitness = 0
            for player in players:
                total_fitness += player.fitness
            probabilities = []
            for player in players:
                probabilities.append(player.fitness / total_fitness)
            # turn it to cumulative probability
            for i in range(1, len(players)):
                probabilities[i] += probabilities[i - 1]

            # Second Part Choosing Randomly on the Ruler

            results = []
            for random_number in np.random.uniform(low=0, high=1, size=num_players):
                for i, probability in enumerate(probabilities):
                    if random_number <= probability:
                        results.append(copy.deepcopy(players[i]))
                    break

        # TODO (additional): plotting
    def save_fitness(self, players):
        if not os.path.exists('fitness'):
            os.makedirs('fitness')

        f = open("fitness/" + self.mode +".txt", "a")
        for p in players:
            f.write(str(p.fitness))
            f.write(" ")
        f.write("\n")
        f.close()

    def save_fitness_average_max_min(self, players):

        players.sort(key=lambda x: x.fitness, reverse=True)

        total_sum = 0
        for player in players: total_sum = total_sum + player.fitness
        maximum = players[0].fitness
        minimum = players[len(players) - 1].fitness
        average = total_sum / len(players)

        if not os.path.exists('fitnessAverage'):
            os.makedirs('fitnessAverage')

        f = open("fitnessAverage/" + self.mode +".txt", "a")
        #f.write(str(counter))
        #f.write(" ")
        f.write(str(minimum))
        f.write(" ")
        f.write(str(maximum))
        f.write(" ")
        f.write(str(average))
        f.write(" ")
        f.write("\n")
        f.close()



