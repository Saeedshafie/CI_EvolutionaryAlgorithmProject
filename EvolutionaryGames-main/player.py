import pygame
import numpy as np

from nn import NeuralNetwork
from config import CONFIG


class Player():

    def __init__(self, mode, control=False):

        self.control = control  # if True, playing mode is activated. else, AI mode.
        self.pos = [100, 275]   # position of the agent
        self.direction = -1     # if 1, goes upwards. else, goes downwards.
        self.v = 0              # vertical velocity
        self.g = 9.8            # gravity constant
        self.mode = mode        # game mode

        # neural network architecture (AI mode)
        layer_sizes = self.init_network(mode)

        self.nn = NeuralNetwork(layer_sizes)
        self.fitness = 0  # fitness of agent

    def move(self, box_lists, camera, events=None):

        if len(box_lists) != 0:
            if box_lists[0].x - camera + 60 < self.pos[0]:
                box_lists.pop(0)

        mode = self.mode

        # manual control
        if self.control:
            self.get_keyboard_input(mode, events)

        # AI control
        else:
            agent_position = [camera + self.pos[0], self.pos[1]]
            self.direction = self.think(mode, box_lists, agent_position, self.v)

        # game physics
        if mode == 'gravity' or mode == 'helicopter':
            self.v -= self.g * self.direction * (1 / 60)
            self.pos[1] += self.v

        elif mode == 'thrust':
            self.v -= 6 * self.direction
            self.pos[1] += self.v * (1 / 40)

        # collision detection
        is_collided = self.collision_detection(mode, box_lists, camera)

        return is_collided

    # reset agent parameters
    def reset_values(self):
        self.pos = [100, 275]
        self.direction = -1
        self.v = 0

    def get_keyboard_input(self, mode, events=None):

        if events is None:
            events = pygame.event.get()

        if mode == 'helicopter':
            self.direction = -1
            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                self.direction = 1

        elif mode == 'thrust':
            self.direction = 0
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.direction = 1
            elif keys[pygame.K_DOWN]:
                self.direction = -1

        for event in events:
            if event.type == pygame.KEYDOWN:

                if mode == 'gravity' and event.key == pygame.K_SPACE:
                    self.direction *= -1

    def init_network(self, mode):

        # you can change the parameters below
        # Probably No Change is Needed

        layer_sizes = None
        if mode == 'gravity':
            layer_sizes = [6, 20, 1]
        elif mode == 'helicopter':
            layer_sizes = [6, 20, 1]
        elif mode == 'thrust':
            layer_sizes = [6, 20, 1]
        return layer_sizes


    def think(self, mode, box_lists, agent_position, velocity):
        import math

        # In This Function we Need to Create the Input Vector that we Want to Give to Our NN.
        # This Happens by FeedForwarding the Surrounding Data to NN through The Vector.

        # Given Examples:
        # mode example: 'helicopter'
        # box_lists: an array of `BoxList` objects
        # agent_position example: [600, 250]
        # velocity example: 7


        # Cause the box_lists can be Empty in the Begining, We need a Counter
        direction = -1
        box_Counter = 0
        if mode == 'helicopter' or mode == 'gravity':
            while 1:
                if len(box_lists):
                    if box_lists[box_Counter].x > agent_position[0]:                    # If the the Box is Ahead(Not Passed)
                        x0 = agent_position[0] - box_lists[box_Counter].x               # Horizantal Distance
                        y0 = agent_position[1] - box_lists[box_Counter].gap_mid         # Vertical Distance(Gap!)
                        if len(box_lists) > 1:                                          # More Boxes than 1 box is Ahead
                            x1 = agent_position[0] - box_lists[box_Counter + 1].x       # Horizantal Distance
                            y1 = agent_position[1] - box_lists[box_Counter + 1].gap_mid # Vertical Distance(Gap!)
                        else:
                            x1 = box_lists[box_Counter].x + 200                         # Might Not Need This Else
                            y1 = y0
                        break
                    else:
                        box_Counter += 1
                else:                                       # Default Status Where box_lists is EMPTY!
                    x0 = x1 = 1280
                    y0 = y1 = 360
                    break

            # Create the Input Vector and NORMALIZE!
            # Divided to There Maximum Possible Value to Be Normalized
            Input_Vector = [[velocity / 10],
                            [math.sqrt((x0 ** 2) + (y0 ** 2)) / 1468.6],
                            [x0 / 1280],
                            [y0 / 720],
                            [x1 / 1280],
                            [y1 / 720]]
            output = self.nn.forward(Input_Vector) # Passing the InputVector to our FeedForward Network
            # Decide the Direction
            if output > 0.5:
                return 1        # This is Direction
            else:
                return -1

        # Thrust Mode
        # Difference in OutPut Creation
        elif mode == 'thrust':
            input_layer = np.zeros((6, 1))
            input_layer[0, 0] = velocity / 10
            input_layer[1, 0] = agent_position[1] / CONFIG["HEIGHT"]
            if len(box_lists) >= 1:
                input_layer[2, 0] = (agent_position[1] - box_lists[0].gap_mid) / CONFIG["HEIGHT"]
                input_layer[3, 0] = (agent_position[0] - box_lists[0].x) / CONFIG["WIDTH"]
            if len(box_lists) >= 2:
                input_layer[4, 0] = (agent_position[1] - box_lists[1].gap_mid) / CONFIG["HEIGHT"]
            input_layer[5, 0] =((CONFIG["HEIGHT"] - agent_position[1]) / CONFIG["HEIGHT"])*-1
            output = self.nn.forward(input_layer)
            if output > 0.7:
                direction = 1
            elif output > 0.35:
                direction = 0
            return direction

    def collision_detection(self, mode, box_lists, camera):
        if mode == 'helicopter':
            rect = pygame.Rect(self.pos[0], self.pos[1], 100, 50)
        elif mode == 'gravity':
            rect = pygame.Rect(self.pos[0], self.pos[1], 70, 70)
        elif mode == 'thrust':
            rect = pygame.Rect(self.pos[0], self.pos[1], 110, 70)
        else:
            rect = pygame.Rect(self.pos[0], self.pos[1], 50, 50)
        is_collided = False

        if self.pos[1] < -60 or self.pos[1] > CONFIG['HEIGHT']:
            is_collided = True

        if len(box_lists) != 0:
            box_list = box_lists[0]
            for box in box_list.boxes:
                box_rect = pygame.Rect(box[0] - camera, box[1], 60, 60)
                if box_rect.colliderect(rect):
                    is_collided = True

        return is_collided
