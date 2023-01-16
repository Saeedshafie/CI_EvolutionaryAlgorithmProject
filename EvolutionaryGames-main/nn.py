import numpy as np


class NeuralNetwork():

    def __init__(self, layer_sizes):

        # layer_sizes example: [4, 10, 2]

        # Array of Layer Sizes as the Input.
        # 3 Parameters : 0. Input Layers Size , 1. Hidden Layers Size , 2. Output Layers Size
        self.input_layer_size = layer_sizes[0]
        self.hidden_layer_size = layer_sizes[1]
        self.output_layer_size = layer_sizes[2]
        
        # Biases
        b1 = np.zeros((self.hidden_layer_size,1))
        b2 = np.zeros((self.output_layer_size,1))
        
        # Weights
        # 0 is for Center, 1 is for Margin
        w1 = np.random.normal(0, 1, size=(self.hidden_layer_size, self.input_layer_size))
        w2 = np.random.normal(0, 1, size=(self.output_layer_size, self.hidden_layer_size))

        # Create Weights and Biases Matrix
        self.b = [b1,b2]
        self.w = [w1,w2]
    

    def activation(self, x):
        
        # Sigmoid is Currently the Only Activation Function
        return 1 / (1 + np.exp(-x))

        #  If Needed Other Activation Function, Declare it in the Input and Put Cases
        ###### elif "Leaky-relu":
        ######   return np.where(x > 0, x, x * 0.01) 
        
    def forward(self, x):
        
        # x example: np.array([[0.1], [0.2], [0.3]])
        
        # R1 is for Hidden Later, R2 is for Output Layer
        R1 = self.activation(self.w[0] @ x + self.b[0])
        R2 = self.activation(self.w[1] @ R1 + self.b[1])

        return R2