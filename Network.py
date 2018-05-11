from easydict import EasyDict as Dict
import numpy as np


class Network:
    def __str__(self):
        return "<object> RNN neural network for language model"

    def __init__(self, params):
        self.input_size = params.dimensions[0]
        self.hidden_size = params.dimensions[1]
        self.output_size = params.dimensions[2]
        self.w_first = np.random.normal(-1, 1, (self.hidden_size, self.input_size))
        self.w_second = np.random.normal(-1, 1, (self.output_size, self.hidden_size))
        self.w_reverse = np.random.normal(-1, 1, (self.hidden_size, self.hidden_size))

params = Dict({
    "dimensions": [5000, 200, 5000]
})
network = Network(params)
print(network.w_first.shape, network.w_second.shape, network.w_reverse.shape)
