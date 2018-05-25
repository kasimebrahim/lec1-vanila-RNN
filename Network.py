from easydict import EasyDict as Dict
from functions import softmax
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

    """
    takes a list of indices of words in a sentence as an input and computes the forward
    chain of the neural net. 
    :return: tuple of states and outputs
    """
    def forward(self, input):
        input_length = len(input)
        # stores each state from initial to every state after an input is inserted
        states = np.zeros((input_length+1, self.hidden_size, 1))
        # stores output of the network after each input is computed
        outputs = np.zeros((input_length, self.input_size, 1))

        for time_t in range(0, input_length):
            prev_state = states[time_t]

            current_input = np.zeros((self.input_size, 1))
            current_input[input[time_t]] = 1;

            current_state = np.tanh(self.w_first.dot(current_input)+self.w_reverse.dot(prev_state))

            current_output = softmax(self.w_second.dot(current_state))

            states[time_t+1] = current_state
            outputs[time_t] = current_output
        # based on the calculated outputs we need to take the most probable output as prediction
        predictions =  [np.argmax(o) for o in outputs]

        return states, outputs, predictions

    def cost(self):
        pass


params = Dict({
    "dimensions": [3, 2, 3]
})
network = Network(params)
print(network.w_first.shape, network.w_second.shape, network.w_reverse.shape)

state, output, prediction = network.forward([2, 0, 1])
print(state, "\n --------")
print(output, "\n --------")
print(prediction)