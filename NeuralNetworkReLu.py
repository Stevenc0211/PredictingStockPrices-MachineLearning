import neuralnetworksA2

class NeuralNetworkReLU(neuralnetworksA2):

    def activation(self, weighted_sum):
        return np.tanh(weighted_sum)

    def activationDerivative(self, activation_value):
        return 1 - activation_value * activation_value
    
    def relu_activation(dataset):
        return np.maximum(dataset, 0)
