from PerceptronParameters import PerceptronParameters
import random


class Perceptron:
    parameters = PerceptronParameters
    input_neurons = []
    output_neurons = []
    weights = [[]]  # [input_neuron][output_neuron]. input_neuron = 0 is a bias.

    def __init__(self, parameters):
        print("*** Initializing ***\n")
        self.input_neurons = []
        self.output_neurons = []
        self.weights = [[]]
        self.parameters = parameters
        for i in range(parameters.input_neurons_num + 1):  # + 1, because of bias
            self.input_neurons.append(0)
            self.weights.append([])
            for o in range(parameters.output_neurons_num):
                self.weights[i].append(random.uniform(parameters.init_weights_min, parameters.init_weights_max))
        for o in range(parameters.output_neurons_num):
            self.output_neurons.append(0)

    def learn(self, learning_vectors):
        print("*** Learning ***")
        learn = True
        epoch = 0
        while learn and epoch < 20000:
            print(f'\nepoch: {epoch}')
            # random.shuffle(learning_vectors)
            are_errors = False
            for vector in learning_vectors:
                self.feedforward(vector)
                for o in range(len(self.output_neurons)):
                    cost = vector.y[o] - self.output_neurons[o]
                    if cost != 0:
                        are_errors = True
                        if self.parameters.use_bias_instead_of_theta:
                            self.weights[0][o] += self.parameters.learning_rate * cost  # bias
                        for i in range(len(self.input_neurons)):
                            # print(f'cost: {cost}')
                            delta = cost * vector.x[i]
                            self.weights[i + 1][o] += self.parameters.learning_rate * delta  # i + 1, because weights[0] are biases
            epoch += 1
            learn = are_errors
        print(f'\nNumber of epochs of learning: {epoch}')

        return epoch

    def feedforward(self, vector):
        self.input_neurons = vector.x
        # self.clear_output_neurons()

        for o in range(len(self.output_neurons)):
            z = 0
            if self.parameters.use_bias_instead_of_theta:
                z += self.weights[0][o]  # bias
            for i in range(len(self.input_neurons)):
                # print(f'weights len: {len(self.weights)}')
                # print(f'wegihts[i] len: {len(self.weights[i])}')
                # print(f'i: {i}, o: {o}')
                z += self.input_neurons[i] * self.weights[i + 1][o]  # i + 1, because weights[0] are biases

            self.output_neurons[o] = self.activation_function(z)

        return self.output_neurons

    def test(self, testing_vectors):
        print('\n*** Testing ***\n')
        good_classifications = 0
        for testing_vector in testing_vectors:
            y = self.feedforward(testing_vector)
            print(f'arguments: {testing_vector.x}')
            print(f'class: {testing_vector.y}')
            print(f'classified to {y}')
            print(f'is classification correct: {y == testing_vector.y}')
            if y == testing_vector.y:
                good_classifications += 1

        print(f'\nAccuracy: {good_classifications / len(testing_vectors) * 100}%')

    def activation_function(self, z):
        return self.parameters.activation_function(z, self.parameters.theta_threshold)

    @staticmethod
    def unipolar_function(z, theta):
        if z > theta:
            return 1
        return 0

    @staticmethod
    def bipolar_function(z, theta):
        if z > theta:
            return 1
        return -1