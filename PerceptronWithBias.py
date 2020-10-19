from PerceptronParameters import PerceptronParameters
import random


class PerceptronWithBias:
    parameters = PerceptronParameters
    input_neurons = []
    output_neurons = []
    weights = [[]]  # [input_neuron][output_neuron]

    def __init__(self, parameters):
        print("*** Initializing ***\n")
        self.parameters = parameters
        self.parameters.thet
        for i in range(parameters.input_neurons_num):
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
        while learn and epoch < 10000:
            print(f'\nepoch: {epoch}')
            # random.shuffle(learning_vectors)
            are_errors = False
            for vector in learning_vectors:
                self.feedforward(vector)
                for o in range(len(self.output_neurons)):
                    for i in range(len(self.input_neurons)):
                        cost = vector.y[o] - self.output_neurons[o]
                        print(f'cost: {cost}')
                        if cost != 0:
                            are_errors = True
                            delta = cost * vector.x[i]
                            self.weights[i][o] += self.parameters.learning_rate * delta
            epoch += 1
            learn = are_errors
            # print(self.weights)
            # input()

    def feedforward(self, vector):
        self.input_neurons = vector.x
        # self.clear_output_neurons()

        for o in range(len(self.output_neurons)):
            z = 0
            for i in range(len(self.input_neurons)):
                # print(f'weights len: {len(self.weights)}')
                # print(f'wegihts[i] len: {len(self.weights[i])}')
                # print(f'i: {i}, o: {o}')
                z += self.input_neurons[i] * self.weights[i][o]

            self.output_neurons[o] = self.activation_function(z)

        return self.output_neurons

    def test(self, testing_vectors):
        print('\n*** Testing ***\n')
        good_classifications = 0
        for testing_vector in testing_vectors:
            y = self.feedforward(testing_vector)
            print(f'y: {y}')
            print(f'testing_vector.y: {testing_vector.y}')
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