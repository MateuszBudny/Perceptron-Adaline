from AdalineParameters import AdalineParameters
import random


class Adaline:
    parameters = AdalineParameters
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
            # print(f'\nepoch: {epoch}')
            random.shuffle(learning_vectors)

            # learning Adaline
            for vector in learning_vectors:
                self.feedforward_sum(vector)
                for o in range(len(self.output_neurons)):
                    error_value = vector.y[o] - self.output_neurons[o]
                    self.weights[0][o] += self.parameters.learning_rate * error_value  # bias
                    for i in range(len(self.input_neurons)):
                        # print(f'error_value: {error_value}')
                        self.weights[i + 1][o] += self.parameters.learning_rate * error_value * vector.x[i]  # i + 1, because weights[0] are biases

            # calculating mean square error
            square_error_sum = 0
            for vector in learning_vectors:
                self.feedforward_sum(vector)
                for o in range(len(self.output_neurons)):
                    square_error_sum += (vector.y[o] - self.output_neurons[o]) ** 2
            mean_square_error_value = square_error_sum / len(learning_vectors)

            epoch += 1
            learn = mean_square_error_value > self.parameters.permissible_error_value
        print(f'\nNumber of epochs of learning: {epoch}')

        return epoch

    def feedforward(self, vector):
        self.feedforward_sum(vector)
        self.apply_activation_function_on_output_neurons()

        return self.output_neurons

    def feedforward_sum(self, vector):
        self.input_neurons = vector.x
        # self.clear_output_neurons()

        for o in range(len(self.output_neurons)):
            z = self.weights[0][o]  # bias
            for i in range(len(self.input_neurons)):
                # print(f'weights len: {len(self.weights)}')
                # print(f'wegihts[i] len: {len(self.weights[i])}')
                # print(f'i: {i}, o: {o}')
                z += self.input_neurons[i] * self.weights[i + 1][o]  # i + 1, because weights[0] are biases
            self.output_neurons[o] = z

    def apply_activation_function_on_output_neurons(self):
        for o in range(len(self.output_neurons)):
            self.output_neurons[o] = self.activation_function(self.output_neurons[o])

    def test(self, testing_vectors):
        print('\n*** Testing ***\n')
        good_classifications = 0
        for testing_vector in testing_vectors:
            y = self.feedforward(testing_vector)
            # print(f'y: {y}')
            # print(f'testing_vector.y: {testing_vector.y}')
            if y == testing_vector.y:
                good_classifications += 1

        print(f'\nAccuracy: {good_classifications / len(testing_vectors) * 100}%')

    def activation_function(self, z):
        return self.parameters.activation_function(z)

    @staticmethod
    def unipolar_function(z):
        if z > 0:
            return 1
        return 0

    @staticmethod
    def bipolar_function(z):
        if z > 0:
            return 1
        return -1