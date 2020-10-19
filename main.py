import statistics

from Adaline import Adaline
from AdalineParameters import AdalineParameters
from Perceptron import Perceptron
from PerceptronParameters import PerceptronParameters
from Vector import Vector

learning_vectors_unipolar = [
    Vector([0, 0], [0]),
    Vector([0, 1], [0]),
    Vector([0.001, 1.001], [0]),
    Vector([0.001, 0.001], [0]),
    Vector([-0.001, 1.001], [0]),
    Vector([-0.001, 0.001], [0]),
    Vector([0.001, 0.999], [0]),
    Vector([0.001, -0.001], [0]),
    Vector([1.001, 0.001], [0]),
    Vector([1.001, 1.001], [1]),
    Vector([0.999, 0.001], [0]),
    Vector([0.999, 1.001], [1]),
    Vector([1, 1.001], [1]),
    Vector([1, 1.002], [1]),
    Vector([0.999, 0.999], [1]),
    Vector([1.002, 0.999], [1]),
    Vector([1.002, 1], [1]),
    Vector([0.999, 0.998], [1]),
    Vector([0.999, 1.002], [1]),
    Vector([1, 1.002], [1]),
]

testing_vectors_unipolar = [
    Vector([1, 0], [0]),
    Vector([1, 1], [1]),
    Vector([-0.001, 0.999], [0]),
    Vector([-0.001, -0.001], [0]),
    Vector([0.999, 0], [0]),
    Vector([0.999, 1], [1]),
    Vector([1.002, 1], [1]),
    Vector([0.999, 1.001], [1]),
]

learning_vectors_bipolar = [
    Vector([0, 0], [-1]),
    Vector([0, 1], [-1]),
    Vector([0.001, 1.001], [-1]),
    Vector([0.001, 0.001], [-1]),
    Vector([-0.001, 1.001], [-1]),
    Vector([-0.001, 0.001], [-1]),
    Vector([0.001, 0.999], [-1]),
    Vector([0.001, -0.001], [-1]),
    Vector([1.001, 0.001], [-1]),
    Vector([1.001, 1.001], [1]),
    Vector([0.999, 0.001], [-1]),
    Vector([0.999, 1.001], [1]),
    Vector([1, 1.001], [1]),
    Vector([1, 1.002], [1]),
    Vector([0.999, 0.999], [1]),
    Vector([1.002, 0.999], [1]),
    Vector([1.002, 1], [1]),
    Vector([0.999, 0.998], [1]),
    Vector([0.999, 1.002], [1]),
    Vector([1, 1.002], [1]),
]

testing_vectors_bipolar = [
    Vector([1, 0], [-1]),
    Vector([1, 1], [1]),
    Vector([-0.001, 0.999], [-1]),
    Vector([-0.001, -0.001], [-1]),
    Vector([0.999, 0], [-1]),
    Vector([0.999, 1], [1]),
    Vector([1.002, 1], [1]),
    Vector([0.999, 1.001], [1]),
]


def test_unipolar_perceptron(parameters):
    parameters.activation_function = Perceptron.unipolar_function
    check_perceptron_parameters(parameters)
    perceptron = Perceptron(parameters)
    epochs = perceptron.learn(learning_vectors_unipolar)
    perceptron.test(testing_vectors_unipolar)

    return epochs


def test_bipolar_perceptron(parameters):
    parameters.activation_function = Perceptron.bipolar_function
    check_perceptron_parameters(parameters)
    perceptron = Perceptron(parameters)
    epochs = perceptron.learn(learning_vectors_bipolar)
    perceptron.test(testing_vectors_bipolar)

    return epochs


def test_bipolar_adaline(parameters):
    parameters.activation_function = Adaline.bipolar_function
    adaline = Adaline(parameters)
    epochs = adaline.learn(learning_vectors_bipolar)
    adaline.test(testing_vectors_bipolar)

    return epochs


def test_multiple_times(test_function, times_num):
    print('*** Testing multiple times has been started ***')
    epochs_list = []
    for t in range(times_num):
        print(f'\n*** Test no. {t + 1} ***')
        epochs_list.append(test_function())
    print(f'\nAfter {times_num} tests, epochs num average: {statistics.mean(epochs_list)}')
    print('*** End of multiple times testing ***')


def check_perceptron_parameters(parameters):
    if parameters.use_bias_instead_of_theta:
        parameters.theta_threshold = 0

    return parameters


if __name__ == '__main__':
    perceptron_parameters = PerceptronParameters(
        Perceptron.unipolar_function,
        2,
        1,
        -0.25,
        0.25,
        0.1,
        1,
        True
    )

    adaline_parameters = AdalineParameters(
        Adaline.bipolar_function,
        2,
        1,
        -0.25,
        0.25,
        0.1,
        10000
    )

    test_multiple_times(lambda: test_bipolar_perceptron(perceptron_parameters), 10)
    # test_multiple_times(lambda: test_bipolar_adaline(adaline_parameters), 10)
