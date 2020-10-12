from Perceptron import Perceptron
from PerceptronParameters import PerceptronParameters
from Vector import Vector


if __name__ == '__main__':
    learning_vectors = [
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

    testing_vectors = [
        Vector([1, 0], [0]),
        Vector([1, 1], [1]),
        Vector([-0.001, 0.999], [0]),
        Vector([-0.001, -0.001], [0]),
        Vector([0.999, 0], [0]),
        Vector([0.999, 1], [1]),
        Vector([1.002, 1], [1]),
        Vector([0.999, 1.001], [1]),
    ]

    parameters = PerceptronParameters(
        Perceptron.unipolar_function,
        2,
        1,
        -0.5,
        0.5,
        0.1,
        1
    )

    perceptron = Perceptron(parameters)
    perceptron.learn(learning_vectors)
    perceptron.test(testing_vectors)
