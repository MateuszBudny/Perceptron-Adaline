class AdalineParameters:
    activation_function = lambda: ()
    input_neurons_num = 1
    output_neurons_num = 1
    init_weights_min = -1.0
    init_weights_max = 1.0
    learning_rate = 1.0
    permissible_error_value = 1.0

    def __init__(self, activation_function, input_neurons_num, output_neurons_num, init_weights_min, init_weights_max,
                 learn_rate, permissible_error_value):
        self.activation_function = activation_function
        self.input_neurons_num = input_neurons_num
        self.output_neurons_num = output_neurons_num
        self.init_weights_min = init_weights_min
        self.init_weights_max = init_weights_max
        self.learning_rate = learn_rate
        self.permissible_error_value = permissible_error_value
