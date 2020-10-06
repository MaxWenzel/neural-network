import logging
import numpy
import scipy.special
import matplotlib.pyplot as plt

class NeuralNetwork:
    logger = logging.getLogger(__name__)

    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        logging.debug("Init NN")
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        self.lr = learning_rate

        self.activation_function = lambda x: scipy.special.expit(x)

        pass

    def train(self, inputs_list, targets_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)

        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))

        pass

    def test(self, test_data):
        all_values = test_data[0].split(',')
        logging.info("Label %s", all_values[0])

        image_array = numpy.asfarray(all_values[1:]).reshape((28, 28))
        plt.imshow(image_array, cmap='Greys', interpolation='None')
        plt.show()
        return self.query((numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01)

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def pre_train(self, training_data):
        for record in training_data:
            all_values = record.split(',')
            inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

            targets = numpy.zeros(self.onodes) + 0.01
            targets[int(all_values[0])] = 0.99
            self.train(inputs, targets)
            pass

    def hello_world(self):
        logging.info("Hello World")