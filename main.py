import logging
import sys
from neural.network.NeuralNetwork import NeuralNetwork

def main():
    file_handler = logging.FileHandler(filename='neural.log')
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [file_handler, stdout_handler]

    logging.basicConfig(
        level=logging.DEBUG,
        datefmt="%d.%m.%Y %H:%M:%S",
        format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
        handlers=handlers
    )

    logger = logging.getLogger('LOGGER_NAME')

    logger.info("python main function")
    execute()

def execute():
    logging.info("Start processing")
    training_data = read_data('resources/mnist_train_100.csv')

    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10

    learning_rate = 0.3

    nn = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    nn.hello_world()
    nn.pre_train(training_data)
    # query_result = nn.query([1.0, 0.5, -1.5])
    # logging.info('Query-Result [%s]' % ', '.join(map(str, query_result)))

    test_data = read_data('resources/mnist_test_10.csv')
    test_result = nn.test(test_data)
    logging.info('Test-Result [%s]' % ', '.join(map(str, test_result)))


def read_data(file_name):
    logging.debug("Try to read file %s", file_name)
    data_file = open(file_name, 'r')
    data_list = data_file.readlines()
    data_file.close()
    return data_list

if __name__ == '__main__':
    main()