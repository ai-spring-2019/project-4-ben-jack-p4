"""
Ben and Jack

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math, time, statistics, ast, argparse

def read_data(filename, delimiter=",", has_header=True):
    """Reads datafile using given delimiter. Returns a header and a list of
    the rows of data."""
    data = []
    header = []
    with open(filename) as f:
        reader = csv.reader(f, delimiter=delimiter)
        if has_header:
            header = next(reader, None)
        for line in reader:
            example = [float(x) for x in line]
            data.append(example)

        return header, data

def convert_data_to_pairs(data, header):
    """Turns a data list of lists into a list of (attribute, target) pairs."""
    pairs = []
    for example in data:
        x = []
        y = []
        for i, element in enumerate(example):
            if header[i].startswith("target"):
                y.append(element)
            else:
                x.append(element)
        pair = (x, y)
        pairs.append(pair)
    return pairs

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom


################################################################################
### Neural Network code goes here

class StructureDefn:
    ''' This class is used to remove some of the structuring uglyness from
        our actual NeuralNetwork class '''
    def __init__(self, input_nodes, output_nodes, hidden_layers=[]):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

class Edge:
    ''' The class used for edges in the network - holds a weight, and the nodes
        after and before the edge '''

    def __init__(self, node_1, node_2, weight):
        self._node_1 = node_1
        self._node_2 = node_2
        self._weight = weight

    def get_weight(self):
        ''' Gives the weight of the edge to be used in calculation '''
        return self._weight

    def set_weight(self, weight):
        ''' Sets a new weight for the edge '''
        self._weight = weight

    def propagate(self):
        ''' Propagates the value of the previous node through the egde'''
        return self._node_1.get_a_i() * self._weight

    def back_propagate(self):
        ''' Sends the error back through the network along the edge '''
        return self._node_2.get_error() * self._weight

    def refresh_weight(self, alpha):
        ''' Resets the weights of the edges using the error of the
            following node and the a_i of the previous '''
        self._weight = self._weight + alpha * self._node_1.get_a_i() * self._node_2.get_error()


class Node:
    def __init__(self, layer):
        self._edges_forward = []
        self._edges_backward = []
        self._layer = layer
        self._a_i = 1

    def set_forward_edge(self, edge):
        self._edges_forward.append(edge)

    def set_backward_edge(self, edge):
        self._edges_backward.append(edge)

    def propagate(self):
        sum = 0
        for e in self._edges_backward:
            sum += e.propagate()
        return sum

    def back_propagate(self):
        sum = 0
        for e in self._edges_forward:
            sum += e.back_propagate()
        return sum

    def set_a_i(self, a_i):
        self._a_i = a_i

    def get_a_i(self):
        return self._a_i

    def set_error(self, error):
        self._error = error

    def get_error(self):
        return self._error

    def __str__(self):
        return "Node: value: " + str(self._value) + " a_i: " + str(self._a_i) + " Layer: " + str(self._layer)

class NeuralNetwork:
    def __init__(self, training_data, structure_defn):
        self._training_data = training_data

        self._input_layer = []
        self._output_layer = []
        self._hidden_layers = []

        self._all_layers = []

        self._edges = []

        # Making all nodes using the structure definition

        # Begin by making nodes for the input layers
        for _ in range(structure_defn.input_nodes):
            self._input_layer.append(Node("input"))

        self._all_layers.append(self._input_layer)

        # Make nodes for all hidden layers
        for l in range(len(structure_defn.hidden_layers)):
            current = []
            for _ in range(structure_defn.hidden_layers[l]):
                current.append(Node(l))
            self._hidden_layers.append(current)
            self._all_layers.append(current)

        # Add the nodes for the output layer
        for _ in range(structure_defn.output_nodes):
            self._output_layer.append(Node("output"))

        self._all_layers.append(self._output_layer)

        # Connects the nodes in each layer to those in the next/previous
        for i in range(len(self._all_layers) - 1):
            for n1 in self._all_layers[i]:
                for n2 in self._all_layers[i+1]:
                    e = Edge(n1, n2, random.random())
                    self._edges.append(e)
                    n1.set_forward_edge(e)
                    n2.set_backward_edge(e)

    def train(self, epochs):
        ''' Trains the network using propagation, then back propagation, the
            correcting the error weights '''
        for t in range(epochs):

            # Decaying rate of learning
            alpha = 100/(100 + t)

            for example in self._training_data:
                self._propagate(example)

                self._back_propagate(example)

                self._correct_edges(alpha)

    def _propagate(self, training_example):
        ''' Propagates the data through the network '''

        # Sets the values in the input layer
        for i in range(len(self._input_layer)):
            self._input_layer[i].set_a_i(training_example[0][i])

        # Pushes the data forward through the netwrok
        for layer in self._all_layers[1:]:
            for node in layer:
                in_j = node.propagate()
                node.set_a_i(logistic(in_j))

        # Returns the output layer so that we can get the results
        return self._output_layer

    def _back_propagate(self, training_example):
        ''' Computes the errors of all nodes in the network'''

        for j in range(len(self._output_layer)):

            # Looks at all nodes in the output layer
            node = self._output_layer[j]
            ex = training_example[1][j]

            # Gets the current a_i from the node
            a_i = node.get_a_i()

            # Computes and sets the error value at that node
            node.set_error(a_i * (1 - a_i) * (ex - a_i))

        # We then do the same for all hidden layers
        for i in range(len(self._all_layers) - 2, 0, -1):

            layer = self._all_layers[i]
            for i in range(len(layer)):
                node = layer[i]
                node.set_error(node.get_a_i() * (1 - node.get_a_i()) * node.back_propagate())

    def _correct_edges(self, alpha):
        ''' Recomputes the weights of the edges '''
        for e in self._edges:
            e.refresh_weight(alpha)

    def validate(self, validation_examples):
        ''' The validation method for a multi Neural Network
            classification problem '''

        total_correct = 0

        # Checks the result from running each example in the test data
        for ex in validation_examples:

            # Propagates the data through the network, and finds the output
            output_values = list(map(lambda n : n.get_a_i(), self._propagate(ex)))

            print("Target: " + str(ex[1]) + " Result: " + str(output_values))

            # Finds the highest of all output values
            greatest = 0
            for i in range(len(output_values)):
                if output_values[greatest] < output_values[i]:
                    greatest = i

            # Checks if we are correct in our classification
            if ex[1][greatest] == 1:
                total_correct += 1

        # Calculates what percent are correct
        percent_correct = total_correct/len(validation_examples)
        print("Accuracy:", percent_correct)
        return percent_correct

    def validate_binary(self, validation_examples):
        ''' The validation method for a binary Neural Network
            classification problem '''
        total_correct = 0

        # Checks the result from running each example in the test data
        for ex in validation_examples:
            # Propagates the data through the network, and finds the output

            output_values = list(map(lambda n : n.get_a_i(), self._propagate(ex)))

            print("Target: " + str(ex[1]) + " Result: " + str(output_values))

            # Finds the class in our binary classifier
            output = (0 if output_values[0] < 0.5 else 1)

            # Checks if we are correct in our classification
            if output == ex[1][0]:
                total_correct += 1

        # Calculates what percent are correct
        percent_correct = total_correct / len(validation_examples)
        print("Fit Percentage:", percent_correct)
        return percent_correct


def cross_validation(data, k, s_defn, epochs):
    ''' Performs cross validation on the dataset data, with k subsets of the
        data, using the structure given by s_defn, and for 'epochs' epochs on
        a multiclass problem '''

    n = len(data)

    total_accuracy = 0
    runs = 0
    # Finds the size of a subset
    subset_size = math.ceil(n/k)

    # Looks at all k subsets
    for i in range(k):

        # Get the index of the beginning and end of the subset to look at
        begin = i*subset_size
        end = ((i+1) * subset_size if (i+1) * subset_size < n else n)

        # In the final iteration the beginning might come after the end position
        if begin < end:
            # Separates the training and test data
            training_data = data[:begin] + data[end:]
            test_data = data[begin:end]

            runs += 1

            # Creates the network
            network = NeuralNetwork(training_data, s_defn)

            # Trains the network
            network.train(epochs)

            # Finds the accuracy of the trained network
            total_accuracy += network.validate(test_data)

    return total_accuracy/runs

def cross_validation_binary(data, k, s_defn, epochs):
    ''' Performs cross validation on the dataset data, with k subsets of the
        data, using the structure given by s_defn, and for 'epochs' epochs on
        a binary problem '''

    n = len(data)

    total_correct = 0
    runs = 0
    # Finds the size of a subset
    subset_size = math.ceil(n/k)

    # Looks at all k subsets
    for i in range(k):

        # Get the index of the beginning and end of the subset to look at
        begin = i*subset_size
        end = ((i+1) * subset_size if (i+1) * subset_size < n else n)

        # In the final iteration the beginning might come after the end position
        if begin < end:
            # Separates the training and test data
            training_data = data[:begin] + data[end:]
            test_data = data[begin:end]

            runs += 1

            # Creates the network
            network = NeuralNetwork(training_data, s_defn)

            # Trains the network
            network.train(epochs)

            # Finds the accuracy of the trained network
            total_correct += network.validate_binary(test_data)

    return total_correct/runs

def setup_train_validate(training, hidden_layers, epochs):
    s_defn = StructureDefn(len(training[0][0]),
                           len(training[0][1]),
                           hidden_layers)
    network = NeuralNetwork(training, s_defn)

    network.train(epochs)

    print(network.validate(training))

def normalize(data):
    ''' This method is used to normalize our dataset in order to prevent values
        that are all larger from having a larger effect than they should '''
    for entry in range(len(data[0][0])):
        all_data = []
        for example in range(len(data)):
            all_data.append(data[example][0][entry])

        ''' Get the mean and std dev to use in narmalizing data '''
        mean = sum(all_data)/len(all_data)
        std_dev = statistics.stdev(all_data)

        ''' If the std dev is 0, then all values are the same,
            so no need to normalized'''
        if std_dev != 0:
            for example in range(len(data)):
                data[example][0][entry] = (data[example][0][entry] - mean)/std_dev


VALIDATION_OPTS = {'multi' : cross_validation,
                   'binary' : cross_validation_binary}

parser = argparse.ArgumentParser(description='AI Project Four -- Ben/Jack')

parser.add_argument('--validation', '-v',
                    dest='validation',
                    choices=VALIDATION_OPTS.keys(),
                    help='Cross Validation: multi binary',
                    default="multi",
                    required=False)

parser.add_argument('--data', '-d',
                    dest='data_path',
                    required=True,
                    help='Data file')

parser.add_argument('--output', '-o',
                    dest='output_file',
                    required=True,
                    help='File to output run data to')

parser.add_argument('--k-val', '-k',
                    dest='k_value',
                    default=5, type=int,
                    required=False,
                    help='K value for cross validation')

parser.add_argument('--layers', '-l',
                    dest='layer_structure',
                    required=False,
                    default="[5,5]",
                    help='Structure of hidden layers. Ex: [10,10,5] (Omit spaces)')

parser.add_argument('--epochs', '-e',
                    dest='epochs',
                    default=500, type=int,
                    required=False,
                    help='Number of epochs')

parser.add_argument('--noheader', '-n',
                    dest='noheader',
                    default=False,
                    action='store_true',
                    required=False,
                    help='Omit header row from csv')


def main():

    args = parser.parse_args()

    header, data = read_data(args.data_path, ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]
    random.shuffle(training)

    normalize(training)

    #WARN: potentially dangerous to evaluate code in this way
    hidden_layers = ast.literal_eval(args.layer_structure)

    if not args.noheader:
        print("Epochs,hidden_layers,k,Accuracy,time", file=open(args.output_file, "a"))

    start = time.time()
    s_defn = StructureDefn(len(training[0][0]),
                           len(training[0][1]),
                           hidden_layers)

    h_l_s = ""
    for layer in hidden_layers:
        h_l_s += str(layer) + "_"

    print(str(args.epochs) + "," + str(h_l_s) + "," + str(args.k_value) + ",", file=open(args.output_file, "a"), end="")
    print(VALIDATION_OPTS[args.validation](training, args.k_value, s_defn, args.epochs),
          file=open(args.output_file, "a"), end="")
    end = time.time() - start
    print("," + str(end), file=open(args.output_file, "a"))


if __name__ == "__main__":
    main()
