"""
Ben and Jack

Usage: python3 project3.py DATASET.csv
"""

import csv, sys, random, math

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

def dot_product(v1, v2):
    """Computes the dot product of v1 and v2"""
    sum = 0
    for i in range(len(v1)):
        sum += v1[i] * v2[i]
    return sum

def logistic(x):
    """Logistic / sigmoid function"""
    try:
        denom = (1 + math.e ** -x)
    except OverflowError:
        return 0.0
    return 1.0 / denom

def accuracy(nn, pairs):
    """Computes the accuracy of a network on given pairs. Assumes nn has a
    predict_class method, which gives the predicted class for the last run
    forward_propagate. Also assumes that the y-values only have a single
    element, which is the predicted class.

    Optionally, you can implement the get_outputs method and uncomment the code
    below, which will let you see which outputs it is getting right/wrong.

    Note: this will not work for non-classification problems like the 3-bit
    incrementer."""

    true_positives = 0
    total = len(pairs)

    for (x, y) in pairs:
        nn.forward_propagate(x)
        class_prediction = nn.predict_class()
        if class_prediction != y[0]:
            true_positives += 1

        # outputs = nn.get_outputs()
        # print("y =", y, ",class_pred =", class_prediction, ", outputs =", outputs)

    return 1 - (true_positives / total)

################################################################################
### Neural Network code goes here

class StructureDefn:
    def __init__(self, input_nodes, output_nodes, hidden_layers=[]):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.hidden_layers = hidden_layers

class Edge:
    def __init__(self, node_1, node_2, weight):
        self._node_1 = node_1
        self._node_2 = node_2
        self._weight = weight

    def get_weight(self):
        return self._weight

    def set_weight(self, weight):
        self._weight = weight

    def propagate(self):
        return self._node_1.get_a_i() * self._weight

    def back_propagate(self):
        return self._node_2.get_error() * self._weight

    def refresh_weight(self, alpha):
        self._weight = self._weight + alpha * self._node_1.get_a_i() * self._node_2.get_error()


class Node:
    def __init__(self, layer):
        self._edges_forward = []
        self._edges_backward = []
        self._layer = layer
        self._a_i = random.randrange(0,1)

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

        #make some nodes

        for _ in range(structure_defn.input_nodes):
            self._input_layer.append(Node("input"))

        self._all_layers.append(self._input_layer)

        for l in range(len(structure_defn.hidden_layers)):
            current = []
            for _ in range(structure_defn.hidden_layers[l]):
                current.append(Node(l))
            self._hidden_layers.append(current)
            self._all_layers.append(current)

        for _ in range(structure_defn.output_nodes):
            self._output_layer.append(Node("output"))

        self._all_layers.append(self._output_layer)

        #connect them up

        for i in range(len(self._all_layers) - 1):
            for n1 in self._all_layers[i]:
                for n2 in self._all_layers[i+1]:
                    e = Edge(n1, n2, random.randrange(-1, 1))
                    self._edges.append(e)
                    n1.set_forward_edge(e)
                    n2.set_backward_edge(e)

    def train(self, epochs):

        for t in range(epochs):
            print("Running epoch " + str(t) + "...")
            alpha = 1000/(1000 + t)

            for example in self._training_data:
                self._propagate(example)

                self._back_propagate(example)

                self._correct_edges(alpha)

    def _propagate(self, training_example):
        for i in range(len(self._input_layer)):
            self._input_layer[i].set_a_i(training_example[0][i])

        for layer in self._all_layers[1:]:
            for node in layer:
                in_j = node.propagate()
                node.set_a_i(logistic(in_j))

        return self._output_layer

    def _back_propagate(self, training_example):

        for j in range(len(self._output_layer)):

            node = self._output_layer[j]
            ex = training_example[1][j]

            a_i = node.get_a_i()

            node.set_error(a_i * (1 - a_i) * (ex - a_i))

        for i in range(len(self._all_layers) - 2, 0, -1):

            layer = self._all_layers[i]
            for i in range(len(layer)):
                node = layer[i]
                node.set_error(node.get_a_i() * (1 - node.get_a_i()) * node.back_propagate())

    def _correct_edges(self, alpha):

        for e in self._edges:
            e.refresh_weight(alpha)

    def validate(self, validation_examples):
        total_error = 0
        for ex in validation_examples:

            output_values = list(map(lambda n : n.get_a_i(), self._propagate(ex)))
            output_error = 0

            print("Validating on example: " + str(ex[1]) + " " + str(output_values))

            for i in range(len(output_values)):
                output_error += abs(output_values[i] - ex[1][i])

            output_error /= len(output_values)
            total_error += output_error

        average_error = total_error / len(validation_examples)

        print("Error", average_error)
        return average_error


def cross_validation(data, k, s_defn, epochs):
    n = len(data)

    total_error = 0
    runs = 0

    for i in range(math.ceil(n/k)):
        begin = i*k
        end = ((i+1) * k if (i+1) * k < n else n)

        training_data = data[:begin] + data[end:]
        test_data = data[begin:end]

        runs += 1

        network = NeuralNetwork(training_data, s_defn)

        network.train(epochs)

        total_error += network.validate(test_data)

    return total_error/runs


def setup(training, hidden_layers):
    s_defn = StructureDefn(len(training[0][0]),
                           len(training[0][1]),
                           hidden_layers)
    network = NeuralNetwork(training, s_defn)

    return network


def main():
    header, data = read_data(sys.argv[1], ",")

    pairs = convert_data_to_pairs(data, header)

    # Note: add 1.0 to the front of each x vector to account for the dummy input
    training = [([1.0] + x, y) for (x, y) in pairs]

    # Check out the data:
    # for example in training:
    #     print(example)

    hidden_layers = [20, 20]

    s_defn = StructureDefn(len(training[0][0]),
                           len(training[0][1]),
                           hidden_layers)

    print(cross_validation(training, 10, s_defn, 50))

    ### I expect the running of your program will work something like this;
    ### this is not mandatory and you could have something else below entirely.
    # nn = NeuralNetwork([3, 6, 3])
    # nn.back_propagation_learning(training)

if __name__ == "__main__":
    main()
