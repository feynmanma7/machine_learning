"""
Node
Edge
Graph
"""


class Node(object):

    inputs = None
    outputs = None

    def __init__(self, name=None):
        self.name = name

    def __call__(self, *args, **kwargs):
        self.inputs = args
        return self


class Edge(object):

    data = None

    def __init__(self, name=None, data=None):
        self.name = name
        self.data = data

    def forward(self, inputs):
        self.saved_tensors = inputs
        outputs = None
        return outputs

    def backward(self, inputs, grad_out):
        middle_tensors = self.saved_tensors
        grad_in = None
        return grad_in


class Graph(Node):

    def __init__(self):
        pass

if __name__ == '__main__':

    edge_a = Edge(name='edge_a', data=1)
    edge_b = Edge(name='edge_b', data=2)

    node_a = Node(name='node_a')(edge_a, edge_b)
    print(node_a.inputs)
