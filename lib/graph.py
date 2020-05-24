"""
@author bri25yu
"""

class Vertex:

    def __init__(self, name : int):
        self._name = name

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return 'Vertex({})'.format(self._name)

    def __eq__(self, other):
        """
        Content equivalence between self and other.

        Parameter
        ---------
        other : Object
            Object to check equality with.

        Returns
        -------
        equals : bool
            True if other is a Vertex and has the same name as self, False otherwise.

        >>> v1, v2 = Vertex(1), Vertex(1)
        >>> v1 == v2
        True
        >>> v2 = v1
        >>> v1 == v2
        True
        >>> v2 == v1
        True
        >>> v2 = Vertex(2)
        >>> v1 == v2
        False
        >>> v2 == v1
        False

        """
        if isinstance(other, Vertex):
            return other._name == self._name
        return False

    def __hash__(self):
        return hash(self._name)

class Edge:
    """
    This is an edge from U to V.
    """

    def __init__(self, u : Vertex, v : Vertex, weight : float = 1):
        self._u = u
        self._v = v
        self._weight = weight

    def __repr__(self):
        return 'Edge({0}, {1}, {2})'.format(str(self._u), str(self._v), self._weight)

    def __str__(self):
        return repr(self)

    def __eq__(self, other):
        if isinstance(other, Edge):
            return self._u == other._u and self._v == other._v and self._weight == other._weight
        return False

    def __hash__(self):
        return hash(self._u) + hash(self._v) + hash(self._weight)

class Graph:
    """
    This is an abstract base class for different representations of graphs e.g. adjacency matrix and adjacency list implementations.
    """

    def __init__(self, numVertices : int, edges : list = []):
        self._numVertices = numVertices
        self._vertices = [Vertex(i) for i in range(self._numVertices)]

    def getVertices(self):
        return self._vertices.copy()

    def validVertex(self, v : Vertex):
        """
        Checks if v is a valid vertex.

        Parameters
        ----------
        v : Vertex
            Vertex to check whether or not is a valid vertex or not.

        Returns
        -------
         : boolean
            Whether or not v is a valid vertex or not.

        >>> g = Graph(4)
        >>> g.validVertex(Vertex(3))
        True
        >>> g.validVertex(Vertex(4))
        False

        """
        return v._name >= 0 and v._name < self._numVertices

    def validateVertex(self, v : Vertex):
        """
        Validates the input vertex v.

        Parameters
        ----------
        v : Vertex
            Vertex to validate.

        Raises
        ------
        AssertionError
            If v is not a valid vertex.

        >>> g = Graph(4)
        >>> g.validateVertex(Vertex(3))
        >>> g.validateVertex(Vertex(4))
        Traceback (most recent call last):
        ...
        AssertionError: Vertex input must be of type Vertex with name in range[0, 3].

        """
        assert self.validVertex(v), \
            'Vertex input must be of type Vertex with name in range[0, {0}].'.format(self._numVertices - 1)

    def validEdge(self, edge : Edge):
        """
        Checks whether the input edge is valid or not.

        Parameters
        ----------
        edge : Edge
            Edge to check validity of.

        Returns
        -------
         : bool
            Whether or not edge is valid or not.

        >>> g = Graph(4)
        >>> g.validEdge(Edge(Vertex(2), Vertex(3)))
        True
        >>> g.validEdge(Edge(Vertex(5), Vertex(3)))
        False

        """
        return self.validVertex(edge._u) and self.validVertex(edge._v)

    def validateEdge(self, edge : Edge):
        """
        Validates the input edge.

        Parameters
        ----------
        edge : Edge
            Edge to validate.

        Raises
        -------
        AssertionError
            If edge is not a valid Edge.

        >>> g = Graph(4)
        >>> g.validateEdge(Edge(Vertex(2), Vertex(3)))
        >>> g.validateEdge(Edge(Vertex(5), Vertex(3)))
        Traceback (most recent call last):
        ...
        AssertionError: Edge must connect two valid vertices.

        """
        assert self.validEdge(edge), 'Edge must connect two valid vertices.'

    def addEdge(self, edge):
        """
        Adds the input edge to the graph.

        Parameters
        ----------
        edge : Edge
            Edge to add.

        """
        pass

    def addEdges(self, edges : list):
        """
        Add all edges to the graph.

        Parameters
        ----------
        edges : list
            List of edges to add to the graph.

        """
        pass

    def getEdge(self, edge : Edge):
        """
        Gets the edge that connects the two vertices specified in the input edge.

        Parameters
        ----------
        edge : Edge
            Data encapsulation of two vertices to determine which edge to return.

        Returns
        -------
        edge : Edge
            Edge between the two vertices specified in the input edge.

        """
        pass

    def getEdges(self, vertex : Vertex):
        """
        Gets all outgoing edges that connect to the input vertex.

        Parameters
        ----------
        vertex : Vertex
            The vertex to return all outgoing edges to.

        Returns
        -------
        edges : list
            All outgoing edges from the input vertex.
        """
        pass

    def updateEdge(self, edge : Edge):
        """
        Updates the edge connecting the two input vertices with the input edge weight.

        Parameters
        ----------
        edge : Edge
            Data encapsulation of the two input vertices and their new weight.

        """
        pass

    def updateEdges(self, edges : list):
        """
        Updates all edges in the list.

        Parameters
        ----------
        edges : list
            List of edges to update.

        """
        pass

    def removeEdge(self, edge : Edge):
        """
        Removes the edge connecting the two input vertices.

        Parameters
        ----------
        edge : Edge
            Data encapsulation of the edge between two vertices to remove.

        """
        pass

    def removeEdges(self, edges : list):
        """
        Removes all edges in the input list.

        Parameters
        ----------
        edges : list
            List of edges to remove.

        """
        pass

class AdjMatrixGraph(Graph):
    
    def __init__(self, numVertices : int, edges : list = []):
        super(AdjMatrixGraph, self).__init__(numVertices, edges)

        self._adjMatrix = [[Edge(self._vertices[i], self._vertices[j], 0) for j in range(self._numVertices)] for i in range(self._numVertices)]
        self.addEdges(edges)

    def addEdge(self, edge : Edge):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'None'
        >>> g.addEdge(Edge(Vertex(1), Vertex(2), 4))
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 4)'
        >>> str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        'Edge(Vertex(2), Vertex(0), -5)'
        """
        self.updateEdge(edge)

    def addEdges(self, edges : list):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(0), Vertex(1), 3)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('None', 'None')
        >>> g.addEdges([Edge(Vertex(1), Vertex(2), 33), Edge(Vertex(2), Vertex(0), -2)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 33)', 'Edge(Vertex(2), Vertex(0), -2)')
        >>> str(g.getEdge(Edge(Vertex(0), Vertex(1))))
        'Edge(Vertex(0), Vertex(1), 3)'
        """
        self.updateEdges(edges)

    def getEdge(self, edge : Edge):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2), 7)))
        'Edge(Vertex(1), Vertex(2), 0.5)'
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(1), 7)))
        'None'
        >>> str(g.getEdge(Edge(Vertex(3), Vertex(0))))
        Traceback (most recent call last):
        ...
        AssertionError: Edge must connect two valid vertices.
        
        """
        self.validateEdge(edge)
        
        contender = self._adjMatrix[edge._u._name][edge._v._name]
        if contender._weight:
            return contender
        return None

    def getEdges(self, vertex : Vertex):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(1), Vertex(0), 2), Edge(Vertex(2), Vertex(0), -5)])
        >>> [str(edge) for edge in g.getEdges(Vertex(1))]
        ['Edge(Vertex(1), Vertex(0), 2)', 'Edge(Vertex(1), Vertex(2), 0.5)']

        """
        self.validateVertex(vertex)

        edges = []
        for edge in self._adjMatrix[vertex._name]:
            if edge._weight:
                edges.append(edge)
        return edges

    def updateEdge(self, edge : Edge):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 0.5)'
        >>> g.updateEdge(Edge(Vertex(1), Vertex(2), 4))
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 4)'
        >>> str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        'Edge(Vertex(2), Vertex(0), -5)'
        """
        self.validateEdge(edge)

        self._adjMatrix[edge._u._name][edge._v._name]._weight = edge._weight

    def updateEdges(self, edges : list):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5), Edge(Vertex(0), Vertex(1), 3)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 0.5)', 'Edge(Vertex(2), Vertex(0), -5)')
        >>> g.updateEdges([Edge(Vertex(1), Vertex(2), 33), Edge(Vertex(2), Vertex(0), -2)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 33)', 'Edge(Vertex(2), Vertex(0), -2)')
        >>> str(g.getEdge(Edge(Vertex(0), Vertex(1))))
        'Edge(Vertex(0), Vertex(1), 3)'
        """
        for edge in edges:
            self.updateEdge(edge)

    def removeEdge(self, edge : Edge):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 0.5)'
        >>> g.removeEdge(Edge(Vertex(1), Vertex(2)))
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'None'
        >>> str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        'Edge(Vertex(2), Vertex(0), -5)'
        """
        self.validateEdge(edge)

        self._adjMatrix[edge._u._name][edge._v._name]._weight = 0

    def removeEdges(self, edges : list):
        """
        >>> g = AdjMatrixGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5), Edge(Vertex(0), Vertex(1), 3)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 0.5)', 'Edge(Vertex(2), Vertex(0), -5)')
        >>> g.removeEdges([Edge(Vertex(1), Vertex(2)), Edge(Vertex(2), Vertex(0))])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('None', 'None')
        >>> str(g.getEdge(Edge(Vertex(0), Vertex(1))))
        'Edge(Vertex(0), Vertex(1), 3)'
        """
        for edge in edges:
            self.removeEdge(edge)

class AdjListGraph(Graph):
        
    def __init__(self, numVertices : int, edges : list = []):
        super(AdjListGraph, self).__init__(numVertices, edges)

        self._adjList = dict()
        for i in range(self._numVertices):
            self._adjList[i] = []

        self.addEdges(edges)

    def addEdge(self, edge : Edge):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'None'
        >>> g.addEdge(Edge(Vertex(1), Vertex(2), 4))
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 4)'
        >>> str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        'Edge(Vertex(2), Vertex(0), -5)'
        """
        self.validateEdge(edge)
        
        contender = self.getEdge(edge)
        if contender:
            contender._weight = edge._weight
        else:
            self._adjList[edge._u._name].append(edge)

    def addEdges(self, edges : list):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1), 3)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('None', 'None')
        >>> g.addEdges([Edge(Vertex(1), Vertex(2), 33), Edge(Vertex(2), Vertex(0), -2)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 33)', 'Edge(Vertex(2), Vertex(0), -2)')
        >>> str(g.getEdge(Edge(Vertex(0), Vertex(1))))
        'Edge(Vertex(0), Vertex(1), 3)'
        """
        for edge in edges:
            self.addEdge(edge)

    def getEdge(self, edge : Edge):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2), 7)))
        'Edge(Vertex(1), Vertex(2), 0.5)'
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(1), 7)))
        'None'
        >>> str(g.getEdge(Edge(Vertex(3), Vertex(0))))
        Traceback (most recent call last):
        ...
        AssertionError: Edge must connect two valid vertices.
        
        """
        self.validateEdge(edge)
        
        for possibleEdge in self._adjList[edge._u._name]:
            if edge._v._name == possibleEdge._v._name:
                return possibleEdge

        return None

    def getEdges(self, vertex : Vertex):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(1), Vertex(0), 2), Edge(Vertex(2), Vertex(0), -5)])
        >>> gotEdges = g.getEdges(Vertex(1))
        >>> gotEdges.sort(key = lambda e: e._weight)
        >>> [str(edge) for edge in gotEdges]
        ['Edge(Vertex(1), Vertex(2), 0.5)', 'Edge(Vertex(1), Vertex(0), 2)']

        """
        self.validateVertex(vertex)

        edges = []
        for edge in self._adjList[vertex._name]:
            if edge._weight:
                edges.append(edge)
        return edges

    def updateEdge(self, edge : Edge):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 0.5)'
        >>> g.updateEdge(Edge(Vertex(1), Vertex(2), 4))
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 4)'
        >>> str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        'Edge(Vertex(2), Vertex(0), -5)'
        """
        self.addEdge(edge)

    def updateEdges(self, edges : list):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5), Edge(Vertex(0), Vertex(1), 3)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 0.5)', 'Edge(Vertex(2), Vertex(0), -5)')
        >>> g.updateEdges([Edge(Vertex(1), Vertex(2), 33), Edge(Vertex(2), Vertex(0), -2)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 33)', 'Edge(Vertex(2), Vertex(0), -2)')
        >>> str(g.getEdge(Edge(Vertex(0), Vertex(1))))
        'Edge(Vertex(0), Vertex(1), 3)'
        """
        self.addEdges(edges)

    def removeEdge(self, edge : Edge):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'Edge(Vertex(1), Vertex(2), 0.5)'
        >>> g.removeEdge(Edge(Vertex(1), Vertex(2)))
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2))))
        'None'
        >>> str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        'Edge(Vertex(2), Vertex(0), -5)'
        """
        self.validateEdge(edge)

        contender = self.getEdge(edge)
        if contender:
            self._adjList[edge._u._name].remove(contender)

    def removeEdges(self, edges : list):
        """
        >>> g = AdjListGraph(3, [Edge(Vertex(1), Vertex(2), 0.5), Edge(Vertex(2), Vertex(0), -5), Edge(Vertex(0), Vertex(1), 3)])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('Edge(Vertex(1), Vertex(2), 0.5)', 'Edge(Vertex(2), Vertex(0), -5)')
        >>> g.removeEdges([Edge(Vertex(1), Vertex(2)), Edge(Vertex(2), Vertex(0))])
        >>> str(g.getEdge(Edge(Vertex(1), Vertex(2)))), str(g.getEdge(Edge(Vertex(2), Vertex(0))))
        ('None', 'None')
        >>> str(g.getEdge(Edge(Vertex(0), Vertex(1))))
        'Edge(Vertex(0), Vertex(1), 3)'
        """
        for edge in edges:
            self.removeEdge(edge)


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    