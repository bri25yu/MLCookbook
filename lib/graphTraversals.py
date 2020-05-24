"""
@author bri25yu
"""

from lib.graph import Graph, Vertex, AdjListGraph, Edge
from lib.tools import Queue, Heap, TreeNode, Trees

class BFS:

    def __init__(self):
        pass

    def shortestPaths(
        self,
        graph : Graph,
        source : Vertex,
        tieBreakFn = lambda listOfVertices: listOfVertices.sort(key = lambda v: v._name),
        preApplyFn = lambda v: v,
        postApplyFn = lambda v: v):
        """
        Runs the BFS shortest paths algorithm where each edge is of uniform cost (typically represented as cost of 1).

        Parameters
        ----------
        graph : Graph
            Graph to run BFS on.
        source : Vertex
            Vertex to start running BFS from i.e. this function will run shortest paths from this vertex.
        tieBreakFn : Function
            Break ties between vertices to consider. Default is breaking ties alphabetically.
        preApplyFn : Function
            Extrinsic function that is applied the first time a vertex is added to the Queue.
        postApplyFn : Function
            Extrinsic function that is applied as soon as a vertex is popped off the Queue.

        Returns
        -------
        paths : Dict
            Shortest paths in the form of a list of vertices between the source and every other reachable vertex from source.

        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2))])
        >>> paths = BFS().shortestPaths(g, Vertex(0))
        >>> paths[Vertex(0)]
        [Vertex(0)]
        >>> paths[Vertex(1)]
        [Vertex(0), Vertex(1)]
        >>> paths[Vertex(2)]
        [Vertex(0), Vertex(2)]

        >>> g = AdjListGraph(5, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2)), Edge(Vertex(2), Vertex(3)), Edge(Vertex(1), Vertex(4)), Edge(Vertex(3), Vertex(4)), Edge(Vertex(4), Vertex(1))])
        >>> paths = BFS().shortestPaths(g, Vertex(0))
        >>> paths[Vertex(0)]
        [Vertex(0)]
        >>> paths[Vertex(1)]
        [Vertex(0), Vertex(1)]
        >>> paths[Vertex(2)]
        [Vertex(0), Vertex(2)]
        >>> paths[Vertex(3)]
        [Vertex(0), Vertex(2), Vertex(3)]
        >>> paths[Vertex(4)]
        [Vertex(0), Vertex(1), Vertex(4)]
        """
        paths = {source : []}
        current = source

        def preApplyFnDist(v):
            nonlocal paths
            paths[v] = paths.get(current, []) + [v]
            preApplyFn(v)

        def postApplyFnDist(v):
            nonlocal current
            current = v
            postApplyFn(v)

        self._genericBFS(graph, source, None, tieBreakFn, preApplyFnDist, postApplyFnDist)

        return paths

    def shortestDists(
        self,
        graph : Graph,
        source : Vertex,
        tieBreakFn = lambda listOfVertices: listOfVertices.sort(key = lambda v: v._name),
        preApplyFn = lambda v: v,
        postApplyFn = lambda v: v):
        """
        Runs the BFS shortest paths algorithm where each edge is of uniform cost (typically represented as cost of 1).

        Parameters
        ----------
        graph : Graph
            Graph to run BFS on.
        source : Vertex
            Vertex to start running BFS from i.e. this function will run shortest paths from this vertex.
        tieBreakFn : Function
            Break ties between vertices to consider. Default is breaking ties alphabetically.
        preApplyFn : Function
            Extrinsic function that is applied the first time a vertex is added to the Queue.
        postApplyFn : Function
            Extrinsic function that is applied as soon as a vertex is popped off the Queue.

        Returns
        -------
        dists : Dict
            Shortest dists from the source to every other reachable vertex from source.

        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2))])
        >>> dists = BFS().shortestDists(g, Vertex(0))
        >>> dists[Vertex(0)]
        0
        >>> dists[Vertex(1)]
        1
        >>> dists[Vertex(2)]
        1

        >>> g = AdjListGraph(5, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2)), Edge(Vertex(2), Vertex(3)), Edge(Vertex(1), Vertex(4)), Edge(Vertex(3), Vertex(4)), Edge(Vertex(4), Vertex(1))])
        >>> dists = BFS().shortestDists(g, Vertex(0))
        >>> dists[Vertex(0)]
        0
        >>> dists[Vertex(1)]
        1
        >>> dists[Vertex(2)]
        1
        >>> dists[Vertex(3)]
        2
        >>> dists[Vertex(4)]
        2
        """
        dists = {source : -1}
        current = source

        def preApplyFnDist(v):
            dists[v] = dists[current] + 1
            preApplyFn(v)

        def postApplyFnDist(v):
            nonlocal current
            current = v
            postApplyFn(v)

        self._genericBFS(graph, source, None, tieBreakFn, preApplyFnDist, postApplyFnDist)

        return dists

    def shortestPathToGoal(
        self,
        graph : Graph,
        source : Vertex,
        goal : Vertex,
        tieBreakFn = lambda listOfVertices: listOfVertices.sort(key = lambda v: v._name),
        preApplyFn = lambda v: v,
        postApplyFn = lambda v: v):
        """
        Finds the shortest path using uniform cost edges between source and goal on graph.

        Parameters
        ----------
        graph : Graph
            Graph to run BFS on.
        source : Vertex
            Vertex to start running BFS from i.e. this function will run shortest paths from this vertex.
        goal : Vertex
            Vertex to reach.
        tieBreakFn : Function
            Break ties between vertices to consider. Default is breaking ties alphabetically.
        preApplyFn : Function
            Extrinsic function that is applied the first time a vertex is added to the Queue.
        postApplyFn : Function
            Extrinsic function that is applied as soon as a vertex is popped off the Queue.

        Returns
        -------
        pathToGoal : list
            List of vertices to traverse from the source to the goal.

        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2))])
        >>> bfs = BFS()
        >>> bfs.shortestPathToGoal(g, Vertex(0), Vertex(2))
        [Vertex(0), Vertex(2)]
        >>> bfs.shortestPathToGoal(g, Vertex(0), Vertex(1))
        [Vertex(0), Vertex(1)]

        >>> g = AdjListGraph(5, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2)), Edge(Vertex(2), Vertex(3)), Edge(Vertex(1), Vertex(4)), Edge(Vertex(3), Vertex(4)), Edge(Vertex(4), Vertex(1))])
        >>> bfs.shortestPathToGoal(g, Vertex(0), Vertex(4))
        [Vertex(0), Vertex(1), Vertex(4)]
        """
        paths = {source : []}
        current = source

        def preApplyFnDist(v):
            nonlocal paths
            paths[v] = paths[current] + [v]
            preApplyFn(v)

        def postApplyFnDist(v):
            nonlocal current
            current = v
            postApplyFn(v)

        self._genericBFS(graph, source, None, tieBreakFn, preApplyFnDist, postApplyFnDist)

        return paths[goal]
        
    def _genericBFS(
        self,
        graph : Graph,
        source : Vertex,
        goal : Vertex = None,
        tieBreakFn = lambda listOfVertices: listOfVertices.sort(key = lambda v: v._name),
        preApplyFn = lambda v: v,
        postApplyFn = lambda v: v):
        """
        Runs the generic BFS graph traversal algorithm.

        Parameters
        ----------
        graph : Graph
            Graph to run BFS on.
        source : Vertex
            Vertex to start running BFS from i.e. this function will run shortest paths from this vertex.
        goal : Vertex
            Vertex to reach (if applicable).
        tieBreakFn : Function
            Break ties between vertices to consider. Default is breaking ties alphabetically.
        preApplyFn : Function
            Extrinsic function that is applied the first time a vertex is added to the Queue.
        postApplyFn : Function
            Extrinsic function that is applied as soon as a vertex is popped off the Queue.

        """
        q = Queue()
        visited = [False] * graph._numVertices
        q.addLast(source)
        visited[source._name] = True
        preApplyFn(source)
        while q.size():
            currVertex = q.removeFirst()
            postApplyFn(currVertex)
            if goal and currVertex == goal:
                return

            adjVertices = [edge._v for edge in graph.getEdges(currVertex)]
            tieBreakFn(adjVertices)

            for vertex in adjVertices:
                if not visited[vertex._name]:
                    q.addLast(vertex)
                    preApplyFn(vertex)
                    visited[vertex._name] = True

class DFS:

    def __init__(self):
        pass

    def prePostOrdering(
        self,
        graph : Graph,
        source : Vertex,
        preApplyFn = lambda v: v,
        postApplyFn = lambda v: v):
        """
        Pre and post orders every Vertex reachable from source.

        Parameters
        ----------
        graph : Graph
            Graph to run DFS on.
        source : Vertex
            Vertex to start running DFS from i.e. this function will run shortest paths from this vertex.
        goal : Vertex
            Vertex to reach (if applicable).
        preApplyFn : Function
            Extrinsic function that is applied the first time a vertex is added to the Stack.
        postApplyFn : Function
            Extrinsic function that is applied as soon as a vertex is popped off the Stack.

        Returns
        -------
        orderings : Dict
            Dictionary of all the pre and post orderings for all vertices reachable from the source.

        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2))])
        >>> orderings = DFS().prePostOrdering(g, Vertex(0))
        >>> orderings[Vertex(0)]
        [0, 5]
        >>> orderings[Vertex(1)]
        [1, 4]
        >>> orderings[Vertex(2)]
        [2, 3]

        >>> g = AdjListGraph(5, [Edge(Vertex(0), Vertex(1)), Edge(Vertex(0), Vertex(2)), Edge(Vertex(1), Vertex(2)), Edge(Vertex(2), Vertex(3)), Edge(Vertex(1), Vertex(4)), Edge(Vertex(3), Vertex(4)), Edge(Vertex(4), Vertex(1))])
        >>> orderings = DFS().prePostOrdering(g, Vertex(0))
        >>> orderings[Vertex(0)]
        [0, 9]
        >>> orderings[Vertex(1)]
        [1, 8]
        >>> orderings[Vertex(2)]
        [2, 7]
        >>> orderings[Vertex(3)]
        [3, 6]
        >>> orderings[Vertex(4)]
        [4, 5]
        """
        orderings = {}
        count = 0
        for v in graph.getVertices():
            orderings[v] = [-1, -1]

        def preApplyFnOrdering(v):
            nonlocal count
            orderings[v][0] = count
            count += 1
            preApplyFn(v)

        def postApplyFnOrdering(v):
            nonlocal count
            orderings[v][1] = count
            count += 1
            postApplyFn(v)

        self._genericDFS(graph, source, None, preApplyFnOrdering, postApplyFnOrdering)

        return orderings

    def _genericDFS(
        self,
        graph : Graph,
        source : Vertex,
        goal : Vertex = None,
        preApplyFn = lambda v: v,
        postApplyFn = lambda v: v):
        """
        Runs the generic DFS graph traversal algorithm.

        Parameters
        ----------
        graph : Graph
            Graph to run DFS on.
        source : Vertex
            Vertex to start running DFS from i.e. this function will run shortest paths from this vertex.
        goal : Vertex
            Vertex to reach (if applicable).
        preApplyFn : Function
            Extrinsic function that is applied the first time a vertex is added to the Stack.
        postApplyFn : Function
            Extrinsic function that is applied as soon as a vertex is popped off the Stack.

        """
        def visit(v):
            preApplyFn(v)
            visited[v._name] = True
            if goal and v == goal:
                return
            
            adjVertices = [edge._v for edge in graph.getEdges(v)]

            for vertex in adjVertices:
                if not visited[vertex._name]:
                    visit(vertex)
            postApplyFn(v)

        visited = [False] * graph._numVertices
        visit(source)

class Dijkstra:

    def __init__(self):
        pass

    def shortestPaths(self, graph : Graph, source : Vertex):
        """
        Runs Dijkstra's algorithm from source to every vertex reachable from source to find the minimum costing path from source.

        Parameters
        ----------
        graph : Graph
            Graph to run Dijkstra's on.
        source : Vertex
            Vertex to run Dijkstra's from.

        Returns
        -------
        paths : list
            List of lists of edges that denote the minimum cost path from source to every vertex reachable from source.
        dists : list
            List of distances that denote the minimum cost from source to every vertex reachable from source.

        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1), 1), Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(1), Vertex(2), 1)])
        >>> paths, dists = Dijkstra().shortestPaths(g, Vertex(0))
        >>> paths[Vertex(0)]
        []
        >>> paths[Vertex(1)]
        [Edge(Vertex(0), Vertex(1), 1)]
        >>> paths[Vertex(2)]
        [Edge(Vertex(0), Vertex(1), 1), Edge(Vertex(1), Vertex(2), 1)]

        >>> g = AdjListGraph(5, [Edge(Vertex(0), Vertex(1), 10), Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(1), Vertex(2), 1), Edge(Vertex(2), Vertex(3), 1), Edge(Vertex(1), Vertex(4), 1), Edge(Vertex(3), Vertex(4), 2), Edge(Vertex(4), Vertex(1), 2)])
        >>> paths, dists = Dijkstra().shortestPaths(g, Vertex(0))
        >>> paths[Vertex(0)]
        []
        >>> paths[Vertex(1)]
        [Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(2), Vertex(3), 1), Edge(Vertex(3), Vertex(4), 2), Edge(Vertex(4), Vertex(1), 2)]
        >>> paths[Vertex(2)]
        [Edge(Vertex(0), Vertex(2), 3)]
        >>> paths[Vertex(3)]
        [Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(2), Vertex(3), 1)]
        >>> paths[Vertex(4)]
        [Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(2), Vertex(3), 1), Edge(Vertex(3), Vertex(4), 2)]

        """
        pq = Heap(type = 'min')
        paths = {}
        dists = {}

        for v in graph.getVertices():
            dists[v] = float('inf')
            pq.addVal(v, float('inf'))
        pq.updateVal(source, 0)
        dists[source] = 0
        paths[source] = []

        while pq.size():
            currVertex = pq.pop()
            for edge in graph.getEdges(currVertex):
                if pq.contains(edge._v):
                    if dists[currVertex] + edge._weight < dists[edge._v]:
                        dists[edge._v] = dists[currVertex] + edge._weight
                        paths[edge._v] = paths[currVertex].copy() + [edge]
                        pq.updateVal(edge._v, dists[edge._v])
        return paths, dists

    def shortestDists(self, graph : Graph, source : Vertex):
        """
        Runs Dijkstra's algorithm from source to every vertex reachable from source to find the minimum cost to that vertex.

        Parameters
        ----------
        graph : Graph
            Graph to run Dijkstra's on.
        source : Vertex
            Vertex to run Dijkstra's from.

        Returns
        -------
        dists : list
            List of distances that denote the minimum cost from source to every vertex reachable from source.

        >>> g = AdjListGraph(3, [Edge(Vertex(0), Vertex(1), 1), Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(1), Vertex(2), 1)])
        >>> dists = Dijkstra().shortestDists(g, Vertex(0))
        >>> dists[Vertex(0)]
        0
        >>> dists[Vertex(1)]
        1
        >>> dists[Vertex(2)]
        2

        >>> g = AdjListGraph(5, [Edge(Vertex(0), Vertex(1), 10), Edge(Vertex(0), Vertex(2), 3), Edge(Vertex(1), Vertex(2), 1), Edge(Vertex(2), Vertex(3), 1), Edge(Vertex(1), Vertex(4), 1), Edge(Vertex(3), Vertex(4), 2), Edge(Vertex(4), Vertex(1), 2)])
        >>> dists = Dijkstra().shortestDists(g, Vertex(0))
        >>> dists[Vertex(0)]
        0
        >>> dists[Vertex(1)]
        8
        >>> dists[Vertex(2)]
        3
        >>> dists[Vertex(3)]
        4
        >>> dists[Vertex(4)]
        6

        """
        pq = Heap(type = 'min')
        dists = {}

        for v in graph.getVertices():
            dists[v] = float('inf')
            pq.addVal(v, float('inf'))
        pq.updateVal(source, 0)
        dists[source] = 0

        while pq.size():
            currVertex = pq.pop()
            for edge in graph.getEdges(currVertex):
                if pq.contains(edge._v):
                    dists[edge._v] = min((dists[edge._v], dists[currVertex] + edge._weight))
                    pq.updateVal(edge._v, dists[edge._v])
        return dists

class TreeTraversals:
    """
    This class is full of tree traversals written iteratively and recursively.
    They currently only work for binary trees!
    """

    def inorderRecursive(self, root : TreeNode):
        vals = []
        if root:
            self.inorderRecursiveHelper(root, vals)
        return vals

    def inorderRecursiveHelper(self, root, vals):
        if root.left:
            self.inorderRecursiveHelper(root.left, vals)
        vals.append(root.val)
        if root.right:
            self.inorderRecursiveHelper(root.right, vals)

    def inorderIterative(self, root : TreeNode):
        """
        >>> ts, tt = Trees(), TreeTraversals()
        >>> trials, p = 50, 0.8
        >>> for _ in range(trials):
        ...     t = ts.randomBinaryTree(p).root
        ...     assert tt.inorderIterative(t) == tt.inorderRecursive(t)
        
        """
        stack, curr, vals = [], root, []
        while stack or curr:
            while curr:
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            vals.append(curr.val)
            curr = curr.right
        return vals

    def preorderRecursive(self, root : TreeNode):
        vals = []
        if root:
            self.preorderRecursiveHelper(root, vals)
        return vals

    def preorderRecursiveHelper(self, root, vals):
        vals.append(root.val)
        if root.left:
            self.preorderRecursiveHelper(root.left, vals)
        if root.right:
            self.preorderRecursiveHelper(root.right, vals)

    def preorderIterative(self, root : TreeNode):
        """
        >>> ts, tt = Trees(), TreeTraversals()
        >>> trials, p = 50, 0.8
        >>> for _ in range(trials):
        ...     t = ts.randomBinaryTree(p).root
        ...     assert tt.preorderIterative(t) == tt.preorderRecursive(t)
        
        """
        vals = []
        if root:
            stack = [root]
            while stack:
                curr = stack.pop()
                vals.append(curr.val)
                if curr.right:
                    stack.append(curr.right)
                if curr.left:
                    stack.append(curr.left)
        return vals

    def preorderIterative2(self, root : TreeNode):
        """
        >>> ts, tt = Trees(), TreeTraversals()
        >>> trials, p = 50, 0.8
        >>> for _ in range(trials):
        ...     t = ts.randomBinaryTree(p).root
        ...     assert tt.preorderIterative2(t) == tt.preorderRecursive(t)
        
        """
        stack, curr, vals = [], root, []
        while stack or curr:
            while curr:
                vals.append(curr.val)
                stack.append(curr)
                curr = curr.left
            curr = stack.pop()
            curr = curr.right
        return vals

    def postorderRecursive(self, root : TreeNode):
        vals = []
        if root:
            self.postorderRecursiveHelper(root, vals)
        return vals

    def postorderRecursiveHelper(self, root, vals):
        if root.left:
            self.postorderRecursiveHelper(root.left, vals)
        if root.right:
            self.postorderRecursiveHelper(root.right, vals)
        vals.append(root.val)

    def postOrderIterative(self, root : TreeNode):
        """
        >>> ts, tt = Trees(), TreeTraversals()
        >>> trials, p = 50, 0.8
        >>> for _ in range(trials):
        ...     t = ts.randomBinaryTree(p).root
        ...     assert tt.postOrderIterative(t) == tt.postorderRecursive(t)
        
        """
        stack, curr, vals = [], root, []
        while stack or curr:
            while curr:
                vals.append(curr.val)
                stack.append(curr)
                curr = curr.right
            curr = stack.pop()
            curr = curr.left
        vals.reverse()
        return vals

    def postOrderIterative2(self, root : TreeNode):
        """
        >>> ts, tt = Trees(), TreeTraversals()
        >>> trials, p = 50, 0.8
        >>> for _ in range(trials):
        ...     t = ts.randomBinaryTree(p).root
        ...     assert tt.postOrderIterative2(t) == tt.postorderRecursive(t)
        
        """
        stack, curr, vals = [], root, []
        while stack or curr:
            while curr:
                if curr.right:
                    stack.append(curr.right)
                stack.append(curr)
                curr = curr.left 
            curr = stack.pop()
            if curr.right and stack and stack[-1] == curr.right:
                curr, stack[-1] = stack[-1], curr
            else:
                vals.append(curr.val)
                curr = None
        return vals

    def levelOrderIterative(self, root : TreeNode):
        vals = []
        if root:
            queue = [root]
            while queue:
                for _ in range(len(queue)):
                    curr = queue.pop(0)
                    vals.append(curr.val)
                    if curr.left: queue.append(curr.left)
                    if curr.right: queue.append(curr.right)
        return vals

    def levelOrderRecursive(self, root : TreeNode):
        """
        >>> ts, tt = Trees(), TreeTraversals()
        >>> trials, p = 50, 0.8
        >>> for _ in range(trials):
        ...     t = ts.randomBinaryTree(p).root
        ...     assert tt.levelOrderRecursive(t) == tt.levelOrderIterative(t)
        
        """
        levels = []
        if root:
            self.levelOrderRecursiveHelper(root, 0, levels)
        vals = []
        for level in levels:
            vals.extend(level)
        return vals

    def levelOrderRecursiveHelper(self, root, level, levels):
        if len(levels) <= level:
            levels.append([])
        levels[level].append(root.val)
        if root.left: self.levelOrderRecursiveHelper(root.left, level + 1, levels)
        if root.right: self.levelOrderRecursiveHelper(root.right, level + 1, levels)

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    