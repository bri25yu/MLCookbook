"""
@author bri25yu
"""

import random
from bisect import bisect_left, insort

class TreeNode:
    
    def __init__(self, val, left = None, right = None):
        self.val, self.left, self.right = val, left, right

class Tree:

    def __init__(self, root : TreeNode = None):
        self.root = root

    def print(self, spacePerLevel = [2]):
        self.printHelper(self.root, spacePerLevel, 0)

    def printHelper(self, root, spacePerLevel, space):
        if root:
            space += spacePerLevel[0]
            
            self.printHelper(root.right, spacePerLevel, space)
            print()
            for _ in range(spacePerLevel[0], space):
                print(end = " ")
            print(root.val)
            self.printHelper(root.left, spacePerLevel, space) 

    def deserializeTree(self, string : str, emptyVal = 'null'):
        """
        Credit to @StefanPochmann from LeetCode!
        """
        if string == '{}':
            return None
        nodes = [None if val == emptyVal else TreeNode(int(val))
                for val in string.strip('[]{}').split(',')]
        kids = nodes[::-1]
        root = kids.pop()
        for node in nodes:
            if node:
                if kids: node.left  = kids.pop()
                if kids: node.right = kids.pop()
        return Tree(root)

class Trees:

    def randomBinaryTree(
                            self,
                            p : float,
                            minVal : int = 0,
                            maxVal : int = 50,
                            maxNodes : int = 100
                        ) -> Tree:
        """
        Produces a random binary tree with nodes existing with probability p [0, 1)
            and node values ranging from [minVal, maxVal].

        Parameters
        ----------
        p : float
            Probability that a node exists.
        minVal : int
            Minimum value of a node.
        maxVal : int
            Maximum value of a node.
        maxNodes : int
            Maximum number of nodes in the tree.

        Returns
        -------
        t : Tree
            A binary tree.

        >>> ts = Trees()
        >>> t = ts.randomBinaryTree(0)
        >>> t = ts.randomBinaryTree(0.25)
        >>> t = ts.randomBinaryTree(0.5)
        >>> t = ts.randomBinaryTree(0.75)
        >>> t = ts.randomBinaryTree(0.9)

        """
        return Tree(self.randomBinaryTreeHelper(p, minVal, maxVal, [0], maxNodes))

    def randomBinaryTreeHelper(self, p, minVal, maxVal, count, m):
        if count[0] < m and random.random() < p:
            r = TreeNode(random.randint(minVal, maxVal))
            count[0] += 1
            r.left = self.randomBinaryTreeHelper(p, minVal, maxVal, count, m)
            r.right = self.randomBinaryTreeHelper(p, minVal, maxVal, count, m)
            return r
        return None

class Queue:
    """
    A queue data structure that follows the First-In-Last-Out (FILO) paradigm.

    >>> q = Queue()
    >>> q.addLast(3)
    >>> q.addLast(1)
    >>> q.addLast(2.5)
    >>> q.removeFirst()
    3
    >>> q.removeFirst()
    1
    >>> q.removeFirst()
    2.5
    >>> q.removeFirst()
    >>> q.addLast(5)
    >>> q.removeFirst()
    5
    """

    class Node:

        def __init__(self, val):
            self._val = val
            self._next = None

    def __init__(self):
        self._root = self.Node(None)
        self._last = self._root
        self._size = 0

    def addLast(self, val):
        """
        Adds the input value to a Node at the end of the queue.

        Parameters
        ----------
        val : object
            Value to add to the end of the queue.

        >>> q = Queue()
        >>> q.addLast(3)
        >>> q.removeFirst()
        3
        """
        self._last._next = self.Node(val)
        self._last = self._last._next
        self._size += 1

    def removeFirst(self):
        """
        Removes and returns the first value in the queue.

        Returns
        -------
        The value at the front of the queue.

        >>> q = Queue()
        >>> q.removeFirst()
        >>> q.addLast(3)
        >>> q.removeFirst()
        3
        """
        toReturn = self._root._next
        if toReturn:
            self._root._next = toReturn._next
            toReturn._next = None
            if toReturn == self._last:
                self._last = self._root
            toReturn = toReturn._val
            self._size -= 1
        return toReturn

    def size(self):
        return self._size

    def display(self):
        """
        Prints the current values in the queue in Queue order.

        >>> q = Queue()
        >>> q.addLast(3)
        >>> q.addLast(2.5)
        >>> q.addLast(1)
        >>> q.display()
        3 2.5 1
        >>> q.removeFirst()
        3
        >>> q.display()
        2.5 1
        """
        curr = self._root._next
        toDisplay = ''
        while curr:
            toDisplay += '{} '.format(curr._val)
            curr = curr._next
        print(toDisplay[:-1])

class Stack:
    """
    >>> s = Stack()
    >>> s.addFirst(3)
    >>> s.addFirst(2.5)
    >>> s.addFirst(1)
    >>> s.removeFirst()
    1
    >>> s.removeFirst()
    2.5
    >>> s.removeFirst()
    3
    >>> s.removeFirst()
    >>> s.addFirst(10)
    >>> s.removeFirst()
    10
    """

    class Node:

        def __init__(self, val):
            self._val = val
            self._next = None

    def __init__(self):
        self._root = self.Node(None)
        self._size = 0

    def addFirst(self, val):
        """
        >>> s = Stack()
        >>> s.addFirst(3)
        >>> s.addFirst(5)
        >>> s.removeFirst()
        5
        >>> s.removeFirst()
        3
        """
        val = self.Node(val)
        val._next = self._root._next
        self._root._next = val
        self._size += 1

    def removeFirst(self):
        """
        >>> s = Stack()
        >>> s.addFirst(3)
        >>> s.removeFirst()
        3
        >>> s.removeFirst()
        """
        toReturn = self._root._next
        if toReturn:
            self._root._next = toReturn._next
            toReturn._next = None
            toReturn = toReturn._val
            self._size -= 1
        return toReturn

    def size(self):
        return self._size

    def display(self):
        """
        Prints all the values in the stack in Stack order.

        >>> s = Stack()
        >>> s.addFirst(3)
        >>> s.addFirst(2.5)
        >>> s.addFirst(1)
        >>> s.display()
        1 2.5 3
        >>> s.removeFirst()
        1
        >>> s.display()
        2.5 3
        """
        curr = self._root._next
        toDisplay = ''
        while curr:
            toDisplay += '{} '.format(curr._val)
            curr = curr._next
        print(toDisplay[:-1])

class Heap:
    """
    >>> h = Heap()
    >>> h.addVal(3, 3)
    >>> h.addVal(2, 4)
    >>> h.addVal(1, 3)
    >>> h.pop()
    2
    >>> h.pop()
    1
    """
    
    class Node:
        """
        Data encapsulation class for values and their respective priorities.
        """

        def __init__(self, val, priority : float):
            """
            Creates a new Node.

            Parameters
            ----------
            val : object
                Value to store.
            priority : float
                Priority of value to store.

            """
            self._val = val
            self._priority = priority

    def __init__(self, type = 'max'):
        """
        Creates a new instance of the Heap class.

        Parameters
        ----------
        type : str
            Type (min or max) of heap to create. A 'max' heap will prioritize higher priorities, and vice versa.

        """
        assert type == 'max' or type == 'min', 'Heap must have a max or min ranking.'

        self._type = type
        comparatorFunc = max if type == 'max' else min
        self._comparator = lambda v1, v2: comparatorFunc((v1, v2), key = lambda x: x._priority)
        self._vals = []

    def addVal(self, item, priority : float):
        """
        Adds the item with priority to the heap.

        Parameters
        ----------
        item : object
            Item to add to the heap.
        priority : float
            Priority of item to add to the heap.

        >>> h = Heap()
        >>> h.addVal(1, 2)
        >>> h.addVal(1, 3)
        Traceback (most recent call last):
        ...
        AssertionError: Can only add items not already in the heap to the heap.

        """
        assert self._find(item) == -1, 'Can only add items not already in the heap to the heap.'
        self._addVal(item, priority)

    def updateVal(self, item, priority : float):
        """
        Updates the item in the heap to the input priority if the item is in the heap.
        Otherwise, adds the item to the heap.

        Parameters
        ----------
        item : object
            Item to update.
        priority : float
            Priority value to update to.

        >>> h = Heap()
        >>> h.addVal(1, 3)
        >>> h.addVal(2, 2)
        >>> h.updateVal(1, 1)
        >>> [h.pop() for _ in range(h.size())]
        [2, 1]

        """
        nodeIndex = self._find(item)
        if nodeIndex != -1:
            self._vals[nodeIndex]._priority = priority
            self._sink(nodeIndex)
            self._swim(nodeIndex)
        else:
            self._addVal(item, priority)

    def contains(self, item):
        """
        Determines whether or not the heap contains the input item.

        Parameters
        ----------
        item : object
            Item to search for.

        Returns
        -------
        found : bool
            Whether or not the input item was found in the heap or not.

        """
        return self._find(item) != -1

    def pop(self):
        """
        Returns the value of the highest/lowest priority item for max/min heaps, respectively.

        Returns
        -------
        val : object
            Value of the object with the most preferred priority.
        
        >>> h = Heap()
        >>> h.pop()
        Traceback (most recent call last):
        ...
        AssertionError: Heap is empty, so cannot return any values!
        >>> h.addVal(4, 3)
        >>> h.pop()
        4

        """
        assert len(self._vals) > 0, 'Heap is empty, so cannot return any values!'
        if len(self._vals) == 1:
            return self._vals.pop(0)._val
        
        toReturn, toReplace = self._vals[0], self._vals.pop()
        self._vals[0] = toReplace
        self._sink(0)
        return toReturn._val

    def size(self):
        """
        Returns the size of the current heap.

        Returns
        -------
        size : int
            Number of items stored in the heap.

        """
        return len(self._vals)

    def _addVal(self, item, priority : float):
        """
        Helper function for adding the input item with priority into the heap.

        Parameters
        ----------
        item : object
            Item to add to the heap.
        priority : float
            Priority of item to add to the heap.

        >>> h = Heap()
        >>> h.addVal(3, 3)
        >>> h.addVal(2, 4)
        >>> h.addVal(17, 17)
        >>> [h.pop() for _ in range(h.size())]
        [17, 2, 3]
        """
        self._vals.append(self.Node(item, priority))
        self._swim(len(self._vals) - 1)

    def _parent(self, index : int):
        """
        Returns the parent index of the input index.

        Parameters
        ----------
        index : int
            Index to find the parent of.

        Returns
        -------
        parent : int
            The parent index of the input index.

        >>> h = Heap()
        >>> h._parent(3) == h._parent(4) == 1
        True
        >>> h._parent(5) == h._parent(6) == 2
        True
        >>> h._parent(0)
        -1
        """
        return (index - 1) // 2

    def _children(self, index : int):
        """
        Returns a list of the children of the input index, specifically in the order of [leftChild, rightChild].

        Parameters
        ----------
        index : int
            Index to find the children of.

        Returns
        -------
        children : list
            List of the children of the input index.

        >>> h = Heap()
        >>> h._vals.extend([1, 2, 3])
        >>> h._children(0)
        [1, 2]
        >>> h._children(17)
        []
        """
        childIndices = []
        leftPossible = index * 2 + 1
        if leftPossible < len(self._vals):
            childIndices.append(leftPossible)
            
        rightPossible = index * 2 + 2
        if rightPossible < len(self._vals):
            childIndices.append(rightPossible)

        return childIndices

    def _swim(self, index : int):
        """
        Moves the node at the input index up so that the heap property is conserved.

        Parameters
        ----------
        index : int
            Index of (item, priority) pair to move upwards.

        >>> h = Heap()
        >>> h.addVal(3, 3)
        >>> h._find(3)
        0
        >>> h.addVal(2, 4)
        >>> h._find(3)
        1

        """
        parentIndex = self._parent(index)
        if parentIndex > -1:
            current, parent = self._vals[index], self._vals[parentIndex]
            if current == self._comparator(current, parent):
                self._vals[index], self._vals[parentIndex] = parent, current
                self._swim(parentIndex)

    def _sink(self, index):
        """
        Moves the node at the input index down so that the heap property is conserved.

        Parameters
        ----------
        index : int
            Index of (item, priority) pair to move downwards.

        >>> h = Heap()
        >>> h.addVal(3, 5)
        >>> h.addVal(4, 4)
        >>> h._find(3)
        0
        >>> h.updateVal(3, 3)
        >>> h._find(3)
        1

        """
        childrenIndices = self._children(index)
        current = self._vals[index]

        swapChild, swapChildIndex = current, -1
        for childIndex in childrenIndices:
            swapChild = self._comparator(swapChild, self._vals[childIndex])
            if swapChild == self._vals[childIndex]:
                swapChildIndex = childIndex

        if current != self._comparator(current, swapChild):
            self._vals[index], self._vals[swapChildIndex] = self._vals[swapChildIndex], self._vals[index]
            self._sink(swapChildIndex)

    def _find(self, item):
        """
        Returns the index that holds item.

        Parameters
        ----------
        item : object
            Item to search for.

        Returns
        -------
        index : int
            Index of the object in the list, or -1 if it isn't found.

        >>> h = Heap()
        >>> h._find(3)
        -1
        >>> h.addVal(3, 3)
        >>> h._find(3)
        0
        >>> h.addVal(1, 1)
        >>> h._find(1)
        1
        >>> h.updateVal(3, 0)
        >>> h._find(1)
        0
        """
        for i in range(len(self._vals)):
            if self._vals[i]._val == item:
                return i
        return -1

class PriorityQueue:
    """
    Only works with hashable and comparable values.

    >>> pq = PriorityQueue()
    >>> pq.add(2, 2)
    >>> pq.add(1, 1)
    >>> pq.add(3, 3)
    >>> [pq.pop() for _ in range(3)]
    [1, 2, 3]
    >>> len(pq.vals)
    0
    >>> import random
    >>> random.seed(1234)
    >>> l = list(range(10000))
    >>> l_sorted = l.copy()
    >>> random.shuffle(l)
    >>> for val in l:
    ...     pq.add(val, val + 1)
    ...
    >>> len(pq.vals)
    10000
    >>> [pq.pop() for _ in range(len(l_sorted))] == l_sorted
    True

    """

    def __init__(self):
        self.vals, self.mappings = [], dict()

    def add(self, val, priority):
        if val in self.mappings: self.vals.pop(bisect_left(self.vals, [self.mappings[val], val]))
        self.mappings[val] = priority
        insort(self.vals, [priority, val])

    def pop(self):
        if self.vals:
            val = self.vals.pop(0)[1]
            self.mappings.pop(val)
            return val
        raise IndexError('Popping from an empty Queue')

class ImplicitHeap(Heap):
    pass

class FibonacciHeap(Heap):
    """
    https://en.wikipedia.org/wiki/Fibonacci_heap
    """
    pass

class Factorial:

    def __init__(self, initialMaxFactorial : int = 100):
        """
        Creates a Factorial class with memoized values for 0! to 100!.

        Parameters
        ----------
        initialMaxFactorial : int
            Initial maximum value to memoize and calculate to.

        >>> len(Factorial(10)._factorials) == 11
        True
        >>> len(Factorial(0)._factorials) == 1
        True
        >>> Factorial(-5)
        Traceback (most recent call last):
        ...
        AssertionError: Initial size must be nonnegative.

        """
        assert initialMaxFactorial >= 0, 'Initial size must be nonnegative.'
        self._factorials = [1]
        self._growTo(initialMaxFactorial)

    def get(self, num : int):
        """
        Gets the factorial value of num!. If it isn't yet calculated, it will calculate it on the fly and memoize the results.

        Parameters
        ----------
        num : int
            Number to get the factorial of.

        Returns
        -------
        numFactorial : int
            num!

        >>> f = Factorial(10)
        >>> f.get(7)
        5040
        >>> f.get(9)
        362880
        >>> f.get(10)
        3628800
        >>> f.get(11)
        39916800

        """
        if num >= len(self._factorials):
            self._growTo(num)
        return self._factorials[num]

    def _growTo(self, toGrowTo : int):
        for _ in range(toGrowTo - len(self._factorials) + 1):
            self._grow()

    def _grow(self):
        self._factorials.append(self._factorials[-1] * len(self._factorials))

class DisjointSet:
    """
    14 line DisjointSet implementation! Kinda bare-bones, even without size lol

    >>> ds = DisjointSet()
    >>> ds.connected('a', 'b')
    False
    >>> ds.union('a', 'b')
    >>> ds.connected('a', 'b')
    True
    >>> ds.connected('b', 'a')
    True
    >>> ds.connected('b', 'c')
    False
    >>> ds.connected('c', 'b')
    False
    >>> ds.connected('a', 'c')
    False
    >>> ds.union('a', 'c')
    >>> ds.connected('b', 'c')
    True
    >>> ds.connected('c', 'b')
    True
    >>> ds.connected('a', 'c')
    True

    """
    class Node:
        def __init__(self, val): self.val = val

    def __init__(self): self.m = {}
    def exist(self, a): return self.m.setdefault(a, self.Node(-1))
    def connected(self, a, b): return self.find(a) == self.find(b)
        
    def union(self, a, b):
        if (a:=self.find(a)) != (b:=self.find(b)):
            if self.m[b].val < self.m[a].val: a, b = b, a
            self.m[a].val, self.m[b] = self.m[a].val + self.m[b].val, a
        
    def find(self, a):
        if isinstance(self.exist(a), self.Node): return a
        self.m[a] = self.find(self.m[a])
        return self.m[a]

class Trie:
    """
    Small Trie implementation that supports exact searching and prefix searching.

    >>> t = Trie()
    >>> t.search('trie')
    False
    >>> t.add('trie')
    >>> t.search('trie')
    True
    >>> t.search('tri')
    False
    >>> t.search('tries')
    False
    >>> t.search('')
    False
    >>> t.add('triumph')
    >>> t.add('triangle')
    >>> t.add('triage')
    >>> t.add('triceratops')
    >>> t.add('trick')
    >>> t.search('trie')
    True
    >>> t.search('triumph')
    True
    >>> t.search('triangle')
    True
    >>> t.search('triage')
    True
    >>> t.search('triceratops')
    True
    >>> t.search('trick')
    True
    >>> t.search('tries')
    False
    >>> t.prefixSearch('triangle')
    ['triangle']
    >>> t.prefixSearch('triage')
    ['triage']
    >>> len(t.prefixSearch('tri'))
    6
    >>> len(t.prefixSearch(''))
    6
    >>> len(t.prefixSearch('tria'))
    2
    >>> len(t.prefixSearch('tric'))
    2
    >>> len(t.prefixSearch('trice'))
    1
    
    """
    root = {}
    def add(self, word): self._add(self.root, word)
    def search(self, word): return bool((pot:=self._find(self.root, word)) and '' in pot)

    def prefixSearch(self, p):
        if start:=self._find(self.root, p):
            self._BFS(start, p, total:=[])
            return total

    def _add(self, r, w):
        if w: self._add(r.setdefault(w[0], {}), w[1:])
        else: r[''] = None

    def _find(self, r, w): return (self._find(r[w[0]], w[1:]) if w[0] in r else None) if w else r

    def _BFS(self, curr, p, total):
        if '' in curr: total.append(p)
        for k, v in curr.items():
            if k: self._BFS(v, p + k, total)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    