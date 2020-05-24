"""
@author bri25yu
"""

import math

class Matrix2D:

    def __init__(self, arr : list):
        self._dim = self._validate2DMatrix(arr)
        self._arr = arr

    def _validate2DMatrix(self, matrix : list):
        """
        Validates that arr is a matrix.

        Parameters
        ----------
        matrix : list
            List to validate as a proper 2D matrix.

        Returns
        -------
        dim : tuple
            The size of the matrix if the input matrix is a valid matrix.

        >>> m = Matrix2D([[1]])
        >>> m._validate2DMatrix([1])
        Traceback (most recent call last):
        ...
        AssertionError: Must be a list of lists!
        >>> m._validate2DMatrix([[1], 1])
        Traceback (most recent call last):
        ...
        AssertionError: Must be a list of lists!
        >>> m._validate2DMatrix([[]])
        Traceback (most recent call last):
        ...
        AssertionError: Matrix must be non-empty
        >>> m._validate2DMatrix([[1], []])
        Traceback (most recent call last):
        ...
        AssertionError: Matrix must be non-empty
        >>> m._validate2DMatrix([[1, 2], [3]])
        Traceback (most recent call last):
        ...
        AssertionError: Must be a complete matrix!
        >>> m._validate2DMatrix([[3], [1, 2]])
        Traceback (most recent call last):
        ...
        AssertionError: Must be a complete matrix!
        >>> m._validate2DMatrix([[[]]])
        Traceback (most recent call last):
        ...
        AssertionError: Value must be an int or a float: []
        >>> m._validate2DMatrix([[1]])
        (1, 1)
        >>> m._validate2DMatrix([[1, 2], [3, 4]])
        (2, 2)
        >>> m._validate2DMatrix([[1, 2, 3], [4, 5, 6]])
        (2, 3)

        """
        if matrix:
            if isinstance(matrix[0], list):
                size = len(matrix[0])
            else:
                assert False, 'Must be a list of lists!'

            for row in matrix:
                assert isinstance(row, list), 'Must be a list of lists!'
                assert len(row) > 0, 'Matrix must be non-empty'
                assert len(row) == size, 'Must be a complete matrix!'
                for val in row:
                    assert isinstance(val, int) or isinstance(val, float), 'Value must be an int or a float: {}'.format(val)
            return (len(matrix), size)

    def size(self):
        """
        Returns the size of the matrix.
        """
        return self._dim

    def hstack(self, matrix):
        """
        Stacks this matrix and the input matrix horizontally.
        i.e. this matrix on the left and the input matrix on the right.

        Parameters
        ----------
        matrix : Matrix2D
            Matrix to stack to the right of this.

        Returns
        -------
        newArr : Matrix2D
            A new matrix with this matrix and the input matrix stacked horizontally.
            This process is non-destructive.

        >>> m1, m2 = Matrix2D([[1], [3]]), Matrix2D([[2], [4]])
        >>> m1.hstack(m2)
        [[1, 2],
        [3, 4]]
        >>> m2.hstack(m1)
        [[2, 1],
        [4, 3]]
        >>> m2 = Matrix2D([[2]])
        >>> m1.hstack(m2)
        Traceback (most recent call last):
        ...
        AssertionError: Matrix vertical axes must be the same size!

        """
        assert matrix.size()[0] == self.size()[0], "Matrix vertical axes must be the same size!"
        newArr = []
        for i in range(self.size()[0]):
            newArr.append(self._arr[i] + matrix._arr[i])
        return Matrix2D(newArr)

    def vstack(self, matrix):
        """
        Vertically stacks this matrix on top of the input matrix

        Parameters
        ----------
        matrix : Matrix2D
            Matrix to stack on the bottom of this matrix.

        Returns
        -------
        newArr : Matrix2D
            This matrix stacked vertically on top of the input matrix.
            This process is nondestructive.

        >>> m1, m2 = Matrix2D([[1, 3]]), Matrix2D([[2, 4]])
        >>> m1.vstack(m2)
        [[1, 3],
        [2, 4]]
        >>> m2.vstack(m1)
        [[2, 4],
        [1, 3]]
        >>> m2 = Matrix2D([[2]])
        >>> m1.vstack(m2)
        Traceback (most recent call last):
        ...
        AssertionError: Matrix vertical axes must be the same size!

        """
        assert matrix.size()[1] == self.size()[1], "Matrix vertical axes must be the same size!"
        newArr = []
        for row in self._arr:
            newArr.append(row.copy())
        for row in matrix._arr:
            newArr.append(row.copy())
        return Matrix2D(newArr)

    def get(self, val):
        """
        Gets the value of this matrix at the input val.

        Parameters
        ----------
        val : object
            The index to get the value of in this matrix.

        Returns
        -------
        val : object
            If val is an int, returns a Matrix2D of that single row at val.
            If val is a tuple, returns the value at that tuple.
            Otherwise, raises a NotImplementedError.

        """
        if isinstance(val, int):
            return Matrix2D([self._arr[val]])
        elif isinstance(val, tuple):
            assert len(val) == 2, 'Tuple indices must be of length 2 for 2-dim matrices.'
            return self._arr[val[0]][val[1]]
        else:
            raise NotImplementedError("This function is not supported.")

    def subMatrix(self, xRange : tuple, yRange : tuple):
        """
        Gets a sub matrix from this matrix.

        Parameters
        ----------
        xRange : tuple
            The range of x-values to take from this matrix.
        yRange : tuple
            The range of y-values to take from this matrix.

        Returns
        -------
        newArr : Matrix2D
            The submatrix from xRange to yRange.

        >>> m = Matrix2D([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> m.subMatrix((0, 4), (0, 1))
        Traceback (most recent call last):
        ...
        AssertionError: Submatrix indices must be at most the size of the matrix.
        >>> m.subMatrix((2, 1), (0, 1))
        Traceback (most recent call last):
        ...
        AssertionError: Submatrix indices must be valid matrix indices.
        >>> m.subMatrix((0, 2), (0, 1))
        [[1],
        [4]]
        >>> m.subMatrix((0, m.size()[0]), (0, m.size()[1]))
        [[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]]

        """
        assert xRange[0] >= 0 and xRange[1] <= self.size()[0] \
            and yRange[0] >= 0 and yRange[1] <= self.size()[1], 'Submatrix indices must be at most the size of the matrix.'
        assert xRange[1] - xRange[0] > 0 and yRange[1] - yRange[0] > 0, 'Submatrix indices must be valid matrix indices.'
        newArr = []
        for i in range(xRange[0], xRange[1]):
            newArr.append([])
            for j in range(yRange[0], yRange[1]):
                newArr[-1].append(self.get((i, j)))
        return Matrix2D(newArr)

    def inverse(self) -> Matrix2D:
        """
        Returns the inverse of this matrix as a new Matrix2D if this is a square matrix.

        Returns
        -------
        matrixInv : Matrix2D
            The inverse of this matrix.
        """
        assert (s:=self.size())[0] == s[1], 'Matrix must be square to have an inverse.'

    def rowReduce(self, m : Matrix2D) -> Matrix2D:
        """
        Performs row reduction with this matrix as the left hand side and m as the right hand side.

        Parameters
        ----------
        m : Matrix2D
            Right hand side of the row reduction.

        Returns
        -------
        rr : Matrix2D
            The row reduced version of the input m.
            
        """

    def __add__(self, val):
        """
        Adds val to this matrix.
        If val is an int or a float, perform elementwise addition on this matrix with val.
        If val is another Matrix2D, perform matrix addition, where every element in this matrix is added with the corresponding element in val.

        Parameters
        ----------
        val : object
            Value to add to this matrix.

        Returns
        -------
        newArr : Matrix2D
            A matrix added with val.

        >>> m1 = Matrix2D([[2, 3], [4, 5]])
        >>> m1 + 2
        [[4, 5],
        [6, 7]]
        >>> m2 = Matrix2D([[1, 0], [0, 1], [1, 2]])
        >>> m1 + m2
        Traceback (most recent call last):
        ...
        AssertionError: Matrices must have the same size: (2, 2), (3, 2)
        >>> m2 = Matrix2D([[1, 0], [0, 1]])
        >>> m1 + m2
        [[3, 3],
        [4, 6]]
        >>> m2 = Matrix2D([[-3, 4], [1, -2]])
        >>> m1 + m2
        [[-1, 7],
        [5, 3]]

        """
        if isinstance(val, int) or isinstance(val, float):
            newArr = []
            for row in self._arr:
                newArr.append([])
                currRow = newArr[-1]
                for i in range(len(row)):
                    currRow.append(row[i] + val)
            return Matrix2D(newArr)
        elif isinstance(val, Matrix2D):
            assert self.size() == val.size(), "Matrices must have the same size: {0}, {1}".format(self.size(), val.size())
            newArr = []
            for i in range(self.size()[0]):
                newArr.append([])
                for j in range(self.size()[1]):
                    newArr[-1].append(self._arr[i][j] + val._arr[i][j])
            return Matrix2D(newArr)
        else:
            raise NotImplementedError("This function is not supported.")

    def __sub__(self, val):
        return self + (val * -1)

    def __mul__(self, val):
        """
        Multiplies this matrix by val. 
        If val is an int or a float, perform elementwise multiplication on this matrix with val.
        If val is another Matrix2D, perform matrix multiply, where this matrix is right multiplied by the input val.

        Parameters
        ----------
        val : object
            Value to multiply this matrix by.

        Returns
        -------
        newArr : Matrix2D
            A matrix multiplied by val.

        >>> m1 = Matrix2D([[2, 3], [4, 5]])
        >>> m1 * 2
        [[4, 6],
        [8, 10]]
        >>> m2 = Matrix2D([[1, 0], [0, 1], [1, 2]])
        >>> m1 * m2
        Traceback (most recent call last):
        ...
        AssertionError: Matrices must have the same axis of multiplication: (2, 2), (3, 2)
        >>> m2 = Matrix2D([[1, 0], [0, 1]])
        >>> m1 * m2
        [[2, 3],
        [4, 5]]
        >>> m2 = Matrix2D([[-3, 4, 1], [1, -2, 2]])
        >>> m1 * m2
        [[-3, 2, 8],
        [-7, 6, 14]]

        """
        if isinstance(val, int) or isinstance(val, float):
            newArr = []
            for row in self._arr:
                newArr.append([])
                currRow = newArr[-1]
                for i in range(len(row)):
                    currRow.append(row[i] * val)
            return Matrix2D(newArr)
        elif isinstance(val, Matrix2D):
            assert self.size()[1] == val.size()[0], "Matrices must have the same axis of multiplication: {0}, {1}".format(self.size(), val.size())
            newArr = []
            for row in self._arr:
                toAppend = []
                for j in range(val.size()[1]):
                    elemVal = 0
                    for i in range(val.size()[0]):
                        elemVal += row[i] * val._arr[i][j]
                    toAppend.append(elemVal)
                newArr.append(toAppend)
            return Matrix2D(newArr)
        else:
            raise NotImplementedError("This function is not supported.")

    def __div__(self, val):
        if isinstance(val, int) or isinstance(val, float):
            return self * (1 / val)
        else:
            raise NotImplementedError("This function is not supported.")

    def __repr__(self):
        """ TODO: FIX PLEASE
        sizeBefore, sizeAfter = lambda n: int(math.log(n // 1 + 1) + 1), lambda n: int(-math.log(n % 1 + 1) + 1)
        lengths = [[0, 0]] * self.size()[1]
        for i in range(self.size()[0]):
            for j in range(self.size()[1]):
                lengths[j][0] = max((lengths[j][0], sizeBefore(self._arr[i][j])))
                lengths[j][1] = max((lengths[j][1], sizeAfter(self._arr[i][j])))
        toPrint = ''
        for i in range(self.size()[0]):
            for j in range(self.size()[1]):
                curr = self._arr[i][j]
                prepend, append = lengths[j][0] - sizeBefore(curr), lengths[j][1] - sizeAfter(curr)
                toPrint += ' ' * prepend + str(curr) + ' ' * append
            toPrint += ''
        return toPrint[:-1]
        """
        toReturn = '['
        for row in self._arr:
            toReturn += str(row) + ',\n'
        return toReturn[:-2] + ']'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
