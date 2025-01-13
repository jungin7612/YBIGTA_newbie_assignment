from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        Sets the value of an element in the matrix at the specified position.

        Args:
            key (tuple[int, int]): A tuple representing the row and column indices (i, j) of the element to set.
            value (int): The value to set at the specified position. The value will be stored modulo `self.MOD`.

        Returns:
            None
        """
        self.matrix[key[0]][key[1]] = value % self.MOD
        pass

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        Raises the matrix to the power of n using exponentiation by squaring.

        Args:
            n (int): The exponent to which the matrix is raised. Must be a non-negative integer.

        Returns:
            Matrix: A new matrix that is the result of raising the current matrix to the power of n.

        """
        result = Matrix.eye(self.shape[0])
        base = self.clone()

        while n > 0:
            if n % 2 == 1:
                result = result @ base
            base = base @ base
            n = n // 2
        return result


    def __repr__(self) -> str:
        """
        Returns a string representation of the matrix.

        Each row of the matrix is represented as a space-separated string of values,
        and rows are joined by newline characters.

        Returns:
            str: A string representation of the matrix.
        """
        rows_as_str = []
        for row in self.matrix:
            row_str = ' '.join(str(item) for item in row)
            rows_as_str.append(row_str)
        return '\n'.join(rows_as_str)