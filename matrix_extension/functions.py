# -*- coding: utf-8 -*-
"""
Created on 2023/1/7 19:08

@author: chenjunhan
"""
import sympy as smp
from sympy.matrices.common import ShapeError, MatrixError, NonSquareMatrixError
from typing import Any, Union, Tuple, Sequence


def is_vector(A: smp.Matrix) -> bool:
    """
    Judge whether the A is vector like.
    :param A: sympy.Matrix
    :return: bool
    """
    return True if 1 in A.shape else False


def is_column_vector(A: smp.Matrix) -> bool:
    """
    Judge whether the A is column vector like.
    :param A: sympy.Matrix
    :return: bool
    """
    return True if A.shape[1] == 1 else False


def is_row_vector(A: smp.Matrix) -> bool:
    """
    Judge whether the A is row vector like.
    :param A: sympy.Matrix
    :return: bool
    """
    return True if A.shape[0] == 1 else False


def get_column_vectors(A: smp.Matrix) -> Tuple[smp.Matrix, ...]:
    """
    Get each column vector of A, then gather them together.
    :param A: sympy.Matrix
    :return: a tuple of column vectors
    """
    return tuple(A.col(i) for i in range(A.cols))


def get_row_vectors(A: smp.Matrix) -> Tuple[smp.Matrix, ...]:
    """
    Get each row vector of A, then gather them together.
    :param A: sympy.Matrix
    :return: a tuple of row vectors
    """
    return tuple(A.row(i) for i in range(A.rows))


def get_sturm_sequence(A: smp.Matrix) -> Sequence[Sequence[smp.Float]]:
    """
    Calculate the sturm sequence of a positive definite or indefinite matrix.
    :param A: positive definite or indefinite matrix
    :return: sturm sequence, a 2-D list
    """
    if not (A.is_positive_definite or A.is_indefinite):
        raise MatrixError("The matrix is not positive definite or indefinite!")
    result = []
    for i in range(A.shape[0]):
        result.append(A[:A.shape[0] - i, :A.shape[1] - i].eigenvals(multiple=True))
    return result


def calculate_vector_angle(u: smp.Matrix, v: smp.Matrix) -> Any:
    """
    Calculate the angle between two vectors.
    :param u: sympy.Matrix, vector like
    :param v: sympy.Matrix, vector like
    :return: the value of angle, in radians
    """
    if not (is_vector(u) or is_vector(v)):
        raise ShapeError("Please enter vector like matrix!")
    return smp.trigsimp(smp.acos(u.dot(v) / (u.norm(2) * v.norm(2))))


def calculate_rayleigh_quotient(A: smp.Matrix, v: smp.Matrix) -> smp.Float:
    """
    Calculate the rayleigh quotient, only suitable for numeric matrix.
    :param A: sympy.Matrix
    :param v: sympy.Matrix, column vector like
    :return: sympy.Float
    """
    if A.is_symbolic() or v.is_symbolic():
        raise MatrixError("Not support for symbolic matrix!")
    if not A.is_square:
        raise NonSquareMatrixError("Parameter A is not a square matrix!")
    if not is_vector(v):
        raise ShapeError("Parameter v is not a row vector or column vector!")
    if is_row_vector(v):
        v = smp.Matrix(v.flat())  # convert v to column vector
    return smp.Float((v.T * A * v)[0, 0] / (v.dot(v)))


def calculate_contravariant_basis(*args: smp.Matrix) -> Tuple[smp.Matrix, ...]:
    """
    Calculate the contravariant basis for specific vector groups.
    :param args: sympy.Matrix, vector like
    :return: a tuple of sympy.Matrix that is vector like
    """
    for vector in args:
        if not is_vector(vector):
            raise ShapeError('Only support for vector like matrix!')

    A = smp.Matrix.vstack(*(to_row_vector(vector) for vector in args))
    return get_column_vectors(A.inv())


def calculate_transform_matrix(old_vectors: Tuple[smp.Matrix, ...], new_vectors: Tuple[smp.Matrix, ...]) -> smp.Matrix:
    """
    Calculate the transform matrix based on old basis vectors and new basis vectors.
    :param old_vectors: a tuple of vector like sympy.Matrix
    :param new_vectors: a tuple of vector like sympy.Matrix
    :return: the transform matrix, sympy.Matrix
    """
    n_1 = len(old_vectors)
    n_2 = len(new_vectors)
    if n_1 != n_2:
        raise ValueError('The count of old vectors is not equal with new vectors!')

    result = []
    for new_vector in new_vectors:
        for old_vector in old_vectors:
            result.append(new_vector.dot(old_vector))
    return smp.Matrix(n_1, n_2, result)


def spectral_decomposition(A: smp.Matrix, n: Union[bool, int] = False) -> Union[smp.Matrix, Sequence[Tuple[Any, Any]]]:
    """
    Do spectral decomposition on target matrix A, if n is instance of int, A**n will be calculated,
    otherwise return a list that contains eigenvalue and correspond eigenvector.
    :param A: sympy.Matrix
    :param n: int or False
    :return: sympy.Matrix or list
    """
    if not A.is_symmetric():
        raise MatrixError('The matrix is not symmetric!')
    P, D = A.diagonalize(normalize=True)
    result = []
    for i in range(A.shape[0]):
        result.append((D[i, i], P.col(i)))
    if not n:
        return result
    elif isinstance(n, int):
        return smp.simplify(smp.Add(*(group[0] ** n * group[1] * group[1].T for group in result)))
    else:
        raise ValueError('Please enter int value or False!')


def to_column_vector(A: smp.Matrix) -> smp.Matrix:
    """
    Convert a vector like Matrix to column vector.
    :param A: sympy.Matrix, vector like
    :return: sympy.Matrix
    """
    if is_column_vector(A):
        return A
    elif is_row_vector(A):
        return smp.Matrix(A.flat())
    else:
        raise ShapeError('The matrix is not vector like!')


def to_row_vector(A: smp.Matrix) -> smp.Matrix:
    """
    Convert a vector like Matrix to row vector.
    :param A: sympy.Matrix, vector like
    :return: sympy.Matrix
    """
    if is_row_vector(A):
        return A
    elif is_column_vector(A):
        return smp.Matrix(A.flat()).T
    else:
        raise ShapeError('The matrix is not vector like!')
