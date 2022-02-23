import numpy as np

"""
Check if the given matrix is unitary
"""
def is_unitary(m):
    return np.allclose(np.eye(len(m)), m.dot(m.T.conj()))

"""
Calculate difference between two matrices
"""
def error(A, B):
    return np.linalg.norm(A - B)