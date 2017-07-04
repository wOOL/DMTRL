import numpy as np
import tensorflow as tf
from scipy.linalg.interpolative import svd


def my_svd(A, eps_or_k=0.01):
    if A.dtype != np.float64:
        A = A.astype(np.float64)
    U, S, V = svd(A, eps_or_k, rand=False)

    return U, S, V.T


def t_unfold(A, k):
    A = np.transpose(A, np.hstack([k, np.delete(np.arange(A.ndim), k)]))
    A = np.reshape(A, [A.shape[0], np.prod(A.shape[1:])])

    return A


def t_dot(A, B, axes=(-1, 0)):
    return np.tensordot(A, B, axes)


def tt_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    n = A.shape

    max_rank = [min(np.prod(n[:i + 1]), np.prod(n[i + 1:])) for i in range(d - 1)]

    if np.any(np.array(eps_or_k) > np.array(max_rank)):
        raise ValueError('the rank is up to %s' % str(max_rank))

    if not isinstance(eps_or_k, list):
        eps_or_k = [eps_or_k] * (d - 1)

    r = [1] * (d + 1)

    TT = []
    C = A.copy()

    for k in range(d - 1):
        C = C.reshape((r[k] * n[k], C.size / (r[k] * n[k])))
        (U, S, V) = my_svd(C, eps_or_k[k])
        r[k + 1] = U.shape[1]
        TT.append(U[:, :r[k + 1]].reshape((r[k], n[k], r[k + 1])))
        C = np.dot(np.diag(S[:r[k + 1]]), V[:r[k + 1], :])
    TT.append(C.reshape(r[k + 1], n[k + 1], 1))

    return TT


def tucker_dcmp(A, eps_or_k=0.01):
    d = A.ndim
    n = A.shape

    max_rank = list(n)

    if np.any(np.array(eps_or_k) > np.array(max_rank)):
        raise ValueError('the rank is up to %s' % str(max_rank))

    if not isinstance(eps_or_k, list):
        eps_or_k = [eps_or_k] * d

    U = [my_svd(t_unfold(A, k), eps_or_k[k])[0] for k in range(d)]
    S = A
    for i in range(d):
        S = t_dot(S, U[i], (0, 0))

    return U, S


def tt_cnst(A):
    S = A[0]
    for i in range(len(A) - 1):
        S = t_dot(S, A[i + 1])

    return np.squeeze(S, axis=(0, -1))


def tucker_cnst(U, S):
    for i in range(len(U)):
        S = t_dot(S, U[i], (0, 1))

    return S


def TensorUnfold(A, k):
    tmp_arr = np.arange(A.get_shape().ndims)
    A = tf.transpose(A, [tmp_arr[k]] + np.delete(tmp_arr, k).tolist())
    shapeA = A.get_shape().as_list()
    A = tf.reshape(A, [shapeA[0], np.prod(shapeA[1:])])

    return A


def TensorProduct(A, B, axes=(-1, 0)):
    shapeA = A.get_shape().as_list()
    shapeB = B.get_shape().as_list()
    shapeR = np.delete(shapeA, axes[0]).tolist() + np.delete(shapeB, axes[1]).tolist()
    result = tf.matmul(tf.transpose(TensorUnfold(A, axes[0])), TensorUnfold(B, axes[1]))

    return tf.reshape(result, shapeR)


def TTTensorProducer(A):
    S = A[0]
    for i in range(len(A) - 1):
        S = TensorProduct(S, A[i + 1])

    return tf.squeeze(S, squeeze_dims=[0, -1])


def TuckerTensorProducer(U, S):
    for i in range(len(U)):
        S = TensorProduct(S, U[i], (0, 1))

    return S


def TensorProducer(X, method, eps_or_k=0.01, datatype=np.float32, return_true_var=False):
    if method == 'Tucker':
        U, S = tucker_dcmp(X, eps_or_k)
        U = [tf.Variable(i.astype(datatype)) for i in U]
        S = tf.Variable(S.astype(datatype))
        W = TuckerTensorProducer(U, S)
        param_dict = {'U': U, 'S': S}
    elif method == 'TT':
        A = tt_dcmp(X, eps_or_k)
        A = [tf.Variable(i.astype(datatype)) for i in A]
        W = TTTensorProducer(A)
        param_dict = {'U': A}
    elif method == 'LAF':
        U, S, V = my_svd(np.transpose(t_unfold(X, -1)), eps_or_k)
        U = tf.Variable(U.astype(datatype))
        V = tf.Variable(np.dot(np.diag(S), V).astype(datatype))
        W = tf.reshape(tf.matmul(U, V), X.shape)
        param_dict = {'U': U, 'V': V}
    if return_true_var:
        return W, param_dict
    else:
        return W
