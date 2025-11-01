"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    """Op to element-wise multiply two nodes."""
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    """Op to element-wise multiply a tensor by a scalar."""
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        base, exponent = node.inputs
        return out_grad * exponent * (base ** (exponent - 1)), out_grad * (base ** exponent) * array_api.log(base)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        base, = node.inputs
        return out_grad * (base ** (self.scalar - 1)) * self.scalar
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, -out_grad * lhs / (rhs * rhs)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        dim = len(a.shape)
        if self.axes is None:
            self.axes = [dim - 2, dim - 1]
        axes = list(range(dim))
        
        axes[self.axes[0]], axes[self.axes[1]] = self.axes[1], self.axes[0]
        return array_api.transpose(a, axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node: Tensor):
        ### BEGIN YOUR SOLUTION
        return out_grad.transpose(self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad.reshape(node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    """Op to broadcast a tensor to a higher dimension."""
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        #########################!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        axes = tuple([len(self.shape) - ind - 1 for ind, dim in enumerate(reversed(self.shape)) if
                      len(input_shape) - ind - 1 < 0 or dim != input_shape[-ind - 1]])
        return out_grad.sum(axes).reshape(input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    """Op to sum along specific axes."""
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return out_grad.broadcast_to(input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    """Op to perform matrix multiplication."""
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return matmul(out_grad, b.transpose()), matmul(a.transpose(), out_grad)
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    """Op to element-wise negate a tensor."""
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    """Op to element-wise compute log."""
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    """Op to element-wise compute exp."""
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * exp(node.inputs[0])
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    """Op to element-wise compute relu."""
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (node.inputs[0] > 0)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

