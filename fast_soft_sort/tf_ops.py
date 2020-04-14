# Copyright 2020 Google LLC.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.

"""Tensorflow operators for soft sorting and ranking.

Fast Differentiable Sorting and Ranking
Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
https://arxiv.org/abs/2002.08871
"""

from . import numpy_ops
import tensorflow.compat.v2 as tf


def _wrap_numpy_op(cls, regularization_strength, direction, regularization):
  """Converts NumPy operator to a TF one."""

  @tf.custom_gradient
  def _func(values):
    """Converts values to numpy array, applies function and returns tensor."""
    dtype = values.dtype

    try:
      values = values.numpy()
    except AttributeError:
      pass

    obj = cls(values, regularization_strength=regularization_strength,
              direction=direction, regularization=regularization)
    result = obj.compute()

    def grad(v):
      v = v.numpy()
      return tf.convert_to_tensor(obj.vjp(v), dtype=dtype)

    return tf.convert_to_tensor(result, dtype=dtype), grad

  return _func


def soft_rank(values, direction="ASCENDING", regularization_strength=1.0,
              regularization="l2"):
  r"""Soft rank the given values (tensor) along the second axis.

  The regularization strength determines how close are the returned values
  to the actual ranks.

  Args:
    values: A 2d-tensor holding the numbers to be ranked.
    direction: Either 'ASCENDING' or 'DESCENDING'.
    regularization_strength: The regularization strength to be used. The smaller
    this number, the closer the values to the true ranks.
    regularization: Which regularization method to use. It
      must be set to one of ("l2", "kl", "log_kl").
  Returns:
    A 2d-tensor, soft-ranked along the second axis.
  """
  if len(values.shape) != 2:
    raise ValueError("'values' should be a 2d-tensor "
                     "but got %r." % values.shape)

  assert tf.executing_eagerly()

  func = _wrap_numpy_op(numpy_ops.SoftRank, regularization_strength, direction,
                        regularization)

  return tf.map_fn(func, values)


def soft_sort(values, direction="ASCENDING",
              regularization_strength=1.0, regularization="l2"):
  r"""Soft sort the given values (tensor) along the second axis.

  The regularization strength determines how close are the returned values
  to the actual sorted values.

  Args:
    values: A 2d-tensor holding the numbers to be sorted.
    direction: Either 'ASCENDING' or 'DESCENDING'.
    regularization_strength: The regularization strength to be used. The smaller
    this number, the closer the values to the true sorted values.
    regularization: Which regularization method to use. It
      must be set to one of ("l2", "log_kl").
  Returns:
    A 2d-tensor, soft-sorted along the second axis.
  """
  if len(values.shape) != 2:
    raise ValueError("'values' should be a 2d-tensor "
                     "but got %s." % str(values.shape))

  assert tf.executing_eagerly()

  func = _wrap_numpy_op(numpy_ops.SoftSort, regularization_strength, direction,
                        regularization)

  return tf.map_fn(func, values)
