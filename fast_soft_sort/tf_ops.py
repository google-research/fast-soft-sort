# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
