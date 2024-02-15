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

"""JAX operators for soft sorting and ranking.

Fast Differentiable Sorting and Ranking
Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
https://arxiv.org/abs/2002.08871
"""

from . import numpy_ops
import jax
import numpy as np
import jax.numpy as jnp
from jax import tree_util


def _wrap_numpy_op(cls, **kwargs):
  """Converts NumPy operator to a JAX one."""

  def _func_fwd(values):
    """Converts values to numpy array, applies function and returns array."""
    dtype = values.dtype
    values = np.array(values)
    obj = cls(values, **kwargs)
    result = obj.compute()
    return jnp.array(result, dtype=dtype), tree_util.Partial(obj.vjp)

  def _func_bwd(vjp, g):
    g = np.array(g)
    result = jnp.array(vjp(g), dtype=g.dtype)
    return (result,)

  @jax.custom_vjp
  def _func(values):
    return _func_fwd(values)[0]

  _func.defvjp(_func_fwd, _func_bwd)

  return _func


def soft_rank(values, direction="ASCENDING", regularization_strength=1.0,
              regularization="l2"):
  r"""Soft rank the given values (array) along the second axis.

  The regularization strength determines how close are the returned values
  to the actual ranks.

  Args:
    values: A 2d-array holding the numbers to be ranked.
    direction: Either 'ASCENDING' or 'DESCENDING'.
    regularization_strength: The regularization strength to be used. The smaller
    this number, the closer the values to the true ranks.
    regularization: Which regularization method to use. It
      must be set to one of ("l2", "kl", "log_kl").
  Returns:
    A 2d-array, soft-ranked along the second axis.
  """
  if len(values.shape) != 2:
    raise ValueError("'values' should be a 2d-array "
                     "but got %r." % values.shape)

  func = _wrap_numpy_op(numpy_ops.SoftRank,
                        regularization_strength=regularization_strength,
                        direction=direction,
                        regularization=regularization)

  return jnp.vstack([func(val) for val in values])


def soft_sort(values, direction="ASCENDING",
              regularization_strength=1.0, regularization="l2"):
  r"""Soft sort the given values (array) along the second axis.

  The regularization strength determines how close are the returned values
  to the actual sorted values.

  Args:
    values: A 2d-array holding the numbers to be sorted.
    direction: Either 'ASCENDING' or 'DESCENDING'.
    regularization_strength: The regularization strength to be used. The smaller
    this number, the closer the values to the true sorted values.
    regularization: Which regularization method to use. It
      must be set to one of ("l2", "log_kl").
  Returns:
    A 2d-array, soft-sorted along the second axis.
  """
  if len(values.shape) != 2:
    raise ValueError("'values' should be a 2d-array "
                     "but got %s." % str(values.shape))

  func = _wrap_numpy_op(numpy_ops.SoftSort,
                        regularization_strength=regularization_strength,
                        direction=direction,
                        regularization=regularization)

  return jnp.vstack([func(val) for val in values])
