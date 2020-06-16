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

"""Tests for jax_ops.py."""

import functools
import itertools
import unittest

from absl.testing import absltest
from absl.testing import parameterized

import numpy as np
import jax.numpy as jnp
import jax

from jax.config import config
config.update("jax_enable_x64", True)

from fast_soft_sort import jax_ops

GAMMAS = (0.1, 1, 10.0)
DIRECTIONS = ("ASCENDING", "DESCENDING")
REGULARIZERS = ("l2", )


class JaxOpsTest(parameterized.TestCase):

  def _test(self, func, regularization_strength, direction, regularization):

    def loss_func(values):
      soft_values = func(values,
                         regularization_strength=regularization_strength,
                         direction=direction,
                         regularization=regularization)
      return jnp.sum(soft_values ** 2)

    rng = np.random.RandomState(0)
    values = jnp.array(rng.randn(5, 10))
    mat = jnp.array(rng.randn(5, 10))
    unitmat = mat / np.sqrt(np.vdot(mat, mat))
    eps = 1e-5
    numerical = (loss_func(values + 0.5 * eps * unitmat) -
                 loss_func(values - 0.5 * eps * unitmat)) / eps
    autodiff = jnp.vdot(jax.grad(loss_func)(values), unitmat)
    np.testing.assert_almost_equal(numerical, autodiff)


  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS))
  def test_soft_rank(self, regularization_strength, direction, regularization):
    self._test(jax_ops.soft_rank,
               regularization_strength, direction, regularization)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS))
  def test_soft_sort(self, regularization_strength, direction, regularization):
    self._test(jax_ops.soft_sort,
               regularization_strength, direction, regularization)


if __name__ == "__main__":
  absltest.main()
