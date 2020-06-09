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

"""Tests for tf_ops.py."""

import functools
import itertools
from absl.testing import parameterized
from fast_soft_sort import tf_ops
import tensorflow.compat.v2 as tf


GAMMAS = (0.1, 1.0, 10.0)
DIRECTIONS = ("ASCENDING", "DESCENDING")
REGULARIZERS = ("l2", "kl")
DTYPES = (tf.float64,)


class TfOpsTest(parameterized.TestCase, tf.test.TestCase):

  def _test(self, func, regularization_strength, direction, regularization,
            dtype):

    precision = 1e-6
    delta = 1e-4

    x = tf.random.normal((5, 10), dtype=dtype)

    func = functools.partial(
        func, regularization_strength=regularization_strength,
        direction=direction, regularization=regularization)

    grad_theoretical, grad_numerical = tf.test.compute_gradient(
        func, [x], delta=delta)

    self.assertAllClose(grad_theoretical[0], grad_numerical[0], precision)

    def _compute_loss(x):
      y = func(x, regularization_strength=regularization_strength,
               direction=direction, regularization=regularization)
      return tf.reduce_mean(y**2)

    grad_theoretical, grad_numerical = tf.test.compute_gradient(
        _compute_loss, [x], delta=delta)

    self.assertAllClose(grad_theoretical[0], grad_numerical[0], precision)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_rank_gradient(self, regularization_strength, direction,
                         regularization, dtype):
    self._test(tf_ops.soft_rank, regularization_strength, direction,
               regularization, dtype)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_sort_gradient(self, regularization_strength, direction,
                         regularization, dtype):
    if regularization == "l2" or regularization_strength < 10:
      # We skip regularization_strength >= 10 when regularization = "kl",
      # due to numerical instability.
      self._test(tf_ops.soft_sort, regularization_strength, direction,
                 regularization, dtype)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
