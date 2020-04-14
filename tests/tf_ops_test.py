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
    self._test(tf_ops.soft_sort, regularization_strength, direction,
               regularization, dtype)


if __name__ == "__main__":
  tf.enable_v2_behavior()
  tf.test.main()
