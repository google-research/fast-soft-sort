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

"""Tests for numpy_ops.py."""

import itertools
import unittest
from absl.testing import parameterized
from fast_soft_sort import numpy_ops
import numpy as np


def _num_jacobian(theta, f, eps=1e-9):
  n_classes = len(theta)
  ret = np.zeros((n_classes, n_classes))

  for i in range(n_classes):
    theta_ = theta.copy()
    theta_[i] += eps
    val = f(theta_)
    theta_[i] -= 2 * eps
    val2 = f(theta_)
    ret[i] = (val - val2) / (2 * eps)

  return ret.T


GAMMAS = (0.1, 1.0, 10.0)
DIRECTIONS = ("ASCENDING", "DESCENDING")
REGULARIZERS = ("l2", "kl")
CLASSES = (numpy_ops.Isotonic, numpy_ops.Projection)


class IsotonicProjectionTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(CLASSES, REGULARIZERS))
  def test_jvp_and_vjp_against_numerical_jacobian(self, cls, regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    w = np.arange(5)[::-1]
    v = rng.randn(5)

    f = lambda x: cls(x, w, regularization=regularization).compute()
    J = _num_jacobian(theta, f)

    obj = cls(theta, w, regularization=regularization)
    obj.compute()

    out = obj.jvp(v)
    np.testing.assert_array_almost_equal(J.dot(v), out)

    out = obj.vjp(v)
    np.testing.assert_array_almost_equal(v.dot(J), out)


class SoftRankTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(DIRECTIONS, REGULARIZERS))
  def test_soft_rank_converges_to_hard(self, direction, regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    soft_rank = numpy_ops.SoftRank(theta, regularization_strength=1e-3,
                                   direction=direction,
                                   regularization=regularization)
    out = numpy_ops.rank(theta, direction=direction)
    out2 = soft_rank.compute()
    np.testing.assert_array_almost_equal(out, out2)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS))
  def test_soft_rank_jvp_and_vjp_against_numerical_jacobian(self,
                                                            regularization_strength,
                                                            direction,
                                                            regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    v = rng.randn(5)

    f = lambda x: numpy_ops.SoftRank(
        x, regularization_strength=regularization_strength, direction=direction,
        regularization=regularization).compute()
    J = _num_jacobian(theta, f)

    soft_rank = numpy_ops.SoftRank(
        theta, regularization_strength=regularization_strength,
        direction=direction, regularization=regularization)
    soft_rank.compute()

    out = soft_rank.jvp(v)
    np.testing.assert_array_almost_equal(J.dot(v), out, 1e-6)

    out = soft_rank.vjp(v)
    np.testing.assert_array_almost_equal(v.dot(J), out, 1e-6)

    out = soft_rank.jacobian()
    np.testing.assert_array_almost_equal(J, out, 1e-6)


class SoftSortTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(DIRECTIONS, REGULARIZERS))
  def test_soft_sort_converges_to_hard(self, direction, regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    soft_sort = numpy_ops.SoftSort(
        theta, regularization_strength=1e-3, direction=direction,
        regularization=regularization)
    sort = numpy_ops.Sort(theta, direction=direction)
    out = sort.compute()
    out2 = soft_sort.compute()
    np.testing.assert_array_almost_equal(out, out2)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS))
  def test_soft_sort_jvp(self, regularization_strength, direction,
                         regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    v = rng.randn(5)

    f = lambda x: numpy_ops.SoftSort(
        x, regularization_strength=regularization_strength,
        direction=direction, regularization=regularization).compute()
    J = _num_jacobian(theta, f)

    soft_sort = numpy_ops.SoftSort(
        theta, regularization_strength=regularization_strength,
        direction=direction, regularization=regularization)
    soft_sort.compute()

    out = soft_sort.jvp(v)
    np.testing.assert_array_almost_equal(J.dot(v), out, 1e-6)

    out = soft_sort.vjp(v)
    np.testing.assert_array_almost_equal(v.dot(J), out, 1e-6)

    out = soft_sort.jacobian()
    np.testing.assert_array_almost_equal(J, out, 1e-6)


class SortTest(parameterized.TestCase):

  @parameterized.parameters(itertools.product(DIRECTIONS))
  def test_sort_jvp(self, direction):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    v = rng.randn(5)

    f = lambda x: numpy_ops.Sort(x, direction=direction).compute()
    J = _num_jacobian(theta, f)

    sort = numpy_ops.Sort(theta, direction=direction)
    sort.compute()

    out = sort.jvp(v)
    np.testing.assert_array_almost_equal(J.dot(v), out)

    out = sort.vjp(v)
    np.testing.assert_array_almost_equal(v.dot(J), out)


if __name__ == "__main__":
  unittest.main()
