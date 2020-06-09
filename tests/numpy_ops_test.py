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

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS))
  def test_soft_rank_works_with_lists(self, regularization_strength, direction,
                         regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    ranks1 = numpy_ops.SoftRank(theta,
                                regularization_strength=regularization_strength,
                                direction=direction,
                                regularization=regularization).compute()
    ranks2 = numpy_ops.SoftRank(list(theta),
                                regularization_strength=regularization_strength,
                                direction=direction,
                                regularization=regularization).compute()
    np.testing.assert_array_almost_equal(ranks1, ranks2)


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

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS))
  def test_soft_sort_works_with_lists(self, regularization_strength, direction,
                         regularization):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    sort1 = numpy_ops.SoftSort(theta,
                               regularization_strength=regularization_strength,
                               direction=direction,
                               regularization=regularization).compute()
    sort2 = numpy_ops.SoftSort(list(theta),
                               regularization_strength=regularization_strength,
                               direction=direction,
                               regularization=regularization).compute()
    np.testing.assert_array_almost_equal(sort1, sort2)


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

  @parameterized.parameters(itertools.product(DIRECTIONS))
  def test_sort_works_with_lists(self, direction):
    rng = np.random.RandomState(0)
    theta = rng.randn(5)
    sort_numpy = numpy_ops.Sort(theta, direction=direction).compute()
    sort_list = numpy_ops.Sort(list(theta), direction=direction).compute()
    np.testing.assert_array_almost_equal(sort_numpy, sort_list)


if __name__ == "__main__":
  unittest.main()
