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

"""Tests for isotonic.py."""

import unittest
from absl.testing import parameterized
from fast_soft_sort.third_party import isotonic
import numpy as np
from sklearn.isotonic import isotonic_regression


class IsotonicTest(parameterized.TestCase):

  def test_l2_agrees_with_sklearn(self):
    rng = np.random.RandomState(0)
    y = rng.randn(10) * rng.randint(1, 5)
    sol = np.zeros_like(y)
    isotonic.isotonic_l2(y, sol)
    sol_skl = isotonic_regression(y, increasing=False)
    np.testing.assert_array_almost_equal(sol, sol_skl)


if __name__ == "__main__":
  unittest.main()
