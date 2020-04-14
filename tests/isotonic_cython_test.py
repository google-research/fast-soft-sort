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

"""Tests for isotonic.py."""

import unittest
from absl.testing import parameterized
from fast_soft_sort import isotonic
from fast_soft_sort import isotonic_cython
import numpy as np


class IsotonicTest(parameterized.TestCase):

  def test_l2_case(self):
    rng = np.random.RandomState(0)
    y = rng.randn(10) * rng.randint(1, 5)
    sol1 = np.zeros_like(y)
    sol2 = np.zeros_like(y)
    isotonic.isotonic_l2(y, sol1)
    isotonic_cython.isotonic_l2(y, sol2)
    np.testing.assert_array_almost_equal(sol1, sol2)

  def test_kl_case(self):
    rng = np.random.RandomState(0)
    y = rng.randn(10) * rng.randint(1, 5)
    w = np.arange(10).astype(np.float64)
    sol1 = np.zeros_like(y)
    sol2 = np.zeros_like(y)
    isotonic.isotonic_kl(y, w, sol1)
    isotonic_cython.isotonic_kl(y, w, sol2)
    np.testing.assert_array_almost_equal(sol1, sol2)


if __name__ == "__main__":
  unittest.main()
