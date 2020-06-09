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

"""Tests for pytorch_ops.py."""

import functools
import itertools
import unittest

from absl.testing import parameterized
from fast_soft_sort import pytorch_ops
import torch


GAMMAS = (0.1, 1, 10.0)
DIRECTIONS = ("ASCENDING", "DESCENDING")
REGULARIZERS = ("l2",)  # The kl case is unstable for gradcheck.
DTYPES = (torch.float64,)


class PyTorchOpsTest(parameterized.TestCase):

  def _test(self, func, regularization_strength, direction, regularization,
            dtype, atol=1e-3, rtol=1e-3, eps=1e-5):
    x = torch.randn(5, 10, dtype=dtype, requires_grad=True)

    func = functools.partial(
        func, regularization_strength=regularization_strength,
        direction=direction, regularization=regularization)

    torch.autograd.gradcheck(func, [x], eps=eps, atol=atol, rtol=rtol)

    def _compute_loss(x):
      y = func(x, regularization_strength=regularization_strength,
               direction=direction, regularization=regularization)
      return torch.sum(y**2)

    torch.autograd.gradcheck(_compute_loss, x, eps=eps, atol=atol, rtol=rtol)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_rank_gradient(self, regularization_strength, direction,
                         regularization, dtype):
    self._test(pytorch_ops.soft_rank, regularization_strength, direction,
               regularization, dtype)

  @parameterized.parameters(itertools.product(GAMMAS, DIRECTIONS, REGULARIZERS,
                                              DTYPES))
  def test_sort_gradient(self, regularization_strength, direction,
                         regularization, dtype):
    self._test(pytorch_ops.soft_sort, regularization_strength, direction,
               regularization, dtype)


if __name__ == "__main__":
  unittest.main()
