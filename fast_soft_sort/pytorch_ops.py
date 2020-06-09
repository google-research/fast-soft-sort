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

"""PyTorch operators for soft sorting and ranking.

Fast Differentiable Sorting and Ranking
Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
https://arxiv.org/abs/2002.08871
"""

from . import numpy_ops
import torch


def wrap_class(cls, **kwargs):
  """Wraps the given NumpyOp in a torch Function."""

  class NumpyOpWrapper(torch.autograd.Function):
    """A torch Function wrapping a NumpyOp."""

    @staticmethod
    def forward(ctx, values):
      obj = cls(values.detach().numpy(), **kwargs)
      ctx.numpy_obj = obj
      return torch.from_numpy(obj.compute())

    @staticmethod
    def backward(ctx, grad_output):
      return torch.from_numpy(ctx.numpy_obj.vjp(grad_output.numpy()))

  return NumpyOpWrapper


def map_tensor(map_fn, tensor):
  return torch.stack([map_fn(tensor_i) for tensor_i in torch.unbind(tensor)])


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

  wrapped_fn = wrap_class(numpy_ops.SoftRank,
                          regularization_strength=regularization_strength,
                          direction=direction,
                          regularization=regularization)
  return map_tensor(wrapped_fn.apply, values)


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

  wrapped_fn = wrap_class(numpy_ops.SoftSort,
                          regularization_strength=regularization_strength,
                          direction=direction,
                          regularization=regularization)

  return map_tensor(wrapped_fn.apply, values)
