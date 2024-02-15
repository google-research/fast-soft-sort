
Fast Differentiable Sorting and Ranking
=======================================

Differentiable sorting and ranking operations in O(n log n).

Dependencies
------------

* NumPy
* SciPy
* Numba
* Tensorflow (optional)
* PyTorch (optional)

Purpose of Project
------------
The sorting operation is widely used in computer programming and machine learning for robust statistics. However, as a function, it has many non-differentiable points, making it challenging for certain mathematical operations. Similarly, the ranking operator, often used for order statistics and ranking metrics, has its own set of challenges due to its piecewise constant nature, which makes its derivatives null or undefined. Although there have been attempts to create differentiable alternatives to sorting and ranking, they have not been able to achieve the expected O(nlogn) time complexity.

In this repository, we propose the first differentiable sorting and ranking operators with O(nlogn) time and O(n) space complexity. Our proposal not only provides precise computation, but also allows for differentiation, overcoming the limitations of current methods. We achieve this feat by constructing differentiable operators as projections onto the permutahedron, the convex hull of permutations, and using a reduction to isotonic optimization.

TensorFlow Example
-------------------

```python
>>> import tensorflow as tf
>>> from fast_soft_sort.tf_ops import soft_rank, soft_sort
>>> values = tf.convert_to_tensor([[5., 1., 2.], [2., 1., 5.]], dtype=tf.float64)
>>> soft_sort(values, regularization_strength=1.0)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[1.66666667, 2.66666667, 3.66666667], [1.66666667, 2.66666667, 3.66666667]])>
>>> soft_sort(values, regularization_strength=0.1)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[1., 2., 5.], [1., 2., 5.]])>
>>> soft_rank(values, regularization_strength=2.0)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[3. , 1.25, 1.75], [1.75, 1.25, 3. ]])>
>>> soft_rank(values, regularization_strength=1.0)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[3., 1., 2.], [2., 1., 3.]])>
```

JAX Example
-----------

```python
>>> import jax.numpy as jnp
>>> from fast_soft_sort.jax_ops import soft_rank, soft_sort
>>> values = jnp.array([[5., 1., 2.], [2., 1., 5.]], dtype=jnp.float64)
>>> soft_sort(values, regularization_strength=1.0)
[[1.66666667 2.66666667 3.66666667]
 [1.66666667 2.66666667 3.66666667]]
>>> soft_sort(values, regularization_strength=0.1)
[[1. 2. 5.]
 [1. 2. 5.]]
>>> soft_rank(values, regularization_strength=2.0)
[[3.   1.25 1.75]
 [1.75 1.25 3.  ]]
>>> soft_rank(values, regularization_strength=1.0)
[[3. 1. 2.]
 [2. 1. 3.]]
```

PyTorch Example
---------------

```python
>>> import torch
>>> from pytorch_ops import soft_rank, soft_sort
>>> values = fast_soft_sort.torch.tensor([[5., 1., 2.], [2., 1., 5.]], dtype=torch.float64)
>>> soft_sort(values, regularization_strength=1.0)
tensor([[1.6667, 2.6667, 3.6667]
        [1.6667, 2.6667, 3.6667]], dtype=torch.float64)
>>> soft_sort(values, regularization_strength=0.1)
tensor([[1., 2., 5.]
        [1., 2., 5.]], dtype=torch.float64)
>>> soft_rank(values, regularization_strength=2.0)
tensor([[3.0000, 1.2500, 1.7500],
        [1.7500, 1.2500, 3.0000]], dtype=torch.float64)
>>> soft_rank(values, regularization_strength=1.0)
tensor([[3., 1., 2.]
        [2., 1., 3.]], dtype=torch.float64)
```


Install
--------

Run `python setup.py install` or copy the `fast_soft_sort/` folder to your
project.


FAQ
------
<h4>Q: What is differentiable sorting and ranking?</h4>
A: Differentiable sorting and ranking refer to operations that maintain differentiability properties while sorting or ranking elements of a sequence. In machine learning, these operations are crucial for tasks such as permutation-invariant neural networks and differentiable optimization.

<h4>Q: Why is differentiable sorting important?</h4>
A: Differentiable sorting enables the use of sorting operations within deep learning models, allowing gradients to flow through the sorting process. This capability is essential for tasks where sorting is a part of the computational graph, such as learning-to-rank algorithms and permutation-invariant architectures.

<h4>Q: What are the advantages of using the provided operations?</h4>
A: The provided operations offer O(n log n) time complexity, making them efficient for large datasets. Additionally, they support various deep learning frameworks, including TensorFlow, PyTorch, and JAX, providing flexibility for integration into existing workflows.

<h4>Q: How can I incorporate differentiable sorting and ranking into my machine learning models?</h4>
A: You can use the provided operations as building blocks within your neural network architectures. For example, you can use differentiable sorting to implement permutation-invariant layers or ranking operations for tasks such as learning-to-rank or top-k selection.

<h4>Q: Are there any limitations to differentiable sorting and ranking?</h4>
A: While differentiable sorting and ranking provide valuable capabilities, it's essential to consider computational costs, especially for large datasets. Additionally, the choice of regularization parameters can affect the behavior of the operations, requiring careful tuning for optimal performance.

Reference
------------

> Fast Differentiable Sorting and Ranking
> Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
> In proceedings of ICML 2020
> [arXiv:2002.08871](https://arxiv.org/abs/2002.08871)
