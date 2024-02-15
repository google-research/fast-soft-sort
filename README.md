
Fast Differentiable Sorting and Ranking Operations in O(nlogn)
=============================================================

* The first differentiable sorting and ranking operators with O(n log n) time and O(n) memory complexity
* Applications to research: machine learning, isotonic regression, robust statistics, optimal transport, etc.


Dependencies
------------

* NumPy
* SciPy
* Numba
* Tensorflow (optional)
* PyTorch (optional)

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
Run `pip install setuptools` (if not yet installed)
Run `python setup.py install` or copy the `fast_soft_sort/` folder to your
project.

Contributers 
------------
Thank you to our researchers:
* mblondel
* josipd
* ita9naiwa 
* francescortu

Frequently Asked Questions
--------------------------
Q: How are differentiable sorting and ranking operations different from regular sorting and ranking operations? 
A: Regular sorting and ranking operations aren't differentiable because the function is not continuous.
Differentiable sorting and ranking operations create "smooth" versions of sorting and ranking that allow us to 
differentiate with respect to the input values.
Analogy: 
Regular sorting and ranking operations are like a staircase, and DSR is like a ramp. Both get you to your destination, but the ramp lets you smoothly roll down.

Q: Was this project developed by Google? 
A: Yes, this research project was carried about by researchers of the Google Brain team directly. 

Q: How do I contribute to this repository? 
A: Guidelines for contributing to this repository can be found in CONTRIBUTING.md

Reference
------------

> Fast Differentiable Sorting and Ranking
> Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
> In proceedings of ICML 2020
> [arXiv:2002.08871](https://arxiv.org/abs/2002.08871)
