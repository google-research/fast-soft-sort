
Fast Differentiable Sorting and Ranking
=======================================

Differentiable sorting and ranking operations in O(n log n).

Dependencies
------------

* NumPy
* SciPy
* Numba
* Tensorflow

Example
--------

```python
>>> import tensorflow as tf
>>> from tf_ops import soft_rank, soft_sort
>>> values = tf.convert_to_tensor([[5., 1., 2.], [2., 1., 5.]], dtype=tf.float64)
>>> tf_ops.soft_sort(values, regularization_strength=1.0)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[1.66666667, 2.66666667, 3.66666667], [1.66666667, 2.66666667, 3.66666667]])>
>>> tf_ops.soft_sort(values, regularization_strength=0.1)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[1., 2., 5.], [1., 2., 5.]])>
>>> tf_ops.soft_rank(values, regularization_strength=2.0)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[3. , 1.25, 1.75], [1.75, 1.25, 3. ]])>
>>> tf_ops.soft_rank(values, regularization_strength=1.0)
<tf.Tensor: shape=(2, 3), dtype=float64, numpy= array([[3., 1., 2.], [2., 1., 3.]])>
```

Install
--------

Run `python setup.py install` or copy the `fast_soft_sort/` folder to your
project.


Reference
------------

> Fast Differentiable Sorting and Ranking
> Mathieu Blondel, Olivier Teboul, Quentin Berthet, Josip Djolonga
> [arXiv:2002.08871](https://arxiv.org/abs/2002.08871)
