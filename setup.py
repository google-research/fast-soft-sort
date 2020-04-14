"""Install fast_soft_sort."""

from setuptools import find_packages
from setuptools import setup

setup(
    name='fast_soft_sort',
    version='0.1',
    description=(
        'Differentiable sorting and ranking in O(n log n).'),
    author='Google LLC',
    author_email='no-reply@google.com',
    url='https://github.com/google-research/fast-soft-sort',
    license='BSD',
    packages=find_packages(),
    package_data={},
    install_requires=[
        'numba',
        'numpy',
        'scipy>=1.2.0',
    ],
    extras_require={
        'tf': ['tensorflow>=1.12'],
    },
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='machine learning sorting ranking',
)
