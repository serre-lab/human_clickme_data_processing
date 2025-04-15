from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

# Define the extension
extensions = [
    Extension(
        "src.cython_utils",
        ["src/cython_utils.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"]  # Optimize for maximum performance
    )
]

# Setup configuration
setup(
    name="clickme_processing",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
    include_dirs=[np.get_include()]
) 