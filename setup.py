from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

# Ensure the file exists and get the absolute path
src_file = os.path.abspath("src/cython_utils.pyx")
if not os.path.exists(src_file):
    raise FileNotFoundError(f"Cython source file not found: {src_file}")
print(f"Found Cython source file: {src_file}")

# Define the extension
extensions = [
    Extension(
        "src.cython_utils",
        sources=[src_file],
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