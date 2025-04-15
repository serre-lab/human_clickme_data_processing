import os
import sys
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

module_name = "src.cython_utils"
source_file = "src/cython_utils.pyx"

# Check if the source file exists
if not os.path.exists(source_file):
    print(f"Error: Source file '{source_file}' not found.")
    print(f"Current directory: {os.getcwd()}")
    print(f"Directory contents: {os.listdir('.')}")
    if os.path.exists('src'):
        print(f"src directory contents: {os.listdir('src')}")
    sys.exit(1)
else:
    print(f"Found source file: {os.path.abspath(source_file)}")

# Define the extension
extension = Extension(
    module_name,
    sources=[source_file],
    include_dirs=[np.get_include()],
    extra_compile_args=["-O3"],
)

# Configure and run setup
setup(
    name="clickme_processing",
    ext_modules=cythonize(
        extension,
        compiler_directives={
            "language_level": 3,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
        },
    ),
    include_dirs=[np.get_include()]
) 