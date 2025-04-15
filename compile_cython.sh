#!/bin/bash

echo "=== Checking environment ==="
echo "Python version: $(python --version)"
echo "Cython version: $(python -c 'import Cython; print(Cython.__version__)')"
echo "NumPy version: $(python -c 'import numpy; print(numpy.__version__)')"

echo "=== Checking src directory ==="
if [ -d "src" ]; then
    echo "src directory exists"
    ls -la src/
else
    echo "src directory does not exist!"
    exit 1
fi

echo "=== Checking Cython source file ==="
if [ -f "src/cython_utils.pyx" ]; then
    echo "src/cython_utils.pyx exists"
else
    echo "src/cython_utils.pyx does not exist!"
    exit 1
fi

echo "=== Cleaning previous builds ==="
rm -rf build/
rm -f src/cython_utils.c
rm -f src/cython_utils*.so

echo "=== Compiling Cython module ==="
python compile_cython.py build_ext --inplace

if [ $? -eq 0 ]; then
    echo "=== Success! ==="
    echo "Testing import:"
    python -c "from src import cython_utils; print('Successfully imported cython_utils')"
else
    echo "=== Compilation failed! ==="
    exit 1
fi 