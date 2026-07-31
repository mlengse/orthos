from setuptools import setup

setup(
    name="orthos",
    version="0.1.0",
    py_modules=["orthos_colab"],
    python_requires=">=3.10",
    install_requires=["numpy", "numba"],
)
