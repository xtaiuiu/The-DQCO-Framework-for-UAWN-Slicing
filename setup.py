from setuptools import setup, find_packages

setup(
    name="your_project",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.26.4,<2.0",
        "matplotlib>=3.9.1",
        "cvxpy>=1.5.2",
        "mealpy>=3.0.1",
        "scipy>=1.12.0",
        "pandas>=2.2.2",
        "seaborn>=0.13.2",
    ],
)
