from setuptools import setup

with open("README", 'r') as f:
    long_description = f.read()

setup(
    name="deltachess",
    version="2.0",
    description="ML based learning chess engine",
    author="Nicolas Weiss",
    author_email="nicolas.alexander.weiss@gmail.com",
    packages=["deltachess"],
    install_requires=["numpy", "scikit-learn", "python-chess"]
)