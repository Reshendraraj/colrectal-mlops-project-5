from setuptools import setup,find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name ="MLOPS-PROJECT-5",
    version=0.5,
    author = "raj",
    packages=find_packages(),
    install_requires = requirements,
)