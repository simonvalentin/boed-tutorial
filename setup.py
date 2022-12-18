from setuptools import setup
from os import path


current_directory = path.abspath(path.dirname(__file__))
with open(path.join(current_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="modelcomp",
    author="Simon Valentin, Steven Kleinegesse",
    description=("Designing Optimal Behavioral Experiments using Machine Learning"),
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/simonvalentin/boed_elife",
    packages=['boed-elife']
)
