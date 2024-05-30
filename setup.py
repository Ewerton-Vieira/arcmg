from setuptools import setup, find_packages

setup (
    name = "arcmg",
    version = "0.0.1",
    author = "Brittany Gelb, Ewerton Rocha Vieira",
    url = "https://github.com/Ewerton-Vieira/arcmg.git",
    description = "arcmg: Attracting Regions Classifier by Morse Graph",
    long_description = open('README.md').read(),
    long_description_content_type='text/markdown',
    ext_package='arcmg',
    packages=find_packages(),
    install_requires = ['numpy', 'torch', 'igraph', 'plotly', 'pandas', 'seaborn', 'matplotlib', 'pyyaml']
)