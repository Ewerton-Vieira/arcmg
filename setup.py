from setuptools import setup

setup (
    name = "arcmg",
    version = "0.0.1",
    author = "Brittany Gelb, Ewerton Rocha Vieira, Miroslav Kramar",
    url = "https://github.com/Ewerton-Vieira/dytop.git",
    description = "arcmg: Attracting Regions Classifier by Morse Graph",
    long_description = open('README.md').read(),
    ext_package='arcmg',
    packages=['arcmg'],
    install_requires = ['numpy', 'torch', 'igraph', 'plotly', 'pandas', 'seaborn', 'matplotlib', 'pyyaml']
)