from setuptools import setup,find_packages

with open("README.md", "r") as file:
    read_me_description = file.read()

setup(
    name='GOGP',
    version='0.1',
    author='Cristian Gabellini',
    #packages=['GaussianProcessTS', 'GaussianProcessTS.GaussianProcess', 'GaussianProcessTS.GaussianProcess.Kernel'],
    packages=find_packages(),
    url='',
    license='',
    author_email='',
    description='Bayesian Optimization with Gaussian Process as surrogate model',
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
    ]
)
