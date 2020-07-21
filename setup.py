from setuptools import setup,find_packages

with open("README.md", "r") as file:
    read_me_description = file.read()

extras = {
   'DIRECT': ['DIRECT'],
    'smt' : ['smt'],
    'all' : ['DIRECT','smt']
}

setup(
    name='GPGO',
    version='0.1.2',
    author='Cristian Gabellini',
    packages=find_packages(),
    url='https://github.com/FNTwin/GPGO',
    download_url = 'https://github.com/FNTwin/GPGO/archive/0.1.2.tar.gz' ,
    license='MIT',
    description='Bayesian Optimization with Gaussian Process as surrogate model',
    long_description=read_me_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.6',
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'scikit-learn'
    ],
    extras_require=extras,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ]
)
