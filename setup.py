from setuptools import setup, find_packages

setup(
    name='FIND_PRODUCT',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            # Add CLI entry points here if needed
        ],
    },
)
