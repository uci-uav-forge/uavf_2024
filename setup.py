from setuptools import setup, find_packages

setup(
    name='uavf_2024',
    version='0.0.1',
    packages=find_packages(include=[
        'uavf_2024', 'uavf_2024.*'
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'casadi',
        'ultalytics'
    ]
)