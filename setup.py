from setuptools import setup, find_packages

setup(
    name='px4_offboard_mpc',
    version='0.0.1',
    packages=find_packages(include=[
        'px4_offboard_mpc', 'px4_offboard_mpc.*'
    ]),
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'casadi',
        'acados_template'
    ]
)