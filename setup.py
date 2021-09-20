from setuptools import setup
setup(
    name='torch_baseline',
    version='0.0.1',
    packages=['torch_baselines'],
    install_requires=[
        'requests',
        'mlagents_envs==0.27.0',
        'gym',
        'torch',
        'numpy',
        'importlib; python_version >= "3.5"',
    ],
)