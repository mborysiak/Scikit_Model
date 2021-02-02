# Import needed function from setuptools
from setuptools import setup

# Create proper setup to be used by pip
setup(name='scikit-model',
      version='0.0.1',
      author='Mark',
      packages=['skmodel'],
      install_requires=['pandas', 'datetime'])