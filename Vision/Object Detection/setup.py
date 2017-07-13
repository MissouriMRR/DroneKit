from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['h5py>=2.7.0', 'Keras>=2.0.6', 'Theano>=0.9.0', 'matplotlib>=2.0.2', 'scikit-learn>=0.18', 'hyperopt>=0.1', 'google-cloud-storage']

setup(
    name='trainer',
    version='3.0',
    install_requires = REQUIRED_PACKAGES,
    packages = find_packages(),
    include_package_data = True,
    description='MRRDT CNN Cascade Trainer Package'
)
