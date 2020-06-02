from setuptools import setup, find_packages

setup(
    name='Joint-Forecasting-and-Interpolation-of-Graph-Signals-Using-Deep-Learning',
   version='0.1.0',
   author='Gabriela Lewenfus',
   author_email='gabriela.lewenfus@gmail.com',
   packages=find_packages(),
   install_requires = ['scipy>=1.4.1', 'pandas>=0.15', 'scikit-learn>=0.22', 'numpy>=0.46'],
   description='Code from the paper Joint Forecasting and Interpolation of Graph Signals Using Deep Learning',

)