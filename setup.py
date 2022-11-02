from setuptools import setup

install_requires = [
    'numpy',
    'scipy',
    'scikit-learn>=0.22',
    'pandas',
    'matplotlib',
    'torch',
    'grad_cam',
    'torchcam'
]

setup(
    name='pyDeepInsight',
    version='0.1.1',
    packages=['pyDeepInsight'],
    url='https://github.com/alok-ai-lab/pyDeepInsight',
    license='GPLv3',
    author='Keith A. Boroevich',
    author_email='kaboroevich@gmail.com',
    description='A methodology to transform a non-image data to an image for'
                ' convolution neural network architecture',
    install_requires=install_requires,
)
