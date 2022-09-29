from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='ppgs',
    description='Phonetic posteriorgrams',
    version='0.0.1',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/ppgs',
    install_requires=[
        'espnet',
        'humanfriendly',
        'librosa',
        'numpy',
        'tensorboard',
        'torch',
        'torchaudio',
        'torch_complex',
        'tqdm',
        'pyyaml',
        'yapecs',
        'pypar',
        'nltk'],
    packages=['ppgs'],
    package_data={'ppgs': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
