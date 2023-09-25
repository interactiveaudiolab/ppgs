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
        'accelerate',
        'apprise',
        'encodec',
        'espnet',
        'ffmpeg<5',
        'g2pM',
        'gdown>=4.6.2',
        'humanfriendly',
        'librosa',
        'moviepy',
        'nltk',
        'numpy',
        'pyfoal',
        'pypar',
        'pyyaml',
        'tensorboard',
        'transformers',
        'torch',
        'torchaudio',
        'torch_complex',
        'tqdm',
        'opencv-python',
        'yapecs>=0.0.7',
    ],
    packages=['ppgs'],
    package_data={'ppgs': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
