from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='ppgs',
    description='Phonetic posteriorgrams',
    version='0.0.6',
    author='Interactive Audio Lab',
    author_email='interactiveaudiolab@gmail.com',
    url='https://github.com/interactiveaudiolab/ppgs',
    extras_require={
        'train': [
            'dac',
            'encodec',
            'g2pM',
            'gdown>=4.6.2',
            'humanfriendly',
            'librosa',
            'nltk',
            'pyyaml',
            'torch-complex',
        ]
    },
    install_requires=[
        'espnet',
        'huggingface-hub',
        'moviepy',
        'numpy',
        'pypar',
        'torch',
        'torchaudio',
        'torchutil',
        'tqdm',
        'transformers',
        'opencv-python',
        'yapecs',
        'gdown>=4.6.2'
    ],
    packages=['ppgs'],
    package_data={'ppgs': ['assets/*', 'assets/*/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=['phonemes', 'ppg', 'pronunciation', 'speech'],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
