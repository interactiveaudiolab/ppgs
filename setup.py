from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


# TODO - replace with details of your project
setup(
    name='ppgs',
    description='DESCRIPTION',
    version='0.0.1',
    author='AUTHOR',
    author_email='EMAIL',
    url='https://github.com/USERppgs/ppgs',
    install_requires=['torch', 'yapecs'],
    packages=['ppgs'],
    package_data={'ppgs': ['assets/*']},
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
