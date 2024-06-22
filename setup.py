from setuptools import setup, find_packages


def read_requirements(file):
    try:
        with open(file, encoding='utf-8') as f:
            return f.read().splitlines()
    except UnicodeError:
        with open(file, encoding='utf-16') as f:
            return f.read().splitlines()
        

requirements = read_requirements('requirements.txt')

setup(
    name='ijig_jepa',
    version='0.1',
    packages=find_packages(),
    description='IJig-Jepa: a novel way to train I-Jepa architectures',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Tommaso Capecchi',
    author_email='tommycaps@hotmail.it',
    url='https://github.com/noisecape/ijig-jepa.git',
    install_requires=requirements,
    python_requires='>=3.12.4'
)