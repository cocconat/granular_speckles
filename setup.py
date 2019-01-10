import os

from setuptools import setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as buf:
        return buf.read()


conf = dict(
        name='granularpy',
        version='0.1',
        description='Analyze diffusing wave spectroscopy tool',
        long_description=read('README.md'),
        author='Gnoli, Quaresima, Plati, Petri',
        author_email='q.ale@protonmail.ch',
        url='https://github.com/cocconat/granular_speckles',
        license='AGPL',
        packages=['granular_speckles'],
        install_requires=[
            'numpy',
            'matplotlib',
            'scipy',
            'imageio'
        ],
        zip_safe=False,
        entry_points={'console_scripts': [
            'granular_speckles=granular_speckles.main:main',
        ]},
        classifiers=[
          "License :: OSI Approved :: GNU Affero General Public License v3",
          "Operating System :: POSIX :: Linux",
          "Programming Language :: Python :: 2.7",
        ])


if __name__ == '__main__':
    setup(**conf)
