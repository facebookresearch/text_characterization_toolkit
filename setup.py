# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup
from setuptools.command.install import install
from setuptools.command.develop import develop
from subprocess import check_call


with open("README.md") as f:
    readme = f.read()

def post_install():
    print("Downloading resources...")
    check_call("sh scripts/download_resources.sh".split())

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        post_install()

class PostDevelopCommand(develop):
    """Post-installation for installation mode."""
    def run(self):
        develop.run(self)
        post_install()
 
 
setup(
    name="text_characterization_toolkit",
    version="0.1",
    description="Text Characterization Toolkit",
    url="https://github.com/fairinternal/data_characterization",
    long_description=readme,
    long_description_content_type="text/markdown",
    install_requires=[
        "pandas",
        "pandarallel",
        "hyphenator",
        "lexical-diversity",
        "nltk",
        "spacy",
        "dill",
        "openpyxl",
        "matplotlib",
        "scipy",
        "seaborn",
        "sklearn",
        "statsmodels",
    ],
    packages=['text_characterization'],
    cmdclass={
        'develop': PostDevelopCommand,
        'install': PostInstallCommand,
    },
)
