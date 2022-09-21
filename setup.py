import os
import sys
from os.path import dirname, join, realpath
from shutil import rmtree

import toml
from setuptools import Command, find_packages, setup

# $ python setup.py upload

__title__ = "pkgu"
__description__ = (
    "Find the out-dated packages installed by the Pip tool and update them."
)
__url__ = "https://github.com/Abeautifulsnow/pipu"
__author_email__ = "lcprunstone@163.com"
__license__ = "MIT"

__keywords__ = ["python", "pip", "pip install pkg --upgrade", "pip-update"]
__modules__ = ["pipu", "-version"]

# Load the package's _version.py module as a dictionary.
PROJECT_ROOT = dirname(realpath(__file__))

with open(join(PROJECT_ROOT, "README.md"), "r") as md_read:
    __long_description__ = md_read.read()

# generate the dependency data
with open(join(PROJECT_ROOT, "pyproject.toml"), "r") as f_read:
    content = toml.load(f_read)
    dependencies = content["tool"]["poetry"]["dependencies"]
    dev_dependencies = content["tool"]["poetry"]["dev-dependencies"]
    __authors__ = content["tool"]["poetry"]["authors"][0]
    __version__ = content["tool"]["poetry"]["version"]

dependencies.update(**dev_dependencies)
__install_reqs__ = []
for dep, version in dependencies.items():
    if dep == 'python':
        continue
    else:
        if version.startswith("^"):
            version = version.split("^")[1]

        __install_reqs__.append(f"{dep}>={version}")


class UploadCommand(Command):
    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        print("✨✨ {0}".format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(PROJECT_ROOT, "dist"))
            rmtree(os.path.join(PROJECT_ROOT, "build"))
            rmtree(os.path.join(PROJECT_ROOT, "{0}.egg-info".format(__title__)))
        except OSError:
            pass

        self.status("Building Source and Wheel distribution…")
        os.system("{0} setup.py bdist_wheel".format(sys.executable))

        self.status("Uploading the package to PyPI via Twine…")
        return_code = os.system("twine upload dist/* --verbose")

        if not return_code:
            self.status("Pushing git tags…")
            os.system('git tag -a v{0} -m "release version v{0}"'.format(__version__))
            os.system("git push origin v{0}".format(__version__))

        sys.exit()


setup(
    name=__title__,
    version=__version__,
    description=__description__,
    url=__url__,
    author=__authors__,
    author_email=__author_email__,
    license=__license__,
    packages=find_packages(exclude=("test",)),
    keywords=__keywords__,
    py_modules=__modules__,
    zip_safe=False,
    long_description=__long_description__,
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=__install_reqs__,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Software Development :: Libraries",
    ],
    cmdclass={"upload": UploadCommand},
    entry_points={"console_scripts": ["pipu=pipu:entry"]},
)
