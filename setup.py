import os
import sys
from shutil import rmtree

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
here = os.path.abspath(os.path.dirname(__file__))
about = {}
with open(os.path.join(here, "_version.py")) as f:
    exec(f.read(), about)


with open("./README.md", "r") as md_read:
    __long_description__ = md_read.read()


__version__ = about["__version__"]


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
            rmtree(os.path.join(here, "dist"))
            rmtree(os.path.join(here, "build"))
            rmtree(os.path.join(here, "{0}.egg-info".format(__title__)))
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
    author=about["__author__"],
    author_email=__author_email__,
    license=__license__,
    packages=find_packages(exclude=("test",)),
    keywords=__keywords__,
    py_modules=__modules__,
    zip_safe=False,
    long_description=__long_description__,
    long_description_content_type="text/markdown",
    include_package_data=True,
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
)
