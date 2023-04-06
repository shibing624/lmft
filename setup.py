# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('lmft/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

extras = {}
extras["dev"] = ["torch"]
extras["test"] = extras["dev"] + ["pytest", "pytest-xdist"]

setup(
    name='lmft',
    version=__version__,
    description='Language Model Fine-tuning Toolkit (LMFT)',
    license_files=["LICENSE"],
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author='XuMing',
    author_email='xuming624@qq.com',
    url='https://github.com/shibing624/lmft',
    license="Apache License 2.0",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    entry_points={},
    python_requires=">=3.7.0",
    keywords='lmft,GPT2,transformers,pytorch,language model',
    install_requires=[
        "loguru",
        "peft",
        "transformers>=4.27.1",
        "datasets",
        "tqdm",
    ],
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)

# Release checklist
# 1. Change the version in __init__.py and setup.py.
# 2. Commit these changes with the message: "Release: VERSION"
# 3. Add a tag in git to mark the release: "git tag VERSION -m 'Adds tag VERSION for pypi' "
#    Push the tag to git: git push --tags origin main
# 4. Run the following commands in the top-level directory:
#      python setup.py bdist_wheel
#      python setup.py sdist
# 5. Upload the package to the pypi test server first:
#      twine upload dist/* -r pypitest
#      twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/
# 6. Check that you can install it in a virtualenv by running:
#      pip install -i https://testpypi.python.org/pypi peft
# 7. Upload the final version to actual pypi:
#      twine upload dist/* -r pypi
# 8. Add release notes to the tag in github once everything is looking hunky-dory.
# 9. Update the version in __init__.py, setup.py to the new version "-dev" and push to master
