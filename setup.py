import setuptools
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt") as fh:
    requirements = fh.read().splitlines()

setuptools.setup(
    name="rat",
    version="0.0.1",
    description="A regression package",
    long_description=long_description,
    url="https://github.com/bbbales2/regressions",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
    rust_extensions=[RustExtension("rat.one_draw", binding=Binding.PyO3, debug=False)],
    scripts=["bin/rat"],
    packages=setuptools.find_packages(),
    python_requires=">=3.10",
    install_requires=requirements,
    zip_safe=False,
    package_data={"rat": ["grammar.ebnf"]},
)
