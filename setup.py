from distutils import util

from setuptools import find_packages, setup

main_ns = {}
ver_path = util.convert_path("chia/version.py")

with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

setup(
    name="chia",
    version=main_ns["__version__"],
    packages=find_packages(),
    python_requires=">=3.7",
    include_package_data=True,
    install_requires=[
        "python-configuration~=0.7",
        "nltk~=3.5",
        "imageio~=2.6",
        "pillow~=7.1.0",
        "gputil~=1.4.0",
        "networkx~=2.4",
        "numpy~=1.18.5",
        "tensorflow-addons==0.11.1",
        "tensorflow==2.3.0",
    ],
    # metadata to display on PyPI
    author="Clemens-Alexander Brust",
    author_email="clemens-alexander.brust@uni-jena.de",
    description="Concept Hierarchies for Incremental and Active Learning",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
        "Environment :: GPU",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
