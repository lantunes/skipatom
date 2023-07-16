from setuptools import setup, find_packages
import re

INIT_FILE = "skipatom/__init__.py"

with open(INIT_FILE) as fid:
    file_contents = fid.read()
    match = re.search(r"^__version__\s?=\s?['\"]([^'\"]*)['\"]", file_contents, re.M)
    if match:
        version = match.group(1)
    else:
        raise RuntimeError("Unable to find version string in %s" % INIT_FILE)

packages = find_packages(exclude=("tests", "demos", "data", "resources", "bin", "out",))

setup(name="skipatom",
      version=version,
      description="SkipAtom, Distributed representations of atoms, inspired by the Skip-gram model.",
      long_description="SkipAtom is an approach for creating distributed representations of atoms, for use in Machine "
                       "Learning contexts. It is based on the Skip-gram model used widely in "
                       "Natural Language Processing.",
      license="MIT",
      classifiers=[
            "Development Status :: 5 - Production/Stable",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "Intended Audience :: System Administrators",
            "Intended Audience :: Information Technology",
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
      ],
      url="http://github.com/lantunes/skipatom",
      author="Luis M. Antunes",
      author_email="luis@materialis.ai",
      packages=packages,
      keywords=["machine learning", "materials science", "materials informatics", "distributed representations", "chemistry"],
      python_requires=">=3.8",
      install_requires=["numpy"],
      extras_require={
            "training": [
                "numpy ~= 1.22.0",
                "tensorflow >= 2.3.2",
                "matbench >= 0.6",
                "pymatgen >= 2023.7.14",
                "pandas >= 2.0.3",
                "scikit-learn >= 1.0.1",
                "tqdm >= 4.65.0",
            ]
      },
      entry_points={
          "console_scripts": [
              "create_cooccurrence_pairs = skipatom.create_cooccurrence_pairs:main",
              "create_skipatom_embeddings = skipatom.create_skipatom_embeddings:main",
              "create_skipatom_training_data = skipatom.create_skipatom_training_data:main"
          ],
      })
