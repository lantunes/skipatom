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

packages = find_packages(exclude=("tests", "demos", "data", "resources",))

setup(name="skipatom",
      version=version,
      description="SkipAtom, Distributed representations of atoms, inspired by the Skip-gram model.",
      long_description="SkipAtom is an approach for creating distributed representations of atoms, for use in Machine "
                       "Learning contexts. It is based on the Skip-gram model used widely in "
                       "Natural Language Processing.",
      license="GNU General Public License v3",
      classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Programming Language :: Python :: 3.6',
      ],
      url='http://github.com/lantunes/skipatom',
      author="Luis M. Antunes",
      author_email="lantunes@gmail.com",
      packages=packages,
      keywords=["machine learning", "materials science", "materials informatics", "distributed representations", "chemistry"],
      python_requires='>=3.6',
      install_requires=["numpy >= 1.18.5", "tqdm >= 4.61.1"],
      extras_require={
            "training": ["pymatgen >= 2021.2.8.1", "pandas >= 1.1.5", "tensorflow == 2.3.2"]
      },
      entry_points={
          'console_scripts': [
              "create_cooccurrence_pairs = skipatom.create_cooccurrence_pairs:main",
              "create_skipatom_embeddings = skipatom.create_skipatom_embeddings:main",
              "create_skipatom_training_data = skipatom.create_skipatom_training_data:main"
          ],
      })
