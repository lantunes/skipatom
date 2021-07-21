from setuptools import setup, find_packages

packages = find_packages(exclude=("tests", "demos",))

setup(name="skipatom",
      version="1.0.1",
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
      install_requires=["pymatgen == 2021.2.8.1", "numpy == 1.18.5", "tqdm == 4.61.1", "jax == 0.2.5", "jaxlib == 0.1.56"])
