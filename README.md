SkipAtom
========

SkipAtom is an approach for creating distributed representations of atoms, for use in Machine Learning contexts. It is
based on the Skip-gram model used widely in Natural Language Processing. The SkipAtom approach is described in the 
paper _"Distributed Representations of Atoms and Materials for Machine Learning"_,
[https://arxiv.org/abs/2107.14664](https://arxiv.org/abs/2107.14664).

SkipAtom can be installed with:
```
pip install skipatom
```
However, this will install a minimal implementation that can be used to work with existing SkipAtom embeddings only. To 
train new SkipAtom embeddings, SkipAtom should be installed with:
```
pip install skipatom[training]
```

Pre-trained 30- and 200-dimensional SkipAtom vectors for 86 atom types are available in the `data` directory, 
in the `mp_2020_10_09.dim30.model` and `mp_2020_10_09.dim200.model` files. To use the pre-trained vectors, follow the 
example in step 4, below.

To create SkipAtom vectors, follow steps 1 to 3 below. A dataset of inorganic crystal structures is required. A dataset 
of 126,335 structures obtained from the [Materials Project](https://materialsproject.org/) is available in 
`data/mp_2020_10_09.pkl.gz`.  From this dataset, pairs of co-occurring atoms will be derived, as depicted in the 
schematic below:

<img src="resources/schematic.png" width="85%"/>

These pairs will be used in the training of the SkipAtom vectors. Pairs that were previously derived from the 
Materials Project dataset are available in the file `data/mp_2020_10_09.pairs.csv.gz`.

_(NOTE: For the following steps 1 to 3, the programs `create_cooccurrence_pairs`, `create_skipatom_training_data` and 
`create_skipatom_embeddings` will be on the command line path if SkipAtom was installed with 
`pip install skipatom[training]`.)_

1. Create the co-occurrence pairs:
```
$ create_cooccurrence_pairs \
--data data/mp_2020_10_09.pkl.gz \
--out data/mp_2020_10_09.pairs.csv.gz.2 \
--processes 70 --workers 200 -z
```

2. Prepare the data for training:
```
$ create_skipatom_training_data \
--data data/mp_2020_10_09.pairs.csv.gz \
--out data/mp_2020_10_09.training.data
```

3. Create the SkipAtom embeddings:
```
$ create_skipatom_embeddings \
--data data/mp_2020_10_09.training.data \
--out data/mp_2020_10_09.dim200.model \
--dim 200 --step 0.01 --epochs 10 --batch 1024
```

4. Load and use the model:
```python
from skipatom import SkipAtomInducedModel

model = SkipAtomInducedModel.load(
    "data/mp_2020_10_09.dim200.model", 
    "data/mp_2020_10_09.training.data", 
    min_count=2e7, top_n=5)

# atom vector for Si
print(model.vectors[model.dictionary["Si"]])
```

### Pooling Operations

The `skipatom` module also contains several utility functions for pooling atom vectors into distributed representations 
of compounds. For example, to create a sum-pooled representation for `Bi2Te3`, use the `sum_pool` function:
```python
from skipatom import SkipAtomInducedModel, sum_pool
from pymatgen import Composition

model = SkipAtomInducedModel.load(
    "data/mp_2020_10_09.dim200.model", 
    "data/mp_2020_10_09.training.data", 
    min_count=2e7, top_n=5)

comp = Composition("Bi2Te3")
pooled = sum_pool(comp, model.dictionary, model.vectors)
# sum-pooled atom vectors representing Bi2Te3 
print(pooled)
``` 

### Neural Network Models

The `skipatom` module contains Keras-based implementations of an ElemNet-type neural network (for both 
regression and classification), and the Elpasolite neural network described by Zhou et al, in 2018. To use these, it is
necessary to have `tensorflow` in the environment. (Have a look at either the `requirements.txt` file or the 
`environment.yml` file for a full list of dependencies.) 

For more information regarding these models, see:

> Jha, D., Ward, L., Paul, A., Liao, W. K., Choudhary, A., Wolverton, C., & Agrawal, A. (2018). "ElemNet: Deep Learning 
the Chemistry of Materials From only Elemental Composition." Scientific reports, 8(1), 1-13.

> Zhou, Quan, et al. "Learning atoms for materials discovery."
Proceedings of the National Academy of Sciences 115.28 (2018): E6411-E6417. 

### One-hot Vectors

For convenience, a class for assigning one-hot vectors to atoms is included in the `skipatom` module. The following 
example demonstrates how to use the class:
```python
from skipatom import OneHotVectors

model = OneHotVectors(["Te", "Bi", "Se"])

# one-hot atom vector for Se
print(model.vectors[model.dictionary["Se"]])
# [0. 0. 1.]
```

- - - - - - - - -

This repository includes data from the [Materials Project](https://materialsproject.org/). 
> A. Jain*, S.P. Ong*, G. Hautier, W. Chen, W.D. Richards, S. Dacek, S. Cholia, D. Gunter, D. Skinner, G. Ceder, K.A. 
Persson (*=equal contributions). The Materials Project: A materials genome approach to accelerating materials innovation.
APL Materials, 2013, 1(1), 011002. 