SkipAtom
========

SkipAtom is an approach for creating distributed representations of atoms, for use in Machine Learning contexts. It is
based on the Skip-gram model used widely in Natural Language Processing.

SkipAtom can be installed with:
```
pip install skipatom
```

Pre-trained 200-dimensional SkipAtom vectors for 86 atom types are available in the `data` directory, in the
`matproj_2020_10_09.pairs.dim200.model` and `matproj_2020_10_09.pairs.training.data` files. To use the pre-trained 
vectors, follow the example in step 4, below.

To create SkipAtom vectors, follow steps 1 to 3 below. A dataset of inorganic crystal structures is required. A dataset 
of 126,335 structures obtained from the [Materials Project](https://materialsproject.org/) is available in 
`data/matproj_2020_10_09.pkl.gz`.  From this dataset, pairs of co-occurring atoms will be derived, as depicted in the 
schematic below:

<img src="resources/schematic.png" width="85%"/>

These pairs will be used in the training of the SkipAtom vectors. Pairs that were previously derived from the 
Materials Project dataset are available in the file `data/matproj_2020_10_09.pairs.csv.gz`.

1. Create the co-occurrence pairs:
```
python bin/create_cooccurrence_pairs.py \
--data out/matproj_2020_10_09.pkl.gz \
--out out/matproj_2020_10_09.pairs.csv.gz.2 \
--processes 70 --workers 200 -z
```

2. Prepare the data for training:
```
python bin/create_skipatom_training_data.py \
--data out/matproj_2020_10_09.pairs.csv.gz \
--out out/matproj_2020_10_09.pairs.training.data
```

3. Create the SkipAtom embeddings:
```
python bin/create_skipatom_embeddings.py \
--data out/matproj_2020_10_09.pairs.training.data \
--out out/matproj_2020_10_09.pairs.dim200.model \
--dim 200 --step 0.01 --epochs 10 --batch 1024
```

4. Load and use the model:
```python
from skipatom import SkipAtomInducedModel

pairs = "data/matproj_2020_10_09.pairs.dim200.model"
td = "data/matproj_2020_10_09.pairs.training.data"
model = SkipAtomInducedModel.load(pairs, td, min_count=2e7, top_n=5)

# atom vector for Si
print(model.vectors[model.dictionary["Si"]])
```

### Pooling Operations

The `skipatom` module also contains several utility functions for pooling atom vectors into distributed representations 
of compounds. For example, to create a sum-pooled representation for `Bi2Te3`:
```python
from skipatom import SkipAtomInducedModel, get_sum_pooled
from pymatgen import Composition

pairs = "data/matproj_2020_10_09.pairs.dim200.model"
td = "data/matproj_2020_10_09.pairs.training.data"
model = SkipAtomInducedModel.load(pairs, td, min_count=2e7, top_n=5)

comp = Composition("Bi2Te3")
pooled = get_sum_pooled(comp, model.dictionary, model.vectors)
# sum-pooled atom vectors representing Bi2Te3 
print(pooled)
``` 

### Neural Network Models

The `lib` directory in this project contains Keras-based implementations of an ElemNet-type neural network (for both 
regression and classification), and the Elpasolite neural network described by Zhou et al, in 2018. To use these, it is
best to check out the code for this repository locally, and install the dependencies using either the `requirements.txt` 
or the `environment.yml` (for Conda environments). 

For more information regarding these models, see:

> Jha, D., Ward, L., Paul, A., Liao, W. K., Choudhary, A., Wolverton, C., & Agrawal, A. (2018). "ElemNet: Deep Learning 
the Chemistry of Materials From only Elemental Composition." Scientific reports, 8(1), 1-13.

> Zhou, Quan, et al. "Learning atoms for materials discovery."
Proceedings of the National Academy of Sciences 115.28 (2018): E6411-E6417. 
