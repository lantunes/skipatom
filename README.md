SkipAtom
========

SkipAtom is an approach for creating distributed representations of atoms, for use in Machine Learning contexts. It is
based on the Skip-gram model used widely in Natural Language Processing.

Pre-trained 200-dimensional SkipAtom vectors for 86 atom types are available in the `data` directory, in the
`matproj_2020_10_09.pairs.dim200.model` and `matproj_2020_10_09.pairs.training.data` files. To use the pre-trained 
vectors, follow the example in step 4, below. 

To create SkipAtom vectors, follow steps 1 to 3 below. A dataset of inorganic crystal structures is required. A dataset 
of 126,335 structures obtained from the [Materials Project](https://materialsproject.org/) is available in 
`data/matproj_2020_10_09.pkl.gz`.  From this dataset, pairs of co-occurring atoms will be derived, as depicted in the 
schematic below:

<img src="resources/schematic.png" width="50%"/>

These pairs will be used in the training of the SkipAtom vectors.

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
model = SkipAtomInducedModel.load(pairs, td, min_count=20000000, top_n=5)

# atom vector for Si
print(model.vectors[model.dictionary["Si"]])
```
