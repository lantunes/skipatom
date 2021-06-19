SkipAtom
========

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

Pre-trained 200-dimensional SkipAtom vectors for 86 atom types are available in the `data` directory, in the
`matproj_2020_10_09.pairs.dim200.model` and `matproj_2020_10_09.pairs.training.data` files. The vectors can be 
loaded and used as in step 4 in the instructions above.
