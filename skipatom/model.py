from .training_data import TrainingData
from .trainer import Trainer


class SkipAtomModel:
    def __init__(self, training_data, embeddings):
        self.vectors = embeddings
        self.dictionary = training_data.atom_to_index

    @staticmethod
    def load(model_file, training_data_file):
        td = TrainingData.load(training_data_file)
        embeddings = Trainer.load_embeddings(model_file)
        return SkipAtomModel(td, embeddings)
