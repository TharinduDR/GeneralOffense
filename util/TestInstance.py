from util.label_converter import encode
import numpy as np


class TestInstance:
    def __init__(self, df, args, name):
        self.name = name
        self.df = df
        self.args = args
        self.df['labels'] = encode(self.df["labels"])
        self.test_sentences = self.df['text'].tolist()
        self.test_preds = np.zeros((len(self.df), self.args["n_fold"]))
