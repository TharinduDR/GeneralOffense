from util.label_converter import encode
import numpy as np


class TestInstance:
    def __init__(self, df, args, name):
        self.name = name
        self.df = df
        self.args = args
        self.df = self.df.rename(columns={'Text': 'text', 'Class': 'labels'})
        self.df['labels'] = encode(self.df["labels"])
        self.test_preds = np.zeros((len(self.df), self.args["n_fold"]))

    def get_sentences(self):
        return self.df['text'].tolist()
