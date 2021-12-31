import pandas as pd
from offensive_nn.offensive_nn_model import OffensiveNNModel
from sklearn.model_selection import train_test_split

from data.HASOC.leave_hasoc_rnn_config import args
from util.TestInstance import TestInstance
from util.label_converter import encode, decode
from util.print_stat import print_information

davidson_train = pd.read_csv('data/DAVIDSON/davidson_train.csv', sep="\t")
davidson_test = pd.read_csv('data/DAVIDSON/davidson_test.csv', sep="\t")

hasoc_train = pd.read_csv('data/HASOC/hasoc_train.csv', sep="\t")
hasoc_test = pd.read_csv('data/HASOC/hasoc_test.csv', sep="\t")

hateval_train = pd.read_csv('data/HATEVAL/hateval_train.csv', sep="\t")
hateval_test = pd.read_csv('data/HATEVAL/hateval_test.csv', sep="\t")

hatexplain_train = pd.read_csv('data/HateXplain/hatexplain_train.csv', sep="\t")
hatexplain_test = pd.read_csv('data/HateXplain/hatexplain_test.csv', sep="\t")

ohc_train = pd.read_csv('data/OHC/ohc_train.csv', sep="\t")
ohc_test = pd.read_csv('data/OHC/ohc_test.csv', sep="\t")

olid_train = pd.read_csv('data/OLID/olid_train.csv', sep="\t")
olid_test = pd.read_csv('data/OLID/olid_test.csv', sep="\t")

tcc_train = pd.read_csv('data/TCC/tcc_train.csv', sep="\t")
tcc_test = pd.read_csv('data/TCC/tcc_test.csv', sep="\t")

trac_train = pd.read_csv('data/TRAC/trac_train.csv', sep="\t")
trac_test = pd.read_csv('data/TRAC/trac_test.csv', sep="\t")

# Prepare training files
train = pd.concat([davidson_train, hateval_train, hatexplain_train, ohc_train, olid_train, tcc_train, trac_train],
                  ignore_index=True)
train = train.rename(columns={'Text': 'text', 'Class': 'labels'})
train = train[['text', 'labels']]
train = train.sample(frac=1).reset_index(drop=True)
train['labels'] = encode(train["labels"])

test_files_dict = {
    "DAVIDSON": davidson_test,
    "HASOC": hasoc_test,
    "HATEVAL": hateval_test,
    "HateXplain": hatexplain_test,
    "OHC": ohc_test,
    "OLID": olid_test,
    "TCC": tcc_test,
    "TRAC": trac_test
}

test_instances = []

for name, file in test_files_dict.items():
    test_instance = TestInstance(file, args, name)
    test_instances.append(test_instance)

# Train the model
print("Started Training")


for i in range(args["n_fold"]):
    print("Started Fold {}".format(i))

    train_df, eval_df = train_test_split(train, test_size=0.2,  random_state=args["manual_seed"])

    model = OffensiveNNModel(model_type_or_path="lstm", embedding_model_name="word2vec-google-news-300",
                                 train_df=train_df, args=args, eval_df=eval_df)
    model.train_model()
    model = OffensiveNNModel(model_type_or_path=args["best_model_dir"])


    for test_instance in test_instances:
        predictions, raw_outputs = model.predict(test_instance.get_sentences())
        test_instance.test_preds[:, i] = predictions

    model = None

    print("Completed Fold {}".format(i))

# select majority class of each instance (row)
for test_instance in test_instances:
    final_predictions = []
    for row in test_instance.test_preds:
        row = row.tolist()
        final_predictions.append(int(max(set(row), key=row.count)))
    test_instance.df['predictions'] = final_predictions


for test_instance in test_instances:
    print()
    print("==================== Results for " + test_instance.name + "========================")
    test_instance.df['predictions'] = decode(test_instance.df['predictions'])
    test_instance.df['labels'] = decode(test_instance.df['labels'])
    print_information(test_instance.df, "predictions", "labels")

