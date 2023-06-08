import os
import shutil

import pandas as pd
import sklearn
import torch
from deepoffense.classification import ClassificationModel
from sklearn.model_selection import train_test_split

from data.OHC.ohc_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, args, SEED
from util.TestInstance import TestInstance
from util.evaluation import macro_f1, weighted_f1
from util.label_converter import encode, decode
from util.print_stat import print_information

if not os.path.exists(TEMP_DIRECTORY): os.makedirs(TEMP_DIRECTORY)

davidson_train = pd.read_csv('data/DAVIDSON/davidson_train.csv', sep="\t")
davidson_test = pd.read_csv('data/DAVIDSON/davidson_test.csv', sep="\t")
davidson_test_sr = pd.read_csv('data/DAVIDSON/davidson_test_sr.csv', sep="\t")
davidson_test_rs = pd.read_csv('data/DAVIDSON/davidson_test_rs.csv', sep="\t")

hasoc_train = pd.read_csv('data/HASOC/hasoc_train.csv', sep="\t")
hasoc_test = pd.read_csv('data/HASOC/hasoc_test.csv', sep="\t")
hasoc_test_sr = pd.read_csv('data/HASOC/hasoc_test_sr.csv', sep="\t")
hasoc_test_rs = pd.read_csv('data/HASOC/hasoc_test_rs.csv', sep="\t")

hateval_train = pd.read_csv('data/HATEVAL/hateval_train.csv', sep="\t")
hateval_test = pd.read_csv('data/HATEVAL/hateval_test.csv', sep="\t")
hateval_test_sr = pd.read_csv('data/HATEVAL/hateval_test_sr.csv', sep="\t")
hateval_test_rs = pd.read_csv('data/HATEVAL/hateval_test_rs.csv', sep="\t")

hatexplain_train = pd.read_csv('data/HateXplain/hatexplain_train.csv', sep="\t")
hatexplain_test = pd.read_csv('data/HateXplain/hatexplain_test.csv', sep="\t")
hatexplain_test_sr = pd.read_csv('data/HateXplain/hatexplain_test_sr.csv', sep="\t")
hatexplain_test_rs = pd.read_csv('data/HateXplain/hatexplain_test_rs.csv', sep="\t")

ohc_train = pd.read_csv('data/OHC/ohc_train.csv', sep="\t")
ohc_test = pd.read_csv('data/OHC/ohc_test.csv', sep="\t")
ohc_test_sr = pd.read_csv('data/OHC/ohc_test_sr.csv', sep="\t")
ohc_test_rs = pd.read_csv('data/OHC/ohc_test_rs.csv', sep="\t")

olid_train = pd.read_csv('data/OLID/olid_train.csv', sep="\t")
olid_test = pd.read_csv('data/OLID/olid_test.csv', sep="\t")
olid_test_sr = pd.read_csv('data/OLID/olid_test_sr.csv', sep="\t")
olid_test_rs = pd.read_csv('data/OLID/olid_test_rs.csv', sep="\t")

tcc_train = pd.read_csv('data/TCC/tcc_train.csv', sep="\t")
tcc_test = pd.read_csv('data/TCC/tcc_test.csv', sep="\t")
tcc_test_sr = pd.read_csv('data/TCC/tcc_test_sr.csv', sep="\t")
tcc_test_rs = pd.read_csv('data/TCC/tcc_test_rs.csv', sep="\t")

trac_train = pd.read_csv('data/TRAC/trac_train.csv', sep="\t")
trac_test = pd.read_csv('data/TRAC/trac_test.csv', sep="\t")
trac_test_sr = pd.read_csv('data/TRAC/trac_test_sr.csv', sep="\t")
trac_test_rs = pd.read_csv('data/TRAC/trac_test_rs.csv', sep="\t")

# Prepare training files
train = pd.concat([ohc_train], ignore_index=True)
test_all = pd.concat([davidson_test, hasoc_test, hateval_test, hatexplain_test, ohc_test, olid_test, tcc_test, trac_test], ignore_index=True)
test_all_sr = pd.concat([davidson_test_sr, hasoc_test_sr, hateval_test_sr, hatexplain_test_sr, ohc_test_sr, olid_test_sr, tcc_test_sr, trac_test_sr], ignore_index=True)
test_all_rs = pd.concat([davidson_test_rs, hasoc_test_rs, hateval_test_rs, hatexplain_test_rs, ohc_test_rs, olid_test_rs, tcc_test_rs, trac_test_rs], ignore_index=True)

train = train.rename(columns={'Text': 'text', 'Class': 'labels'})
train = train[['text', 'labels']]
train = train.sample(frac=1).reset_index(drop=True)
train['labels'] = encode(train["labels"])

train = train[:1000]

test_files_dict = {
    "DAVIDSON": davidson_test,
    "DAVIDSON_RS": davidson_test_rs,
    "DAVIDSON_SR": davidson_test_sr,
    "HASOC": hasoc_test,
    "HASOC_RS": hasoc_test_rs,
    "HASOC_SR": hasoc_test_sr,
    "HATEVAL": hateval_test,
    "HATEVAL_RS": hateval_test_rs,
    "HATEVAL_SR": hateval_test_sr,
    "HateXplain": hatexplain_test,
    "HateXplain_RS": hatexplain_test_rs,
    "HateXplain_SR": hatexplain_test_sr,
    "OHC": ohc_test,
    "OHC_RS": ohc_test_rs,
    "OHC_SR": ohc_test_sr,
    "OLID": olid_test,
    "OLID_RS": olid_test_rs,
    "OLID_SR": olid_test_sr,
    "TCC": tcc_test,
    "TCC_RS": tcc_test_rs,
    "TCC_SR": tcc_test_sr,
    "TRAC": trac_test,
    "TRAC_RS": trac_test_rs,
    "TRAC_SR": trac_test_sr,
    "All": test_all,
    "All_RS": test_all_rs,
    "All_SR": test_all_sr
}

test_instances = []

for name, file in test_files_dict.items():
    test_instance = TestInstance(file, args, name)
    test_instances.append(test_instance)

# Train the model
print("Started Training")

# You can set class weights by using the optional weight argument

if args["evaluate_during_training"]:
    for i in range(args["n_fold"]):
        if os.path.exists(args['output_dir']) and os.path.isdir(args['output_dir']):
            shutil.rmtree(args['output_dir'])
        print("Started Fold {}".format(i))

        model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                    use_cuda=torch.cuda.is_available(),
                                    cuda_device=3)

        train_df, eval_df = train_test_split(train, test_size=0.2, random_state=SEED * i)
        model.train_model(train_df, eval_df=eval_df, macro_f1=macro_f1, weighted_f1=weighted_f1,
                          accuracy=sklearn.metrics.accuracy_score)
        model = ClassificationModel(MODEL_TYPE, args["best_model_dir"], args=args,
                                    use_cuda=torch.cuda.is_available(), cuda_device=3)

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

else:
    model = ClassificationModel(MODEL_TYPE, MODEL_NAME, args=args,
                                use_cuda=torch.cuda.is_available(),
                                cuda_device=2)
    model.train_model(train, macro_f1=macro_f1, weighted_f1=weighted_f1, accuracy=sklearn.metrics.accuracy_score)
    for test_instance in test_instances:
        predictions, raw_outputs = model.predict(test_instance.get_sentences())
        test_instance.df['predictions'] = predictions

for test_instance in test_instances:
    print()
    print("==================== Results for " + test_instance.name + "========================")
    test_instance.df['predictions'] = decode(test_instance.df['predictions'])
    test_instance.df['labels'] = decode(test_instance.df['labels'])
    print_information(test_instance.df, "predictions", "labels")
