from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd

from util.print_stat import print_information

tokenizer = AutoTokenizer.from_pretrained("bigscience/T0_3B")
model = AutoModelForSeq2SeqLM.from_pretrained("bigscience/T0_3B")


test = pd.read_csv('data/HASOC/hasoc_test.csv', sep="\t")
final_predictions = []

for index, row in test.iterrows():
    question = "Comments containing any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct are offensive comments. This includes insults, threats, and postscontaining profane language or swear words. Comments that do not contain offense or profanity are not offensive. Is this comment offensive or not? Comment: " + row['Text']
    inputs = tokenizer.encode(question, return_tensors="pt")
    outputs = model.generate(inputs)
    response = tokenizer.decode(outputs[0]).lower()
    if "no" in response.strip():
        final_predictions.append("NOT")
    else:
        final_predictions.append("OFF")


test['predictions'] = final_predictions
print_information(test, "predictions", "Class")
