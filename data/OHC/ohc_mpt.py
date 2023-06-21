from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, pipeline
import pandas as pd
import torch

from util.print_stat import print_information

model = "mosaicml/mpt-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=200,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
template = """Comments containing any form of non-acceptable language (profanity) or a targeted offense, which can be veiled or direct are offensive comments. This includes insults, threats, and postscontaining profane language or swear words. Comments that do not contain offense or profanity are not offensive.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

test = pd.read_csv('data/OHC/ohc_test.csv', sep="\t")
final_predictions = []

for index, row in test.iterrows():
    question = "Is this comment offensive or not? Comment: " + row['Text']
    response = llm_chain.run(question)
    if "not offensive" in response:
        final_predictions.append("NOT")
    else:
        final_predictions.append("OFF")

test['predictions'] = final_predictions
print_information(test, "predictions", "Class")





