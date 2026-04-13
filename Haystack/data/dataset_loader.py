from datasets import load_dataset
from haystack import Document

def load_data():
    dataset = load_dataset("bilgeyucel/seven-wonders", split="train")
    docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
    return docs
