# document_store.py
from haystack.document_stores.in_memory import InMemoryDocumentStore
from datasets import load_dataset
from haystack import Document

class DocumentStoreManager:
    def __init__(self, dataset_name, dataset_split):
        self.document_store = InMemoryDocumentStore()
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
    
    def load_documents(self):
        """加载数据集并转换为Document对象"""
        dataset = load_dataset(self.dataset_name, split=self.dataset_split)
        docs = [Document(content=doc["content"], meta=doc["meta"]) for doc in dataset]
        return docs
    
    def get_store(self):
        """获取文档存储实例"""
        return self.document_store
    
    def write_documents(self, documents):
        """写入文档到存储"""
        self.document_store.write_documents(documents)