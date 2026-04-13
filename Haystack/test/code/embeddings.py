# embeddings.py
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder

class EmbeddingManager:
    def __init__(self, model_name):
        self.model_name = model_name
        self.doc_embedder = None
        self.text_embedder = None
        
    def initialize_embedders(self):
        """初始化嵌入器"""
        self.doc_embedder = SentenceTransformersDocumentEmbedder(model=self.model_name)
        self.doc_embedder.warm_up()
        
        self.text_embedder = SentenceTransformersTextEmbedder(model=self.model_name)
        return self
    
    def embed_documents(self, documents):
        """为文档生成嵌入"""
        if not self.doc_embedder:
            self.initialize_embedders()
        return self.doc_embedder.run(documents)
    
    def get_text_embedder(self):
        """获取文本嵌入器"""
        if not self.text_embedder:
            self.initialize_embedders()
        return self.text_embedder