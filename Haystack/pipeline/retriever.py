from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever

def create_retriever(document_store):
    retriever = InMemoryEmbeddingRetriever(document_store)
    return retriever
