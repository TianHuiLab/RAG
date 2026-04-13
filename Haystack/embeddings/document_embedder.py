from haystack.components.embedders import SentenceTransformersDocumentEmbedder

def create_document_embedder(model_name):
    doc_embedder = SentenceTransformersDocumentEmbedder(model=model_name)
    doc_embedder.warm_up()
    return doc_embedder
