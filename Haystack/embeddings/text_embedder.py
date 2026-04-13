from haystack.components.embedders import SentenceTransformersTextEmbedder

def create_text_embedder(model_name):
    text_embedder = SentenceTransformersTextEmbedder(model=model_name)
    return text_embedder
