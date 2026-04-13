# pipeline.py
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage
from haystack.components.generators.chat import OpenAIChatGenerator
from haystack import Pipeline

class PipelineBuilder:
    @staticmethod
    def create_retriever(document_store):
        """创建检索器"""
        return InMemoryEmbeddingRetriever(document_store)
    
    @staticmethod
    def create_prompt_template():
        """创建提示模板"""
        template = [
            ChatMessage.from_user(
                """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
            )
        ]
        return ChatPromptBuilder(template=template)
    
    @staticmethod
    def create_chat_generator(model_name, timeout):
        """创建聊天生成器"""
        return OpenAIChatGenerator(
            model=model_name,
            timeout=timeout,
        )
    
    @staticmethod
    def build_rag_pipeline(text_embedder, retriever, prompt_builder, chat_generator):
        """构建RAG管道"""
        pipeline = Pipeline()
        
        # 添加组件
        pipeline.add_component("text_embedder", text_embedder)
        pipeline.add_component("retriever", retriever)
        pipeline.add_component("prompt_builder", prompt_builder)
        pipeline.add_component("llm", chat_generator)
        
        # 连接组件
        pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
        pipeline.connect("retriever", "prompt_builder")
        pipeline.connect("prompt_builder.prompt", "llm.messages")
        
        return pipeline