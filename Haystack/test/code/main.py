# main.py
from config import Config, setup_environment
from document_store import DocumentStoreManager
from embeddings import EmbeddingManager
from pipeline import PipelineBuilder
from query_handler import QueryHandler

def main():
    # 1. 设置环境
    setup_environment()
    
    # 2. 初始化文档存储
    print("加载文档...")
    doc_manager = DocumentStoreManager(Config.DATASET_NAME, Config.DATASET_SPLIT)
    docs = doc_manager.load_documents()
    
    # 3. 初始化嵌入模型
    print("初始化嵌入模型...")
    embed_manager = EmbeddingManager(Config.EMBEDDING_MODEL)
    embed_manager.initialize_embedders()
    
    # 4. 嵌入文档并存储
    print("生成文档嵌入...")
    docs_with_embeddings = embed_manager.embed_documents(docs)
    doc_manager.write_documents(docs_with_embeddings["documents"])
    
    # 5. 构建检索器
    retriever = PipelineBuilder.create_retriever(doc_manager.get_store())
    
    # 6. 构建管道组件
    prompt_builder = PipelineBuilder.create_prompt_template()
    chat_generator = PipelineBuilder.create_chat_generator(
        Config.LLM_MODEL, 
        Config.TIMEOUT
    )
    
    # 7. 构建完整管道
    print("构建RAG管道...")
    rag_pipeline = PipelineBuilder.build_rag_pipeline(
        text_embedder=embed_manager.get_text_embedder(),
        retriever=retriever,
        prompt_builder=prompt_builder,
        chat_generator=chat_generator
    )
    
    # 8. 创建查询处理器
    query_handler = QueryHandler(rag_pipeline)
    
    # 9. 执行查询
    questions = [
        "Where is Gardens of Babylon?",
        # 可以添加更多问题...
    ]
    
    for question in questions:
        print(f"\n问题: {question}")
        print("正在查询...")
        answer = query_handler.query(question)
        print(f"回答: {answer}")

if __name__ == "__main__":
    main()