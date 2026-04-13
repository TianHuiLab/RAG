# query_handler.py
class QueryHandler:
    def __init__(self, pipeline):
        self.pipeline = pipeline
    
    def query(self, question):
        """执行查询"""
        try:
            response = self.pipeline.run({
                "text_embedder": {"text": question},
                "prompt_builder": {"question": question}
            })
            return response["llm"]["replies"][0].text
        except Exception as e:
            return f"查询出错: {str(e)}"