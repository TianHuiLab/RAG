# config.py
import os

class Config:
    # 模型配置
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "gpt-5"
    
    # OpenAI配置
    OPENAI_API_KEY = "sk-IWcHVBG9f1UFOHYM3vyXBUJWgQfcbwH7KahozVAYFlLfXkO4"
    OPENAI_BASE_URL = "http://43.159.131.233:3001/v1"
    
    # 数据集配置
    DATASET_NAME = "bilgeyucel/seven-wonders"
    DATASET_SPLIT = "train"
    
    # 超参数
    TIMEOUT = 60

def setup_environment():
    """设置环境变量"""
    os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
    os.environ["OPENAI_BASE_URL"] = Config.OPENAI_BASE_URL