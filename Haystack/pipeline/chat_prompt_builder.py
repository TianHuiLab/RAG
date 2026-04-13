# pipeline/chat_prompt_builder.py
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

def build_chat_prompt_template():
    """
    构建聊天提示模板
    """
    try:
        print("正在构建提示模板...")
        
        # 定义提示模板
        template = [
            ChatMessage.from_user(
                """You are a helpful assistant. Answer the question based on the provided context.

Context:
{% for document in documents %}
{{ document.content }}
{% endfor %}

Question: {{ question }}

Answer the question based on the context above. If the answer is not in the context, say "I don't have enough information to answer this question."

Answer:"""
            )
        ]
        
        # 创建提示构建器，明确指定必需变量
        prompt_builder = ChatPromptBuilder(
            template=template,
            required_variables=["documents", "question"]  # 明确指定必需变量
        )
        
        print("✓ 提示模板构建完成")
        return prompt_builder
        
    except Exception as e:
        print(f"❌ 构建提示模板失败: {e}")
        raise