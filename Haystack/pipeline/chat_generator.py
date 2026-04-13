from haystack.components.generators.chat import OpenAIChatGenerator

def create_chat_generator(model_name, timeout):
    chat_generator = OpenAIChatGenerator(
        model=model_name,
        timeout=timeout,
    )
    return chat_generator
