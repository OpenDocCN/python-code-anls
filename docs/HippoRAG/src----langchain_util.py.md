# `.\HippoRAG\src\langchain_util.py`

```py
# 导入 argparse 模块，用于处理命令行参数
import argparse
# 导入 os 模块，用于操作环境变量
import os

# 导入 tiktoken 库，用于处理文本编码
import tiktoken

# 获取针对 "gpt-3.5-turbo" 模型的编码器
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")


# 定义一个函数，用于计算文本的 token 数量
def num_tokens_by_tiktoken(text: str):
    # 使用 tiktoken 编码器对文本进行编码，并返回编码后的 token 数量
    return len(enc.encode(text))


# 定义一个类，用于表示语言模型
class LangChainModel:
    def __init__(self, provider: str, model_name: str, **kwargs):
        # 初始化语言模型的提供者、模型名称及其它参数
        self.provider = provider
        self.model_name = model_name
        self.kwargs = kwargs


# 定义一个函数，用于初始化 LangChain 语言模型
def init_langchain_model(llm: str, model_name: str, temperature: float = 0.0, max_retries=5, timeout=60, **kwargs):
    """
    从 langchain 库初始化语言模型。
    :param llm: 要使用的 LLM，例如 'openai'，'together'
    :param model_name: 要使用的模型名称，例如 'gpt-3.5-turbo'
    """
    # 如果 LLM 是 'openai'
    if llm == 'openai':
        # 从 langchain_openai 模块导入 ChatOpenAI 类
        from langchain_openai import ChatOpenAI
        # 确保模型名称以 'gpt-' 开头
        assert model_name.startswith('gpt-')
        # 返回一个初始化好的 ChatOpenAI 对象
        return ChatOpenAI(api_key=os.environ.get("OPENAI_API_KEY"), model=model_name, temperature=temperature, max_retries=max_retries, timeout=timeout, **kwargs)
    # 如果 LLM 是 'together'
    elif llm == 'together':
        # 从 langchain_together 模块导入 ChatTogether 类
        from langchain_together import ChatTogether
        # 返回一个初始化好的 ChatTogether 对象
        return ChatTogether(api_key=os.environ.get("TOGETHER_API_KEY"), model=model_name, temperature=temperature, **kwargs)
    # 如果 LLM 是 'ollama'
    elif llm == 'ollama':
        # 从 langchain_community.chat_models 模块导入 ChatOllama 类
        from langchain_community.chat_models import ChatOllama
        # 返回一个初始化好的 ChatOllama 对象
        return ChatOllama(model=model_name)  # e.g., 'llama3'
    # 如果 LLM 是 'llama.cpp'
    elif llm == 'llama.cpp':
        # 从 langchain_community.chat_models 模块导入 ChatLlamaCpp 类
        from langchain_community.chat_models import ChatLlamaCpp
        # 返回一个初始化好的 ChatLlamaCpp 对象，model_name 是模型路径（gguf 文件）
        return ChatLlamaCpp(model_path=model_name, verbose=True)  # model_name is the model path (gguf file)
    else:
        # 如果提供的 LLM 不在已实现的列表中，则抛出未实现异常
        raise NotImplementedError(f"LLM '{llm}' not implemented yet.")


# 如果该脚本是作为主程序运行
if __name__ == '__main__':
    # 创建一个 ArgumentParser 对象，用于解析命令行参数
    parser = argparse.ArgumentParser()
    # 添加 '--llm' 参数，用于指定要使用的语言模型
    parser.add_argument('--llm', type=str)
    # 添加 '--model_name' 参数，用于指定模型名称
    parser.add_argument('--model_name', type=str)
    # 添加 '--query' 参数，用于指定查询文本，默认值为 "who are you?"
    parser.add_argument('--query', type=str, help='query text', default="who are you?")
    # 解析命令行参数
    args = parser.parse_args()

    # 使用命令行参数初始化语言模型
    model = init_langchain_model(args.llm, args.model_name)
    # 构建消息列表，其中包括系统消息和用户消息
    messages = [("system", "You are a helpful assistant. Please answer the question from the user."), ("human", args.query)]
    # 使用模型处理消息列表并获取完成内容
    completion = model.invoke(messages)
    # 打印模型返回的内容
    print(completion.content)
```