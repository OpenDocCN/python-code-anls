# `.\chatglm4-finetune\composite_demo\src\clients\hf.py`

```py
"""
HuggingFace client.  # HuggingFace 客户端的文档说明
"""

import threading  # 导入 threading 模块以支持多线程
from collections.abc import Generator  # 从 abc 模块导入 Generator 类型以定义生成器
from threading import Thread  # 从 threading 模块导入 Thread 类以便于创建线程

import torch  # 导入 PyTorch 库以进行张量操作
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer  # 从 transformers 导入必要的类

from client import Client, process_input, process_response  # 从 client 模块导入 Client 类和处理函数
from conversation import Conversation  # 从 conversation 模块导入 Conversation 类


class HFClient(Client):  # 定义 HFClient 类，继承自 Client 类
    def __init__(self, model_path: str):  # 构造函数，接收模型路径作为参数
        # 使用预训练的模型路径初始化分词器，信任远程代码
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True,
        )
        # 使用预训练的模型路径初始化因果语言模型，信任远程代码，设置数据类型和设备映射
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,  # 使用 bfloat16 数据类型
            device_map="cuda",  # 将模型加载到 GPU
        ).eval()  # 将模型设置为评估模式

    def generate_stream(  # 定义生成流的方法，接收工具、历史记录和可变参数
        self,
        tools: list[dict],  # 工具列表，每个工具为字典
        history: list[Conversation],  # 对话历史记录列表
        **parameters,  # 其他参数
    ) -> Generator[tuple[str | dict, list[dict]]]:  # 返回生成器，输出为字符串或字典的元组和字典列表
        # 处理输入的对话历史和工具
        chat_history = process_input(history, tools)
        # 使用分词器将对话历史转化为模型输入格式
        model_inputs = self.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,  # 添加生成提示
            tokenize=True,  # 对输入进行分词
            return_tensors="pt",  # 返回 PyTorch 张量
            return_dict=True,  # 返回字典格式
        ).to(self.model.device)  # 将模型输入移到模型的设备上
        # 初始化文本迭代流处理器
        streamer = TextIteratorStreamer(
            tokenizer=self.tokenizer,  # 使用分词器
            timeout=5,  # 设置超时时间为5秒
            skip_prompt=True,  # 跳过提示
        )
        # 准备生成参数，包括模型输入和其他参数
        generate_kwargs = {
            **model_inputs,  # 解包模型输入
            "streamer": streamer,  # 添加流处理器
            "eos_token_id": [151329, 151336, 151338],  # 设置结束标记 ID
            "do_sample": True,  # 启用采样
        }
        generate_kwargs.update(parameters)  # 更新生成参数，包含额外的可变参数
        # 创建线程以生成文本
        t = Thread(target=self.model.generate, kwargs=generate_kwargs)  # 将生成方法作为线程目标
        t.start()  # 启动线程
        total_text = ""  # 初始化总文本字符串
        for token_text in streamer:  # 遍历生成的每个令牌文本
            total_text += token_text  # 将令牌文本追加到总文本中
            # 生成并返回处理后的响应
            yield process_response(total_text, chat_history)
```