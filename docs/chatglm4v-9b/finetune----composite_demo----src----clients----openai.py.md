# `.\chatglm4-finetune\composite_demo\src\clients\openai.py`

```
"""
OpenAI API client.  # 定义一个文档字符串，说明该模块是 OpenAI API 客户端
"""
from openai import OpenAI  # 从 openai 模块导入 OpenAI 类
from collections.abc import Generator  # 从 collections.abc 导入 Generator 类型

from client import Client, process_input, process_response  # 从 client 模块导入 Client 类及处理输入和输出的函数
from conversation import Conversation  # 从 conversation 模块导入 Conversation 类

def format_openai_tool(origin_tools):  # 定义一个函数，将原始工具格式化为 OpenAI 工具
    openai_tools = []  # 初始化一个空列表，用于存储格式化后的 OpenAI 工具
    for tool in origin_tools:  # 遍历每个原始工具
        openai_param={}  # 初始化一个空字典，用于存储工具参数
        for param in tool['params']:  # 遍历工具的参数
            openai_param[param['name']] = {}  # 将每个参数名称添加到字典中，值为空字典
        openai_tool = {  # 创建一个字典，表示格式化后的 OpenAI 工具
            "type": "function",  # 设置工具类型为函数
            "function": {  # 定义函数相关的信息
                "name": tool['name'],  # 设置函数名称
                "description": tool['description'],  # 设置函数描述
                "parameters": {  # 定义函数参数
                    "type": "object",  # 设置参数类型为对象
                    "properties": {  # 定义参数的属性
                        param['name']: {'type': param['type'], 'description': param['description']} for param in tool['params']  # 将每个参数名称及其类型和描述添加到属性中
                    },
                    "required": [param['name'] for param in tool['params'] if param['required']]  # 获取必需参数的名称列表
                    }
                }
            }
        openai_tools.append(openai_tool)  # 将格式化后的工具添加到列表中
    return openai_tools  # 返回格式化后的 OpenAI 工具列表

class APIClient(Client):  # 定义 APIClient 类，继承自 Client 类
    def __init__(self, model_path: str):  # 初始化方法，接受模型路径作为参数
        base_url = "http://127.0.0.1:8000/v1/"  # 定义基础 URL
        self.client = OpenAI(api_key="EMPTY", base_url=base_url)  # 创建 OpenAI 客户端实例，API 密钥设置为“EMPTY”
        self.use_stream = False  # 设置使用流的标志为 False
        self.role_name_replace = {'observation': 'tool'}  # 定义角色名称替换映射

    def generate_stream(  # 定义生成流的方法
        self,
        tools: list[dict],  # 接受工具的列表
        history: list[Conversation],  # 接受对话历史的列表
        **parameters,  # 接受额外参数
    ) -> Generator[tuple[str | dict, list[dict]]]:  # 返回生成器，类型为字符串或字典的元组和字典的列表
        chat_history = process_input(history, '', role_name_replace=self.role_name_replace)  # 处理输入历史，返回聊天历史
        # messages = process_input(history, '', role_name_replace=self.role_name_replace)  # 注释掉的代码，处理输入历史，返回消息（未使用）
        openai_tools = format_openai_tool(tools)  # 格式化工具列表
        response = self.client.chat.completions.create(  # 调用 OpenAI 客户端创建聊天补全请求
            model="glm-4",  # 指定使用的模型
            messages=chat_history,  # 提供聊天历史
            tools=openai_tools,  # 提供格式化后的工具
            stream=self.use_stream,  # 指定是否使用流
            max_tokens=parameters["max_new_tokens"],  # 设置最大生成的 tokens 数量
            temperature=parameters["temperature"],  # 设置生成的温度参数
            presence_penalty=1.2,  # 设置存在惩罚的值
            top_p=parameters["top_p"],  # 设置 top-p 采样的值
            tool_choice="auto"  # 设置工具选择为自动
        )
        output = response.choices[0].message  # 获取响应中的第一个选择的消息
        if output.tool_calls:  # 检查输出是否包含工具调用
            glm4_output = output.tool_calls[0].function.name + '\n' + output.tool_calls[0].function.arguments  # 构建工具调用的输出
        else:  # 如果没有工具调用
            glm4_output = output.content  # 获取响应内容
        yield process_response(glm4_output, chat_history)  # 处理输出并生成结果
```