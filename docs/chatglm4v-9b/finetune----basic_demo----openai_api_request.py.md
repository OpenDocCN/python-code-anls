# `.\chatglm4-finetune\basic_demo\openai_api_request.py`

```
"""
该脚本创建了一个 OpenAI 请求示例，用于与 glm-4-9b 模型进行交互，只需使用 OpenAI API 进行交互。
"""

# 从 openai 库导入 OpenAI 类
from openai import OpenAI
# 导入 base64 编码库
import base64

# 定义 API 的基本 URL
base_url = "http://127.0.0.1:8000/v1/"
# 创建 OpenAI 客户端，使用空的 API 密钥和自定义基本 URL
client = OpenAI(api_key="EMPTY", base_url=base_url)


# 定义与聊天模型交互的函数，接受是否使用流式响应的参数
def function_chat(use_stream=False):
    # 定义消息列表，包含用户发送的消息
    messages = [
        {
            "role": "user", "content": "What's the Celsius temperature in San Francisco?"
        },

        # 给出观察结果的注释示例
        # {
        #     "role": "assistant",
        #         "content": None,
        #         "function_call": None,
        #         "tool_calls": [
        #             {
        #                 "id": "call_1717912616815",
        #                 "function": {
        #                     "name": "get_current_weather",
        #                     "arguments": "{\"location\": \"San Francisco, CA\", \"format\": \"celsius\"}"
        #                 },
        #                 "type": "function"
        #             }
        #         ]
        # },
        # {
        #     "tool_call_id": "call_1717912616815",
        #     "role": "tool",
        #     "name": "get_current_weather",
        #     "content": "23°C",
        # }
    ]
    # 定义工具列表，包含获取当前天气的工具
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",  # 工具名称
                "description": "Get the current weather",  # 工具描述
                "parameters": {  # 工具参数定义
                    "type": "object",
                    "properties": {  # 参数属性
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",  # 位置参数描述
                        },
                        "format": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],  # 温度单位可选项
                            "description": "The temperature unit to use. Infer this from the users location.",  # 温度单位描述
                        },
                    },
                    "required": ["location", "format"],  # 必需的参数
                },
            }
        },
    ]

    # 所有工具示例：CogView
    # messages = [{"role": "user", "content": "帮我画一张天空的画画吧"}]
    # tools = [{"type": "cogview"}]

    # 所有工具示例：搜索工具
    # messages = [{"role": "user", "content": "今天黄金的价格"}]
    # tools = [{"type": "simple_browser"}]

    # 调用 OpenAI API 创建聊天响应
    response = client.chat.completions.create(
        model="glm-4",  # 指定使用的模型
        messages=messages,  # 传入消息列表
        tools=tools,  # 传入工具列表
        stream=use_stream,  # 是否使用流式响应
        max_tokens=256,  # 设置生成响应的最大 token 数量
        temperature=0.9,  # 设置温度以控制生成文本的多样性
        presence_penalty=1.2,  # 设置存在惩罚以增加话题多样性
        top_p=0.1,  # 设置累积概率阈值
        tool_choice="auto"  # 工具选择方式
    )
    # 检查是否有响应
    if response:
        # 如果使用流式响应，则逐块打印响应
        if use_stream:
            for chunk in response:
                print(chunk)
        # 否则，直接打印完整响应
        else:
            print(response)
    # 如果没有响应，则打印错误信息
    else:
        print("Error:", response.status_code)


# 定义一个简单聊天函数，接受是否使用流式响应的参数
def simple_chat(use_stream=False):
    # 创建一个包含系统消息和用户消息的列表
        messages = [
            {
                # 指定角色为系统
                "role": "system",
                # 系统消息内容，要求输出时带有“喵喵喵”
                "content": "请在你输出的时候都带上“喵喵喵”三个字，放在开头。",
            },
            {
                # 指定角色为用户
                "role": "user",
                # 用户消息内容，询问身份
                "content": "你是谁"
            }
        ]
        # 调用客户端的聊天完成方法，生成响应
        response = client.chat.completions.create(
            # 指定使用的模型
            model="glm-4",
            # 传入消息列表
            messages=messages,
            # 是否使用流式响应
            stream=use_stream,
            # 设置最大令牌数
            max_tokens=256,
            # 设置温度参数，影响生成的随机性
            temperature=0.4,
            # 设置存在惩罚，控制重复性
            presence_penalty=1.2,
            # 设置采样的累积概率
            top_p=0.8,
        )
        # 检查响应是否存在
        if response:
            # 如果使用流式响应
            if use_stream:
                # 遍历响应的每个块并打印
                for chunk in response:
                    print(chunk)
            # 如果不是流式响应
            else:
                # 打印完整响应
                print(response)
        # 如果响应不存在，打印错误信息
        else:
            print("Error:", response.status_code)
# 创建聊天补全函数，接收消息列表和是否使用流的标志
def create_chat_completion(messages, use_stream=False):
    # 调用客户端的聊天补全方法，创建模型响应
    response = client.chat.completions.create(
        model="glm-4v",  # 指定使用的模型
        messages=messages,  # 传入消息列表
        stream=use_stream,  # 指定是否使用流式响应
        max_tokens=256,  # 设置生成的最大令牌数
        temperature=0.4,  # 设置生成的温度，影响随机性
        presence_penalty=1.2,  # 增加新主题的惩罚系数
        top_p=0.8,  # 设置 nucleus 采样的阈值
    )
    # 如果有响应，进行处理
    if response:
        # 如果使用流式响应，逐块打印响应
        if use_stream:
            for chunk in response:
                print(chunk)  # 打印每个数据块
        # 否则打印完整的响应
        else:
            print(response)
    # 如果没有响应，打印错误信息
    else:
        print("Error:", response.status_code)  # 输出错误状态码


# 编码图像文件为 base64 字符串的函数
def encode_image(image_path):
    """
    将图像文件编码为 base64 字符串。
    参数：
        image_path (str): 图像文件的路径。

    此函数打开指定的图像文件，读取其内容，并将其编码为 base64 字符串。
    base64 编码用于通过 HTTP 作为文本发送图像。
    """

    # 以二进制读取模式打开图像文件
    with open(image_path, "rb") as image_file:
        # 读取文件内容并返回 base64 编码的字符串
        return base64.b64encode(image_file.read()).decode("utf-8")


# 简单图像聊天的函数，涉及到图像
def glm4v_simple_image_chat(use_stream=False, img_path=None):
    """
    促进简单的图像聊天互动。

    参数：
        use_stream (bool): 指定是否使用流式聊天响应。
        img_path (str): 要包含在聊天中的图像文件路径。

    此函数编码指定的图像，并构建包含该图像的预定义对话。
    然后调用 `create_chat_completion` 生成模型响应。
    对话包括询问图像内容及后续问题。
    """

    # 将图像路径编码为 base64 URL
    img_url = f"data:image/jpeg;base64,{encode_image(img_path)}"
    # 构建消息列表，包括用户和助手的对话内容
    messages = [
        {
            "role": "user",  # 用户角色
            "content": [
                {
                    "type": "text",  # 文本类型消息
                    "text": "What’s in this image?",  # 用户提问
                },
                {
                    "type": "image_url",  # 图像 URL 类型消息
                    "image_url": {
                        "url": img_url  # 图像的 base64 URL
                    },
                },
            ],
        },
        {
            "role": "assistant",  # 助手角色
            "content": "The image displays a wooden boardwalk extending through a vibrant green grassy wetland. The sky is partly cloudy with soft, wispy clouds, indicating nice weather. Vegetation is seen on either side of the boardwalk, and trees are present in the background, suggesting that this area might be a natural reserve or park designed for ecological preservation and outdoor recreation. The boardwalk allows visitors to explore the area without disturbing the natural habitat.",  # 助手的回答
        },
        {
            "role": "user",  # 用户角色
            "content": "Do you think this is a spring or winter photo?"  # 用户的后续问题
        },
    ]
    # 调用创建聊天补全的函数，生成响应
    create_chat_completion(messages=messages, use_stream=use_stream)
    

# 主程序入口
if __name__ == "__main__":
    simple_chat(use_stream=False)  # 调用简单聊天函数
    # function_chat(use_stream=False)  # 注释掉的函数调用
    # glm4v_simple_image_chat(use_stream=False, img_path="demo.jpg")  # 注释掉的图像聊天调用
```