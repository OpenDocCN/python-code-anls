# `.\chatglm4-finetune\basic_demo\trans_web_vision_demo.py`

```
"""
# 该脚本创建一个 Gradio 演示，使用 glm-4v-9b 模型作为 Transformers 后端，允许用户通过 Gradio Web UI 与模型互动。

# 使用方法：
# - 运行脚本以启动 Gradio 服务器。
# - 通过 Web UI 与模型互动。

# 需求：
# - Gradio 包
#   - 输入 `pip install gradio` 来安装 Gradio。
"""

# 导入必要的库
import os  # 用于处理操作系统功能，如环境变量
import torch  # PyTorch 库，用于深度学习
import gradio as gr  # Gradio 库，用于创建 Web UI
from threading import Thread  # 用于多线程处理
from transformers import (  # 从 transformers 库导入模型和 tokenizer
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer, AutoModel, BitsAndBytesConfig
)
from PIL import Image  # 用于处理图像
import requests  # 用于发送 HTTP 请求
from io import BytesIO  # 用于在内存中处理字节流

# 从环境变量获取模型路径，默认值为 'THUDM/glm-4v-9b'
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4v-9b')

# 加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,  # 使用指定的模型路径
    trust_remote_code=True,  # 信任远程代码
    encode_special_tokens=True  # 编码特殊标记
)

# 加载预训练的模型
model = AutoModel.from_pretrained(
    MODEL_PATH,  # 使用指定的模型路径
    trust_remote_code=True,  # 信任远程代码
    device_map="auto",  # 自动选择设备
    torch_dtype=torch.bfloat16  # 指定模型使用的浮点精度
).eval()  # 将模型设置为评估模式

# 定义停止条件类
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = model.config.eos_token_id  # 获取结束标记 ID
        for stop_id in stop_ids:  # 遍历所有结束标记 ID
            if input_ids[0][-1] == stop_id:  # 如果当前输入的最后一个 ID 是结束标记
                return True  # 返回 True 以停止生成
        return False  # 否则返回 False

# 定义获取图像的函数
def get_image(image_path=None, image_url=None):
    if image_path:  # 如果提供了本地图像路径
        return Image.open(image_path).convert("RGB")  # 打开图像并转换为 RGB 格式
    elif image_url:  # 如果提供了图像 URL
        response = requests.get(image_url)  # 发送 GET 请求获取图像
        return Image.open(BytesIO(response.content)).convert("RGB")  # 打开图像并转换为 RGB 格式
    return None  # 如果没有提供图像，则返回 None

# 定义聊天机器人的主函数
def chatbot(image_path=None, image_url=None, assistant_prompt=""):
    image = get_image(image_path, image_url)  # 获取图像

    # 准备消息列表，包括助手的提示和用户的输入
    messages = [
        {"role": "assistant", "content": assistant_prompt},  # 助手消息
        {"role": "user", "content": "", "image": image}  # 用户消息
    ]

    # 使用 tokenizer 将消息转换为模型输入格式
    model_inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,  # 添加生成提示
        tokenize=True,  # 启用标记化
        return_tensors="pt",  # 返回 PyTorch 张量
        return_dict=True  # 返回字典格式
    ).to(next(model.parameters()).device)  # 移动到模型所在设备

    # 创建文本流迭代器以进行实时生成
    streamer = TextIteratorStreamer(
        tokenizer=tokenizer,  # 使用的 tokenizer
        timeout=60,  # 超时时间
        skip_prompt=True,  # 跳过提示
        skip_special_tokens=True  # 跳过特殊标记
    )

    # 设置生成的参数
    generate_kwargs = {
        **model_inputs,  # 包含模型输入
        "streamer": streamer,  # 使用的文本流迭代器
        "max_new_tokens": 1024,  # 生成的最大新标记数量
        "do_sample": True,  # 启用采样
        "top_p": 0.8,  # 使用 Top-p 采样
        "temperature": 0.6,  # 控制输出的随机性
        "stopping_criteria": StoppingCriteriaList([StopOnTokens()]),  # 设置停止条件
        "repetition_penalty": 1.2,  # 重复惩罚
        "eos_token_id": [151329, 151336, 151338],  # 结束标记 ID 列表
    }

    # 启动生成线程
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # 启动线程

    response = ""  # 初始化响应字符串
    for new_token in streamer:  # 遍历生成的每个新标记
        if new_token:  # 如果有新标记
            response += new_token  # 将新标记添加到响应中

    return image, response.strip()  # 返回图像和去除空白的响应

# 使用 Gradio 创建演示界面
with gr.Blocks() as demo:
    demo.title = "GLM-4V-9B Image Recognition Demo"  # 设置演示的标题
    demo.description = """  # 设置演示的描述
    This demo uses the GLM-4V-9B model to got image infomation.
    """
    # 创建一个水平排列的容器
        with gr.Row():
            # 创建一个垂直排列的容器
            with gr.Column():
                # 创建一个文件输入框，供用户上传高优先级的图像
                image_path_input = gr.File(label="Upload Image (High-Priority)", type="filepath")
                # 创建一个文本框，供用户输入低优先级的图像 URL
                image_url_input = gr.Textbox(label="Image URL (Low-Priority)")
                # 创建一个文本框，供用户输入助手提示，可以修改
                assistant_prompt_input = gr.Textbox(label="Assistant Prompt (You Can Change It)", value="这是什么？")
                # 创建一个提交按钮
                submit_button = gr.Button("Submit")
            # 另一个垂直排列的容器
            with gr.Column():
                # 创建一个文本框，用于显示 GLM-4V-9B 模型的响应
                chatbot_output = gr.Textbox(label="GLM-4V-9B Model Response")
                # 创建一个图像组件，用于显示图像预览
                image_output = gr.Image(label="Image Preview")
    
        # 为提交按钮设置点击事件，调用 chatbot 函数
        submit_button.click(chatbot,
                            # 定义输入为三个组件：上传的图像路径、图像 URL 和助手提示
                            inputs=[image_path_input, image_url_input, assistant_prompt_input],
                            # 定义输出为两个组件：图像输出和聊天机器人的输出
                            outputs=[image_output, chatbot_output])
# 启动 demo 应用，指定服务器地址和端口
demo.launch(server_name="127.0.0.1", server_port=8911, inbrowser=True, share=False)
```