# `.\chatglm4-finetune\basic_demo\trans_web_demo.py`

```
# 该脚本使用 Gradio 创建 GLM-4-9B 模型的交互式网络演示
"""
This script creates an interactive web demo for the GLM-4-9B model using Gradio,
a Python library for building quick and easy UI components for machine learning models.
It's designed to showcase the capabilities of the GLM-4-9B model in a user-friendly interface,
allowing users to interact with the model through a chat-like interface.
"""
# 导入操作系统模块
import os
# 从 pathlib 导入 Path 类以处理路径
from pathlib import Path
# 导入 Thread 类以支持多线程
from threading import Thread
# 导入 Union 类型以支持类型注解
from typing import Union

# 导入 Gradio 库以构建用户界面
import gradio as gr
# 导入 PyTorch 库以支持深度学习模型
import torch
# 从 peft 导入相关模型类
from peft import AutoPeftModelForCausalLM, PeftModelForCausalLM
# 从 transformers 导入所需的类
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer
)

# 定义模型类型的别名
ModelType = Union[PreTrainedModel, PeftModelForCausalLM]
# 定义分词器类型的别名
TokenizerType = Union[PreTrainedTokenizer, PreTrainedTokenizerFast]

# 从环境变量获取模型路径，若不存在则使用默认路径
MODEL_PATH = os.environ.get('MODEL_PATH', 'THUDM/glm-4-9b-chat')
# 从环境变量获取分词器路径，若不存在则使用模型路径
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", MODEL_PATH)


# 定义解析路径的辅助函数
def _resolve_path(path: Union[str, Path]) -> Path:
    # 扩展用户路径并解析为绝对路径
    return Path(path).expanduser().resolve()


# 定义加载模型和分词器的函数
def load_model_and_tokenizer(
        model_dir: Union[str, Path], trust_remote_code: bool = True
) -> tuple[ModelType, TokenizerType]:
    # 解析模型目录路径
    model_dir = _resolve_path(model_dir)
    # 检查是否存在适配器配置文件
    if (model_dir / 'adapter_config.json').exists():
        # 从预训练模型中加载适配器模型
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        # 获取基础模型名称
        tokenizer_dir = model.peft_config['default'].base_model_name_or_path
    else:
        # 从预训练模型中加载普通模型
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=trust_remote_code, device_map='auto'
        )
        # 将模型目录设为分词器目录
        tokenizer_dir = model_dir
    # 从预训练分词器中加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir, trust_remote_code=trust_remote_code, use_fast=False
    )
    # 返回模型和分词器的元组
    return model, tokenizer


# 加载模型和分词器
model, tokenizer = load_model_and_tokenizer(MODEL_PATH, trust_remote_code=True)


# 定义基于特定标记停止的类
class StopOnTokens(StoppingCriteria):
    # 定义调用方法以检查是否应停止
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # 获取结束标记 ID
        stop_ids = model.config.eos_token_id
        # 检查输入 ID 的最后一个是否为停止 ID
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        # 如果没有停止条件，返回 False
        return False


# 定义预测函数
def predict(history, prompt, max_length, top_p, temperature):
    # 创建停止条件实例
    stop = StopOnTokens()
    # 初始化消息列表
    messages = []
    # 如果提示存在，将其添加到消息中
    if prompt:
        messages.append({"role": "system", "content": prompt})
    # 遍历历史消息
    for idx, (user_msg, model_msg) in enumerate(history):
        # 如果提示存在且是第一条消息，则跳过
        if prompt and idx == 0:
            continue
        # 如果是最后一条消息且模型消息为空
        if idx == len(history) - 1 and not model_msg:
            messages.append({"role": "user", "content": user_msg})
            break
        # 如果用户消息存在，则添加到消息中
        if user_msg:
            messages.append({"role": "user", "content": user_msg})
        # 如果模型消息存在，则添加到消息中
        if model_msg:
            messages.append({"role": "assistant", "content": model_msg})
    # 使用 tokenizer 将消息应用于聊天模板，生成模型输入
        model_inputs = tokenizer.apply_chat_template(messages,
                                                     add_generation_prompt=True,  # 添加生成提示
                                                     tokenize=True,  # 启用标记化
                                                     return_tensors="pt").to(next(model.parameters()).device)  # 转移到模型设备
    
        # 创建一个文本迭代器流，用于实时生成文本
        streamer = TextIteratorStreamer(tokenizer, timeout=60, skip_prompt=True, skip_special_tokens=True)
    
        # 定义生成所需的参数
        generate_kwargs = {
            "input_ids": model_inputs,  # 模型输入的标记ID
            "streamer": streamer,  # 指定文本流
            "max_new_tokens": max_length,  # 生成的最大新标记数量
            "do_sample": True,  # 启用采样以生成多样化输出
            "top_p": top_p,  # 限制采样的前p概率
            "temperature": temperature,  # 控制生成文本的随机性
            "stopping_criteria": StoppingCriteriaList([stop]),  # 设置停止标准
            "repetition_penalty": 1.2,  # 防止重复生成
            "eos_token_id": model.config.eos_token_id,  # 结束标记的ID
        }
    
        # 创建一个线程用于执行生成
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()  # 启动线程
    
        # 迭代从流中接收的新标记
        for new_token in streamer:
            if new_token:  # 如果新标记存在
                history[-1][1] += new_token  # 将新标记添加到历史记录的最后一个条目
            yield history  # 生成历史记录的当前状态
# 使用 Gradio 创建一个聊天应用的块
with gr.Blocks() as demo:
    # 添加 HTML 标题，居中显示
    gr.HTML("""<h1 align="center">GLM-4-9B Gradio Simple Chat Demo</h1>""")
    # 初始化聊天机器人对象
    chatbot = gr.Chatbot()

    # 创建一个行布局
    with gr.Row():
        # 第一列，宽度比例为 3
        with gr.Column(scale=3):
            # 嵌套的列，宽度比例为 12
            with gr.Column(scale=12):
                # 用户输入框，隐藏标签，提示为 "Input..."
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            # 嵌套的列，最小宽度为 32，宽度比例为 1
            with gr.Column(min_width=32, scale=1):
                # 提交按钮，标签为 "Submit"
                submitBtn = gr.Button("Submit")
        # 第二列，宽度比例为 1
        with gr.Column(scale=1):
            # 提示输入框，隐藏标签，提示为 "Prompt"
            prompt_input = gr.Textbox(show_label=False, placeholder="Prompt", lines=10, container=False)
            # 设置提示按钮，标签为 "Set Prompt"
            pBtn = gr.Button("Set Prompt")
        # 第三列，宽度比例为 1
        with gr.Column(scale=1):
            # 清除历史记录按钮，标签为 "Clear History"
            emptyBtn = gr.Button("Clear History")
            # 最大长度滑动条，范围为 0 到 32768，初始值为 8192
            max_length = gr.Slider(0, 32768, value=8192, step=1.0, label="Maximum length", interactive=True)
            # Top P 滑动条，范围为 0 到 1，初始值为 0.8
            top_p = gr.Slider(0, 1, value=0.8, step=0.01, label="Top P", interactive=True)
            # 温度滑动条，范围为 0.01 到 1，初始值为 0.6
            temperature = gr.Slider(0.01, 1, value=0.6, step=0.01, label="Temperature", interactive=True)

    # 用户输入处理函数，返回空字符串和更新的历史记录
    def user(query, history):
        return "", history + [[query, ""]]

    # 设置提示处理函数，返回包含提示文本和成功消息的列表
    def set_prompt(prompt_text):
        return [[prompt_text, "成功设置prompt"]]

    # 点击设置提示按钮时，调用 set_prompt 函数，输入为 prompt_input，输出为 chatbot
    pBtn.click(set_prompt, inputs=[prompt_input], outputs=chatbot)

    # 点击提交按钮时，调用 user 函数，输入为 user_input 和 chatbot，输出为更新后的 user_input 和 chatbot，且不排队
    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        # 在用户提交后调用 predict 函数，输入为 chatbot、prompt_input、max_length、top_p 和 temperature，输出为 chatbot
        predict, [chatbot, prompt_input, max_length, top_p, temperature], chatbot
    )
    # 点击清除历史记录按钮时，调用匿名函数返回 None，更新 chatbot 和 prompt_input，且不排队
    emptyBtn.click(lambda: (None, None), None, [chatbot, prompt_input], queue=False)

# 启用队列处理
demo.queue()
# 启动 Gradio 应用，设置服务器名称和端口，自动在浏览器中打开并共享
demo.launch(server_name="127.0.0.1", server_port=8000, inbrowser=True, share=True)
```