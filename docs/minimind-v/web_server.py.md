# `.\minimind-v\web_server.py`

```py
import gradio as gr  # 导入 Gradio 库，用于创建用户界面
import os  # 导入 os 库，用于与操作系统进行交互（如文件路径处理）
import random  # 导入 random 库，用于生成随机数
import time  # 导入 time 库，用于处理时间相关的功能

import numpy as np  # 导入 numpy 库，用于高效的数值计算
import torch  # 导入 PyTorch 库，用于深度学习模型
import warnings  # 导入 warnings 库，用于管理警告信息

from PIL import Image  # 从 PIL 库导入 Image，用于图像处理
from transformers import AutoTokenizer, AutoModelForCausalLM  # 从 transformers 库导入模型和分词器
from model.model import Transformer  # 从自定义模型导入 Transformer 类
from model.LMConfig import LMConfig  # 从自定义配置导入 LMConfig 配置类
from model.vision_utils import get_vision_model, get_img_process, get_img_embedding  # 从 vision_utils 导入图像处理相关函数

warnings.filterwarnings('ignore')  # 忽略所有警告信息


def count_parameters(model):  # 定义一个函数来统计模型的可训练参数数量
    return sum(p.numel() for p in model.parameters() if p.requires_grad)  # 返回模型中所有可训练参数的总数


def init_model(lm_config):  # 初始化模型的函数，传入配置对象
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')  # 从预训练模型加载分词器
    model_from = 2  # 设置模型来源，1代表从权重加载，2代表使用 transformers 库加载

    if model_from == 1:  # 如果从本地权重加载
        moe_path = '_moe' if lm_config.use_moe else ''  # 根据配置决定是否使用moe模型
        ckp = f'./out/{lm_config.dim}{moe_path}_vlm_sft.pth'  # 构造模型权重文件的路径

        model = Transformer(lm_config)  # 使用 Transformer 类初始化模型
        state_dict = torch.load(ckp, map_location=device)  # 加载模型权重

        # 处理不需要的前缀
        unwanted_prefix = '_orig_mod.'  # 定义不需要的前缀
        for k, v in list(state_dict.items()):  # 遍历所有权重项
            if k.startswith(unwanted_prefix):  # 如果权重项名称以不需要的前缀开头
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)  # 去除前缀并更新字典

        for k, v in list(state_dict.items()):  # 遍历所有权重项
            if 'mask' in k:  # 如果权重项名称包含'mask'
                del state_dict[k]  # 删除该项

        # 加载权重到模型中
        model.load_state_dict(state_dict, strict=False)  # 加载权重，strict=False表示忽略缺失项
    else:  # 如果从 transformers 库加载模型
        model = AutoModelForCausalLM.from_pretrained('minimind-v-v1',  # 加载预训练的 causal language model
                                                     trust_remote_code=True)  # 允许加载远程代码

    model = model.to(device)  # 将模型移动到指定的设备（如 GPU）

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')  # 输出模型参数数量，单位为百万和十亿

    (vision_model, preprocess) = get_vision_model()  # 获取视觉模型及其预处理函数
    vision_model = vision_model.to(device)  # 将视觉模型移动到指定设备
    return model, tokenizer, (vision_model, preprocess)  # 返回初始化好的模型、分词器和视觉模型


def chat(prompt, current_image_path):  # 定义一个聊天函数，传入用户输入的提示词和当前图像路径
    # 打开图像并转换为RGB格式
    image = Image.open(current_image_path).convert('RGB')  # 使用 PIL 打开图像，并转换为 RGB 模式
    image_process = get_img_process(image, preprocess).to(vision_model.device)  # 对图像进行预处理并移动到视觉模型所在设备
    # 对图像进行编码
    image_encoder = get_img_embedding(image_process, vision_model).unsqueeze(0)  # 获取图像的嵌入向量并增加一个维度

    prompt = f'{lm_config.image_special_token}\n{prompt}'  # 将图像特定的标记加到用户的提示词前面
    messages = [{"role": "user", "content": prompt}]  # 将提示词封装成消息格式

    # print(messages)  # 可选：打印消息（此行已被注释）
    new_prompt = tokenizer.apply_chat_template(  # 使用分词器将消息应用聊天模板
        messages,
        tokenize=False,  # 不进行分词
        add_generation_prompt=True  # 添加生成提示
    )[-(max_seq_len - 1):]  # 保留最大序列长度的部分

    x = tokenizer(new_prompt).data['input_ids']  # 使用分词器获取新的提示的 token IDs
    x = (torch.tensor(x, dtype=torch.long, device=device)[None, ...])  # 将 token IDs 转换为张量并移到指定设备
    # 使用 torch.no_grad() 上下文管理器，关闭梯度计算
    with torch.no_grad():
        # 生成模型的输出结果，生成的结果存储在 res_y 中
        res_y = model.generate(x, tokenizer.eos_token_id, max_new_tokens=max_seq_len, temperature=temperature,
                               top_k=top_k, stream=stream, image_encoders=image_encoder)
        # 尝试获取下一个生成的结果
        try:
            y = next(res_y)
        except StopIteration:
            # 如果没有生成结果，则打印"No answer"并返回"No answer"
            print("No answer")
            return "No answer"

        # 初始化历史索引为0
        history_idx = 0
        # 当生成的结果不为None时循环
        while y != None:
            # 将生成的结果解码为文本形式
            answer = tokenizer.decode(y[0].tolist())
            # 如果答案不为空且最后一个字符为'�'，则继续生成下一个结果
            if answer and answer[-1] == '�':
                try:
                    y = next(res_y)
                except:
                    break
                continue
            # 如果答案长度为0，则继续生成下一个结果
            if not len(answer):
                try:
                    y = next(res_y)
                except:
                    break
                continue

            # 生成答案并返回，从历史索引开始
            yield answer[history_idx:]
            try:
                y = next(res_y)
            except:
                break
            # 更新历史索引为当前答案的长度
            history_idx = len(answer)
            # 如果不是流式生成，则跳出循环
            if not stream:
                break
# 启动 Gradio 服务的函数，提供服务器名称和端口号
def launch_gradio_server(server_name="0.0.0.0", server_port=7788):
    # 检查脚本是否作为主程序运行
    if __name__ == '__main__':
        # 根据是否有 CUDA 可用，选择计算设备为 'cuda:0' 或 'cpu'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        # 获取语言模型的配置
        lm_config = LMConfig()
        # 获取模型配置中的最大序列长度
        max_seq_len = lm_config.max_seq_len
        # 初始化模型、分词器以及视觉模型与预处理函数
        model, tokenizer, (vision_model, preprocess) = init_model(lm_config)
        # 设置流式传输模式为开启
        stream = True
        # 初始化当前图像路径为空
        current_image_path = ''
        # 设置生成的温度参数
        temperature = 0.5
        # 设置 top-k 参数用于采样
        top_k = 12

        # 启动 Gradio 服务器，监听指定的服务器名称和端口号
        launch_gradio_server(server_name="0.0.0.0", server_port=7788)
```