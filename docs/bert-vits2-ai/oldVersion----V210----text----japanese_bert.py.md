# `d:/src/tocomm/Bert-VITS2\oldVersion\V210\text\japanese_bert.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建深度学习模型
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量
from .japanese import text2sep_kata  # 从当前目录下的japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"  # 设置本地路径变量为"./bert/deberta-v2-large-japanese-char-wwm"

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径LOCAL_PATH中加载分词器

models = dict()  # 创建一个空字典变量models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,  # 定义一个名为get_bert_feature的函数，接受text、word2ph、device和style_text等参数
    style_weight=0.7,  # 设置默认的 style_weight 为 0.7

):
    text = "".join(text2sep_kata(text)[0])  # 将文本转换为片假名，并拼接成字符串
    if style_text:  # 如果有 style_text
        style_text = "".join(text2sep_kata(style_text)[0])  # 将 style_text 转换为片假名，并拼接成字符串
    if (
        sys.platform == "darwin"  # 如果操作系统是 macOS
        and torch.backends.mps.is_available()  # 并且支持 MPS
        and device == "cpu"  # 并且设备是 CPU
    ):
        device = "mps"  # 将设备设置为 "mps"
    if not device:  # 如果设备未指定
        device = "cuda"  # 将设备设置为 "cuda"
    if device not in models.keys():  # 如果设备不在 models 字典的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 根据设备从预训练模型中加载模型，并将其移动到指定设备
    with torch.no_grad():  # 使用 torch.no_grad() 上下文管理器，禁用梯度计算
        inputs = tokenizer(text, return_tensors="pt")  # 使用 tokenizer 对文本进行处理，返回 PyTorch 张量
        for i in inputs:  # 遍历 inputs
            inputs[i] = inputs[i].to(device)  # 将 inputs 中的张量移动到指定设备
        res = models[device](**inputs, output_hidden_states=True)  # 使用指定设备的模型对输入进行推理，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将模型输出的隐藏状态拼接起来，并转移到 CPU 上
        if style_text:  # 如果有样式文本
            style_inputs = tokenizer(style_text, return_tensors="pt")  # 使用分词器对样式文本进行处理，返回张量
            for i in style_inputs:  # 遍历样式输入
                style_inputs[i] = style_inputs[i].to(device)  # 将样式输入转移到指定设备上
            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用模型对样式输入进行处理，返回隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将样式输出的隐藏状态拼接起来，并转移到 CPU 上
            style_res_mean = style_res.mean(0)  # 计算样式输出的平均值

    assert len(word2ph) == len(text) + 2  # 断言确保 word2ph 的长度等于 text 的长度加 2
    word2phone = word2ph  # 将 word2ph 赋值给 word2phone
    phone_level_feature = []  # 初始化 phone_level_feature 列表
    for i in range(len(word2phone)):  # 遍历 word2phone 的长度
        if style_text:  # 如果有样式文本
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)  # 重复 res[i]，并根据样式权重进行加权
                + style_res_mean.repeat(word2phone[i], 1) * style_weight  # 重复样式输出的平均值，并根据样式权重进行加权
            )
        else:  # 如果没有样式文本
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 重复 res[i]

    phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 使用torch.cat函数将phone_level_feature列表中的tensor连接起来，dim=0表示按行连接

    return phone_level_feature.T  # 返回phone_level_feature的转置
```