# `d:/src/tocomm/Bert-VITS2\oldVersion\V220\text\japanese_bert.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建深度学习模型
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量
from text.japanese import text2sep_kata  # 从text.japanese模块中导入text2sep_kata函数

LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"  # 设置本地路径变量

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用AutoTokenizer类从预训练模型路径中加载tokenizer

models = dict()  # 创建一个空字典变量models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,  # 定义一个名为get_bert_feature的函数，接受text、word2ph、device和style_text等参数
    style_weight=0.7,  # 设置样式权重为0.7

):
    text = "".join(text2sep_kata(text)[0])  # 将文本转换为特定格式的文本
    if (
        sys.platform == "darwin"  # 检查操作系统是否为 macOS
        and torch.backends.mps.is_available()  # 检查是否支持多进程
        and device == "cpu"  # 检查设备是否为 CPU
    ):
        device = "mps"  # 如果满足条件，将设备设置为 "mps"
    if not device:  # 如果设备未指定
        device = "cuda"  # 将设备设置为 "cuda"
    if device not in models.keys():  # 如果设备不在模型字典中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型中加载模型并将其移动到指定设备
    with torch.no_grad():  # 禁用梯度计算
        inputs = tokenizer(text, return_tensors="pt")  # 使用分词器对文本进行处理并返回张量
        for i in inputs:  # 遍历输入
            inputs[i] = inputs[i].to(device)  # 将输入移动到指定设备
        res = models[device](**inputs, output_hidden_states=True)  # 使用模型进行推理并获取隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到 CPU
        if style_text:  # 如果有样式文本
            style_inputs = tokenizer(style_text, return_tensors="pt")  # 使用tokenizer将style_text转换为模型可接受的输入格式，并返回张量
            for i in style_inputs:  # 遍历style_inputs中的张量
                style_inputs[i] = style_inputs[i].to(device)  # 将style_inputs中的张量移动到指定的设备上
            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用指定设备上的模型对style_inputs进行推理，输出隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态的最后三层拼接起来，并移动到CPU上
            style_res_mean = style_res.mean(0)  # 计算隐藏状态的均值

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化phone_level_feature为空列表
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        if style_text:  # 如果style_text存在
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)  # 将res[i]重复word2phone[i]次，并乘以(1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight  # 将style_res_mean重复word2phone[i]次，并乘以style_weight
            )
        else:  # 如果style_text不存在
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中
    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 使用torch.cat函数将phone_level_feature列表中的张量按照dim=0的维度拼接起来，得到一个新的张量

    return phone_level_feature.T  # 返回phone_level_feature的转置，即将其行列互换后的张量
```