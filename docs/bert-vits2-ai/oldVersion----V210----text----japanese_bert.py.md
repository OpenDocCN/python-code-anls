# `Bert-VITS2\oldVersion\V210\text\japanese_bert.py`

```py
# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 AutoModelForMaskedLM 和 AutoTokenizer 类
from transformers import AutoModelForMaskedLM, AutoTokenizer
# 从 config 模块中导入 config 变量
from config import config
# 从当前目录下的 japanese 模块中导入 text2sep_kata 函数
from .japanese import text2sep_kata

# 设置本地路径常量
LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"
# 使用 AutoTokenizer 类从预训练模型路径中加载分词器
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空字典 models
models = dict()

# 定义函数 get_bert_feature，接受文本、word2ph、设备、风格文本和风格权重作为参数
def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    # 将文本转换为片假名，并连接成字符串
    text = "".join(text2sep_kata(text)[0])
    # 如果存在风格文本，将其转换为片假名，并连接成字符串
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])
    # 如果操作系统为 macOS，并且支持多进程并行计算，并且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在 models 字典的键中
    if device not in models.keys():
        # 将预训练模型加载到指定设备上，并存储到 models 字典中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用分词器对文本进行编码，并返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出的隐藏状态，并拼接成一个张量
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在风格文本
        if style_text:
            # 使用分词器对风格文本进行编码，并返回 PyTorch 张量
            style_inputs = tokenizer(style_text, return_tensors="pt")
            # 将输入张量移动到指定设备上
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            # 获取模型输出的隐藏状态，并拼接成一个张量
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            # 计算风格文本的平均隐藏状态
            style_res_mean = style_res.mean(0)

    # 断言 word2ph 的长度等于文本长度加上 2
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表 phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 如果存在风格文本
        if style_text:
            # 重复当前隐藏状态，并根据风格权重计算混合特征
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            # 重复当前隐藏状态
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复特征添加到 phone_level_feature 列表中
        phone_level_feature.append(repeat_feature)
    # 将列表中的张量按照指定维度进行拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # 返回拼接后的张量的转置
    return phone_level_feature.T
```