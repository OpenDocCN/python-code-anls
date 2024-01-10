# `Bert-VITS2\text\japanese_bert.py`

```
# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 AutoModelForMaskedLM 和 AutoTokenizer 类
from transformers import AutoModelForMaskedLM, AutoTokenizer

# 从 config 模块中导入 config 变量
from config import config
# 从 text.japanese 模块中导入 text2sep_kata 函数
from text.japanese import text2sep_kata

# 设置本地路径常量
LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"

# 使用预训练模型的 tokenizer 初始化 tokenizer 对象
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数 get_bert_feature，接受文本、word2ph、设备、样式文本和样式权重作为参数
def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    # 将文本转换为片假名，并连接成字符串
    text = "".join(text2sep_kata(text)[0])
    # 如果存在样式文本，将样式文本转换为片假名，并连接成字符串
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])
    # 如果操作系统为 macOS，且支持多进程并行计算，且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典的键中，则根据本地路径初始化模型，并将其移动到指定设备
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，并返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型的输出，并返回隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 将模型的隐藏状态连接起来，并移动到 CPU
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本
        if style_text:
            # 使用 tokenizer 对样式文本进行编码，并返回 PyTorch 张量
            style_inputs = tokenizer(style_text, return_tensors="pt")
            # 将样式输入数据移动到指定设备
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            # 获取模型的输出，并返回隐藏状态
            style_res = models[device](**style_inputs, output_hidden_states=True)
            # 将模型的隐藏状态连接起来，并移动到 CPU
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            # 计算样式隐藏状态的均值
            style_res_mean = style_res.mean(0)

    # 断言 word2ph 的长度等于文本长度加 2
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空的 phone_level_feature 列表
    phone_level_feature = []
    # 遍历 word2phone 列表的长度范围
    for i in range(len(word2phone)):
        # 如果 style_text 为真
        if style_text:
            # 生成重复特征，使用 res[i] 重复 word2phone[i] 次，乘以 (1 - style_weight)，再加上 style_res_mean 重复 word2phone[i] 次乘以 style_weight
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        # 如果 style_text 为假
        else:
            # 生成重复特征，使用 res[i] 重复 word2phone[i] 次
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将生成的重复特征添加到 phone_level_feature 列表中
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的 tensor 沿着 dim=0 维度拼接起来
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回 phone_level_feature 的转置
    return phone_level_feature.T
```