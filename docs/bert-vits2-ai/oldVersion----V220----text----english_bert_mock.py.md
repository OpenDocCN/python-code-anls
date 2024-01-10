# `Bert-VITS2\oldVersion\V220\text\english_bert_mock.py`

```
# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 DebertaV2Model 和 DebertaV2Tokenizer 类
from transformers import DebertaV2Model, DebertaV2Tokenizer

# 从 config 模块中导入 config 变量
from config import config

# 设置本地路径
LOCAL_PATH = "./bert/deberta-v3-large"

# 使用 DebertaV2Tokenizer 类从预训练模型中加载 tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

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
    # 如果操作系统是 macOS，并且支持 MPS，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典的键中
    if device not in models.keys():
        # 将预训练模型加载到指定设备上，并存储到模型字典中
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出的隐藏状态，并拼接成一个张量
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本
        if style_text:
            # 使用 tokenizer 对样式文本进行编码，返回 PyTorch 张量
            style_inputs = tokenizer(style_text, return_tensors="pt")
            # 将样式输入张量移动到指定设备上
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            # 获取样式文本的隐藏状态，并拼接成一个张量
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            # 计算样式文本的均值
            style_res_mean = style_res.mean(0)
    # 断言 word2ph 的长度与 res 的形状的第一个维度相等
    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表 phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 如果存在样式文本
        if style_text:
            # 重复特征并加权合并
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            # 重复特征
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复的特征添加到 phone_level_feature 列表中
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的张量拼接成一个张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回 phone_level_feature 的转置
    return phone_level_feature.T
```