# `Bert-VITS2\oldVersion\V110\text\japanese_bert.py`

```

# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 导入 sys 库
import sys

# 从本地路径加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# 定义函数 get_bert_feature，用于获取 BERT 特征
def get_bert_feature(text, word2ph, device=None):
    # 检查系统平台和设备是否支持多进程并且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 从本地路径加载预训练的 BERT 模型，并将其移动到指定设备上
    model = AutoModelForMaskedLM.from_pretrained("./bert/bert-base-japanese-v3").to(
        device
    )
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，并返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取 BERT 模型的输出隐藏状态
        res = model(**inputs, output_hidden_states=True)
        # 选择倒数第三层到倒数第二层的隐藏状态，并将其转移到 CPU 上
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的 token 数与 word2ph 的长度相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 初始化 phone_level_feature 列表
    phone_level_feature = []
    # 遍历 word2phone，重复对应位置的特征，并添加到 phone_level_feature 中
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 拼接成张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回 phone_level_feature 的转置
    return phone_level_feature.T

```