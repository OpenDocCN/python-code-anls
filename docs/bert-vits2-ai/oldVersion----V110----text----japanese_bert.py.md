# `Bert-VITS2\oldVersion\V110\text\japanese_bert.py`

```py
# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 导入 sys 库
import sys

# 从本地路径加载预训练的 tokenizer
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# 定义函数 get_bert_feature，接受文本、word2ph 和设备参数
def get_bert_feature(text, word2ph, device=None):
    # 如果运行平台是 macOS，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备参数未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 从本地路径加载预训练的模型，并将其移动到指定设备
    model = AutoModelForMaskedLM.from_pretrained("./bert/bert-base-japanese-v3").to(
        device
    )
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，并返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用模型对输入进行推理，输出隐藏状态
        res = model(**inputs, output_hidden_states=True)
        # 从隐藏状态中取出倒数第三层的结果，并将其移动到 CPU
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的 input_ids 的最后一个维度长度等于 word2ph 的长度
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 初始化 phone_level_feature 列表
    phone_level_feature = []
    # 遍历 word2phone 的长度
    for i in range(len(word2phone)):
        # 将 res[i] 重复 word2phone[i] 次，并添加到 phone_level_feature 列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的张量拼接成一个张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回 phone_level_feature 的转置
    return phone_level_feature.T
```