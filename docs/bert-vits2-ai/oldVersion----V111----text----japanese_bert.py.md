# `Bert-VITS2\oldVersion\V111\text\japanese_bert.py`

```py
# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 导入 sys 库
import sys

# 使用指定路径下的模型创建 tokenizer 对象
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# 创建空的模型字典
models = dict()

# 定义函数，获取 BERT 特征
def get_bert_feature(text, word2ph, device=None):
    # 如果运行平台是 macOS，并且支持多进程，且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则根据指定路径创建模型，并将模型移动到指定设备
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    # 禁止梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出，包括隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 选择倒数第三层和倒数第二层的隐藏状态拼接成特征向量
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的词语数量与音素数量相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将输入的 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表，用于存储每个音素对应的特征向量
    phone_level_feature = []
    # 遍历每个词语对应的音素数量
    for i in range(len(word2phone)):
        # 将对应词语的特征向量重复 word2phone[i] 次，并添加到列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将列表中的特征向量拼接成一个张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的音素级特征
    return phone_level_feature.T
```