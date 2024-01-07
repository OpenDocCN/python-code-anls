# `Bert-VITS2\oldVersion\V111\text\japanese_bert.py`

```

# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 导入 sys 库
import sys

# 从本地路径加载预训练的 BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# 创建一个空的模型字典
models = dict()

# 定义一个函数，用于获取 BERT 特征
def get_bert_feature(text, word2ph, device=None):
    # 如果运行平台是 macOS，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则默认使用 CUDA
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载对应设备的预训练模型，并将其移动到指定设备上
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    # 禁止梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出的隐藏状态，并将其拼接成一个张量
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的词语数量与对应的音素数量相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将输入的词语到音素的映射关系保存到 word2phone 变量中
    word2phone = word2ph
    # 创建一个空列表，用于存储每个音素对应的特征
    phone_level_feature = []
    # 遍历每个词语对应的音素
    for i in range(len(word2phone)):
        # 将对应词语的特征重复 word2phone[i] 次，并添加到特征列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将所有音素的特征拼接成一个张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回音素级别的特征，转置后的结果
    return phone_level_feature.T

```