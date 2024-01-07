# `Bert-VITS2\oldVersion\V200\text\chinese_bert.py`

```

# 导入必要的库
import sys
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from config import config

# 设置本地路径
LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"

# 从本地路径加载预训练的tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建模型字典
models = dict()

# 获取BERT特征
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 检查设备类型
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载模型到该设备
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 使用torch.no_grad()关闭梯度计算
    with torch.no_grad():
        # 使用tokenizer将文本转换为模型输入
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出的隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言确保word2ph的长度等于文本长度加2
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    # 遍历word2phone，重复res[i] word2phone[i]次，并添加到phone_level_feature中
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将phone_level_feature拼接成tensor
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

# 主函数
if __name__ == "__main__":
    # 创建随机的word_level_feature和word2phone
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    word2phone = [
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将phone_level_feature拼接成tensor
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])

```