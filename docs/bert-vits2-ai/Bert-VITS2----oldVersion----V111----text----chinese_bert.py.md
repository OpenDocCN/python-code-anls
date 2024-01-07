# `Bert-VITS2\oldVersion\V111\text\chinese_bert.py`

```

# 导入所需的库
import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 从本地预训练模型加载分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

# 创建模型字典
models = dict()

# 定义函数，获取BERT特征
def get_bert_feature(text, word2ph, device=None):
    # 检查系统平台和设备类型，设置默认设备为cuda
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载模型并放到对应设备上
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/chinese-roberta-wwm-ext-large"
        ).to(device)
    # 使用torch.no_grad()上下文管理器，避免梯度计算
    with torch.no_grad():
        # 使用分词器对文本进行编码，返回PyTorch张量
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出的隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层和倒数第二层的隐藏状态拼接成特征
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言，确保word2ph的长度等于文本长度加2
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    # 遍历word2phone，重复对应的特征并添加到phone_level_feature中
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 拼接phone_level_feature，得到每个音素的特征
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

# 主函数
if __name__ == "__main__":
    import torch

    # 随机生成word_level_feature和word2phone
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

    # 拼接phone_level_feature，得到每个音素的特征
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])

```