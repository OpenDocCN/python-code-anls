# `Bert-VITS2\oldVersion\V110\text\chinese_bert.py`

```

# 导入所需的库
import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 从预训练模型路径加载分词器
tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

# 定义函数，获取BERT特征
def get_bert_feature(text, word2ph, device=None):
    # 检查系统平台和设备是否支持多进程
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果未指定设备，则默认使用cuda
    if not device:
        device = "cuda"
    # 从预训练模型路径加载BERT模型，并将其移动到指定设备
    model = AutoModelForMaskedLM.from_pretrained(
        "./bert/chinese-roberta-wwm-ext-large"
    ).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用分词器对文本进行编码，返回PyTorch张量
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取BERT模型的输出隐藏状态
        res = model(**inputs, output_hidden_states=True)
        # 提取倒数第三层和倒数第二层的隐藏状态，并拼接起来
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言确保word2ph的长度等于文本长度加2
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    # 遍历word2phone列表
    for i in range(len(word2phone)):
        # 将res[i]重复word2phone[i]次，并添加到phone_level_feature列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 拼接phone_level_feature列表中的张量，按行拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的phone_level_feature
    return phone_level_feature.T


if __name__ == "__main__":
    import torch

    # 创建一个随机张量，表示每个词的特征，38行1024列
    word_level_feature = torch.rand(38, 1024)
    # 定义每个词对应的音素数量
    word2phone = [
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)  # 打印张量的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []
    # 遍历word2phone列表
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)  # 打印每个词的特征形状

        # 对每个词的特征重复word2phone[i]次，并添加到phone_level_feature列表中
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 拼接phone_level_feature列表中的张量，按行拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状

```