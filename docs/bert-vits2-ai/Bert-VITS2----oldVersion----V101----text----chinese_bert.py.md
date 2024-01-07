# `Bert-VITS2\oldVersion\V101\text\chinese_bert.py`

```

# 导入所需的库
import torch
import sys
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 检查可用的设备并将其分配给变量device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else (
        "mps"
        if sys.platform == "darwin" and torch.backends.mps.is_available()
        else "cpu"
    )
)

# 从预训练模型中加载tokenizer和model
tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")
model = AutoModelForMaskedLM.from_pretrained("./bert/chinese-roberta-wwm-ext-large").to(
    device
)

# 定义一个函数用于获取BERT特征
def get_bert_feature(text, word2ph):
    # 禁用梯度计算
    with torch.no_grad():
        # 使用tokenizer将文本转换为模型输入
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用模型获取隐藏状态
        res = model(**inputs, output_hidden_states=True)
        # 选择倒数第三层的隐藏状态并转移到CPU上
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言确保word2ph的长度等于文本长度加2
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    # 遍历word2phone并重复对应位置的特征
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将重复后的特征拼接在一起
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T

# 如果作为独立脚本运行，则执行以下代码
if __name__ == "__main__":
    # 创建一个随机的word_level_feature张量
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    # 定义每个词对应的音素数量
    word2phone = [
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)
    print(word2phone)
    phone_level_feature = []
    # 遍历word2phone并重复对应位置的特征
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将重复后的特征拼接在一起
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])

```