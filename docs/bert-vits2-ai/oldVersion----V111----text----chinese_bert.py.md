# `Bert-VITS2\oldVersion\V111\text\chinese_bert.py`

```
# 导入 torch 库
import torch
# 导入 sys 库
import sys
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM 类
from transformers import AutoTokenizer, AutoModelForMaskedLM

# 使用指定路径的预训练模型创建 tokenizer 对象
tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")

# 创建空字典 models
models = dict()

# 定义函数 get_bert_feature，接受文本、word2ph 和 device 作为参数
def get_bert_feature(text, word2ph, device=None):
    # 如果系统是 macOS，并且支持多进程，且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在 models 字典的键中
    if device not in models.keys():
        # 根据指定路径的预训练模型创建 AutoModelForMaskedLM 对象，并将其移动到指定设备
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/chinese-roberta-wwm-ext-large"
        ).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用模型进行推理，输出隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 选择倒数第三层到倒数第二层的隐藏状态，并进行拼接和转置
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言 word2ph 的长度等于文本长度加上2
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表 phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 将 res[i] 重复 word2phone[i] 次，并添加到 phone_level_feature 列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的张量进行拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回 phone_level_feature 的转置
    return phone_level_feature.T


# 如果当前脚本为主程序
if __name__ == "__main__":
    # 导入 torch 库
    import torch

    # 创建随机张量 word_level_feature，形状为 (38, 1024)
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
    # 创建 word2phone 列表
    word2phone = [
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    # 打印 word_level_feature 的形状
    print(word_level_feature.shape)
    # 打印 word2phone 列表
    print(word2phone)
    # 创建空列表 phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone 列表的长度范围
    for i in range(len(word2phone)):
        # 打印 word_level_feature[i] 的形状
        print(word_level_feature[i].shape)

        # 对每个词的特征进行重复操作，重复次数为 word2phone[i]，在第一维度上进行重复
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        # 将重复后的特征添加到 phone_level_feature 列表中
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的特征拼接在一起，沿着第一维度拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    # 打印拼接后的 phone_level_feature 的形状
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
```