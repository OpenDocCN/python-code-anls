# `Bert-VITS2\text\chinese_bert.py`

```
# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 AutoModelForMaskedLM 和 AutoTokenizer 类
from transformers import AutoModelForMaskedLM, AutoTokenizer
# 从 config 模块中导入 config 变量
from config import config

# 设置本地路径常量
LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"

# 使用 AutoTokenizer 类从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

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
    # 如果操作系统是 macOS，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
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
        # 从本地路径加载预训练模型，并将其移动到指定设备
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用分词器对文本进行编码，并返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用模型进行推理，获取隐藏状态的输出
        res = models[device](**inputs, output_hidden_states=True)
        # 选择倒数第三层到倒数第二层的隐藏状态，并进行拼接操作
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本
        if style_text:
            # 使用分词器对样式文本进行编码，并返回 PyTorch 张量
            style_inputs = tokenizer(style_text, return_tensors="pt")
            # 将样式输入张量移动到指定设备
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            # 使用模型进行推理，获取隐藏状态的输出
            style_res = models[device](**style_inputs, output_hidden_states=True)
            # 选择倒数第三层到倒数第二层的隐藏状态，并进行拼接操作
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            # 计算样式隐藏状态的均值
            style_res_mean = style_res.mean(0)
    # 断言 word2ph 的长度等于文本长度加上 2
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表 phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 如果存在样式文本
        if style_text:
            # 重复当前隐藏状态特征，并根据样式权重进行加权平均
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:
            # 重复当前隐藏状态特征
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复特征添加到 phone_level_feature 列表中
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的张量进行拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的 phone_level_feature
    return phone_level_feature.T


# 如果当前脚本为主程序
if __name__ == "__main__":
    # 生成一个38x1024的随机张量，表示12个词，每个词有1024维特征
    word_level_feature = torch.rand(38, 1024)  
    # 每个词对应的音素序列
    word2phone = [
        1, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 1, 2, 2, 2, 2, 2, 1, 1, 2, 2, 1, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 1,
    ]

    # 计算总帧数
    total_frames = sum(word2phone)
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)  # 打印word_level_feature[i]的形状

        # 对每个词的特征重复word2phone[i]次，维度为1
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 沿着指定维度拼接张量列表
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状，torch.Size([36, 1024])
```