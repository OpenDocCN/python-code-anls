# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\japanese_bert.py`

```
import torch  # 导入PyTorch库
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM类
import sys  # 导入sys库

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 使用预训练的BERT模型tokenizer

models = dict()  # 创建一个空的字典用于存储模型

def get_bert_feature(text, word2ph, device=None):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (  # 如果条件判断开始
        sys.platform == "darwin"  # 当操作系统为MacOS
        and torch.backends.mps.is_available()  # 并且PyTorch的多进程服务可用
        and device == "cpu"  # 并且设备为CPU
    ):  # 条件判断结束
        device = "mps"  # 将设备设置为"mps"
    if not device:  # 如果设备为空
        device = "cuda"  # 将设备设置为"cuda"
    if device not in models.keys():  # 如果设备不在models字典的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(  # 使用预训练的BERT模型
    "./bert/bert-base-japanese-v3"  # 设置模型路径
).to(device)  # 将模型移动到指定设备上

with torch.no_grad():  # 在此范围内不进行梯度计算
    inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对文本进行处理，返回PyTorch张量
    for i in inputs:  # 遍历inputs中的键
        inputs[i] = inputs[i].to(device)  # 将inputs中的值移动到指定设备上
    res = models[device](**inputs, output_hidden_states=True)  # 使用模型对输入进行推理，输出隐藏状态
    res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将模型输出的隐藏状态进行拼接和转移至CPU

assert inputs["input_ids"].shape[-1] == len(word2ph)  # 断言输入的input_ids的最后一个维度长度与word2ph的长度相等
word2phone = word2ph  # 将word2ph赋值给word2phone
phone_level_feature = []  # 初始化一个空列表用于存储特征
for i in range(len(word2phone)):  # 遍历word2phone的长度
    repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次，沿着第二个维度
    phone_level_feature.append(repeat_feature)  # 将重复的特征添加到phone_level_feature中

phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature中的特征进行拼接，沿着第一个维度

return phone_level_feature.T  # 返回phone_level_feature的转置
```