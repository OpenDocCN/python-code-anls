# `d:/src/tocomm/Bert-VITS2\oldVersion\V200\text\english_bert_mock.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建深度学习模型
from transformers import DebertaV2Model, DebertaV2Tokenizer  # 从transformers模块中导入DebertaV2Model和DebertaV2Tokenizer类

from config import config  # 从config模块中导入config变量

LOCAL_PATH = "./bert/deberta-v3-large"  # 设置本地路径变量

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 使用本地路径初始化DebertaV2Tokenizer对象

models = dict()  # 创建一个空的字典对象

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (  # 如果以下条件满足：
        sys.platform == "darwin"  # 当前操作系统为darwin
        and torch.backends.mps.is_available()  # 并且torch后端支持MPS（Multi-Process Service）
        and device == "cpu"  # 并且设备为CPU
    ):
        device = "mps"  # 如果device为空，则将device设置为"mps"
    if not device:
        device = "cuda"  # 如果device仍为空，则将device设置为"cuda"
    if device not in models.keys():  # 如果device不在models字典的键中
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)  # 使用预训练模型创建device对应的模型并将其移动到device上
    with torch.no_grad():  # 使用torch的no_grad上下文管理器，表示在此范围内不进行梯度计算
        inputs = tokenizer(text, return_tensors="pt")  # 使用tokenizer对文本进行处理，返回输入的张量
        for i in inputs:  # 遍历inputs中的元素
            inputs[i] = inputs[i].to(device)  # 将inputs中的张量移动到device上
        res = models[device](**inputs, output_hidden_states=True)  # 使用device对应的模型进行推理，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到CPU上
    # assert len(word2ph) == len(text)+2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化phone_level_feature列表
    for i in range(len(word2phone)):  # 遍历word2phone的长度
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 将res[i]重复word2phone[i]次
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature拼接在一起，沿着0维度
# 返回变量 phone_level_feature 的转置结果
```