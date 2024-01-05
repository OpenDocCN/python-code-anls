# `d:/src/tocomm/Bert-VITS2\oldVersion\V110\text\japanese_bert.py`

```
import torch  # 导入torch模块
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers模块中导入AutoTokenizer和AutoModelForMaskedLM类
import sys  # 导入sys模块

tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")  # 使用"./bert/bert-base-japanese-v3"路径下的预训练模型创建一个AutoTokenizer对象


def get_bert_feature(text, word2ph, device=None):  # 定义一个名为get_bert_feature的函数，接受text、word2ph和device三个参数
    if (
        sys.platform == "darwin"  # 判断操作系统是否为darwin（苹果电脑）
        and torch.backends.mps.is_available()  # 判断是否支持torch.backends.mps
        and device == "cpu"  # 判断device是否为cpu
    ):
        device = "mps"  # 如果满足上述条件，则将device设置为"mps"
    if not device:  # 如果device为空
        device = "cuda"  # 将device设置为"cuda"
    model = AutoModelForMaskedLM.from_pretrained("./bert/bert-base-japanese-v3").to(
        device  # 使用"./bert/bert-base-japanese-v3"路径下的预训练模型创建一个AutoModelForMaskedLM对象，并将其移动到指定的device上
    )
    with torch.no_grad():  # 在下面的代码块中，不计算梯度
        inputs = tokenizer(text, return_tensors="pt")
```
这行代码使用tokenizer将输入的文本转换为模型可接受的输入格式，并将结果存储在变量inputs中。

```
        for i in inputs:
            inputs[i] = inputs[i].to(device)
```
这个循环将inputs中的每个张量转移到指定的设备上，以便在该设备上进行模型推理。

```
        res = model(**inputs, output_hidden_states=True)
```
这行代码使用模型对输入进行推理，并将结果存储在变量res中。设置output_hidden_states=True可以使模型返回所有隐藏状态。

```
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```
这行代码将模型输出的隐藏状态拼接在一起，并将结果存储在变量res中。然后，它选择最后一个隐藏状态，并将其移动到CPU上。

```
    assert inputs["input_ids"].shape[-1] == len(word2ph)
```
这行代码使用断言来确保输入的input_ids的最后一个维度与word2ph的长度相等。

```
    word2phone = word2ph
```
这行代码将word2ph赋值给变量word2phone。

```
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
```
这段代码创建一个空列表phone_level_feature，并使用循环遍历word2phone的每个元素。对于每个元素，它将res中的对应隐藏状态重复word2phone[i]次，并将结果添加到phone_level_feature中。

```
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
```
这行代码将phone_level_feature中的所有张量按行拼接在一起，形成一个新的张量。

```
    return phone_level_feature.T
```
这行代码返回phone_level_feature的转置。
```