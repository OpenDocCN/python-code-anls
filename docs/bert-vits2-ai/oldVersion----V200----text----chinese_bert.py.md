# `d:/src/tocomm/Bert-VITS2\oldVersion\V200\text\chinese_bert.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入PyTorch库，用于构建深度学习模型
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers库中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量

LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"  # 设置本地路径变量

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用预训练模型路径初始化tokenizer对象

models = dict()  # 创建一个空字典用于存储模型

def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):  # 定义一个函数，接受文本、word2ph和设备参数
    if (  # 如果条件判断语句开始
        sys.platform == "darwin"  # 当操作系统为MacOS
        and torch.backends.mps.is_available()  # 并且PyTorch的多进程服务可用
        and device == "cpu"  # 并且设备为CPU
    ):  # 条件判断语句结束
        device = "mps"  # 设置默认设备为 "mps"
    if not device:  # 如果设备未指定
        device = "cuda"  # 将设备设置为 "cuda"
    if device not in models.keys():  # 如果设备不在模型字典中
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)  # 从预训练模型中加载模型，并将其移动到指定设备
    with torch.no_grad():  # 禁用梯度计算
        inputs = tokenizer(text, return_tensors="pt")  # 使用分词器对文本进行分词，并返回 PyTorch 张量
        for i in inputs:  # 遍历输入张量
            inputs[i] = inputs[i].to(device)  # 将输入张量移动到指定设备
        res = models[device](**inputs, output_hidden_states=True)  # 使用模型对输入进行推理，输出隐藏状态
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并移动到 CPU 上

    assert len(word2ph) == len(text) + 2  # 断言确保 word2ph 的长度等于文本长度加 2
    word2phone = word2ph  # 将 word2ph 赋值给 word2phone
    phone_level_feature = []  # 初始化电话级别特征列表
    for i in range(len(word2phone)):  # 遍历 word2phone
        repeat_feature = res[i].repeat(word2phone[i], 1)  # 重复 res[i] 特征 word2phone[i] 次
        phone_level_feature.append(repeat_feature)  # 将重复特征添加到电话级别特征列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将电话级别特征列表拼接成张量
    return phone_level_feature.T
```
这行代码返回phone_level_feature的转置。

```python
if __name__ == "__main__":
```
这行代码检查当前模块是否是主程序，如果是则执行下面的代码。

```python
    word_level_feature = torch.rand(38, 1024)  # 12个词,每个词1024维特征
```
这行代码创建一个38行1024列的张量，表示38个词，每个词有1024维的特征。

```python
    word2phone = [
        1,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
        2,
        1,
        2,
```
这部分代码创建了一个名为word2phone的列表，其中包含了一系列数字。由于代码不完整，无法确定这些数字的具体含义。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 读取指定文件的二进制内容，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建一个 ZIP 对象，以便后续操作
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名列表，读取每个文件的数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象，释放资源
    # 返回结果字典
    return fdict  # 返回包含文件名到数据的字典
        2,  # 将数字 2 添加到列表 word2phone 中
        2,  # 将数字 2 添加到列表 word2phone 中
        2,  # 将数字 2 添加到列表 word2phone 中
        2,  # 将数字 2 添加到列表 word2phone 中
        1,  # 将数字 1 添加到列表 word2phone 中
    ]

    # 计算总帧数
    total_frames = sum(word2phone)  # 计算列表 word2phone 中所有数字的总和，存储在 total_frames 中
    print(word_level_feature.shape)  # 打印 word_level_feature 的形状
    print(word2phone)  # 打印列表 word2phone
    phone_level_feature = []  # 创建一个空列表 phone_level_feature
    for i in range(len(word2phone)):  # 遍历列表 word2phone 的索引
        print(word_level_feature[i].shape)  # 打印 word_level_feature 中第 i 个元素的形状

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 将 word_level_feature 中第 i 个元素沿着第二个维度重复 word2phone[i] 次，存储在 repeat_feature 中
        phone_level_feature.append(repeat_feature)  # 将 repeat_feature 添加到列表 phone_level_feature 中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 沿着第一个维度将列表 phone_level_feature 中的元素连接起来，存储在 phone_level_feature 中
    print(phone_level_feature.shape)  # 打印出张量的形状，即它的维度信息，这里的输出是 torch.Size([36, 1024])
```