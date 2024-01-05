# `d:/src/tocomm/Bert-VITS2\oldVersion\V210\text\chinese_bert.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建深度学习模型和进行张量运算
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config类

LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"  # 设置本地路径常量

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用预训练模型路径初始化tokenizer对象

models = dict()  # 创建一个空字典用于存储模型

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,  # 设置device参数，默认值为config.bert_gen_config.device
    style_text=None,  # 设置style_text参数，默认值为None
    style_weight=0.7,  # 设置style_weight参数，默认值为0.7
    # 检查当前操作系统是否为 macOS，并且是否支持 MPS（Metal Performance Shaders），以及设备是否为 CPU
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        # 如果满足条件，将设备设置为 "mps"
        device = "mps"
    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则根据本地路径加载预训练模型，并将其移动到指定设备
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用分词器处理文本，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用指定设备上的模型进行推理，输出隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 将最后三层的隐藏状态拼接起来，并移动到 CPU 上
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本
        if style_text:
            # 使用分词器处理样式文本，返回 PyTorch 张量
            style_inputs = tokenizer(style_text, return_tensors="pt")
            # 遍历样式输入数据
            for i in style_inputs:
    assert len(word2ph) == len(text) + 2  # 确保word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 创建一个空列表phone_level_feature用于存储特征

    # 遍历word2phone的长度
    for i in range(len(word2phone)):
        # 如果style_text为真
        if style_text:
            # 重复res[i]的特征值word2phone[i]次，乘以(1 - style_weight)，再加上重复style_res_mean的特征值word2phone[i]次，乘以style_weight
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        else:  # 如果style_text为假
            # 重复res[i]的特征值word2phone[i]次
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将repeat_feature添加到phone_level_feature列表中
        phone_level_feature.append(repeat_feature)

    # 将phone_level_feature列表中的特征值按行拼接成一个张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T
```
这行代码返回转置后的phone_level_feature矩阵。

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
        2,
```
这段代码创建了一个名为word2phone的列表，其中包含了一系列数字。
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制内容，封装成字节流
    使用字节流里面内容创建 ZIP 对象  # 使用字节流内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 创建 ZIP 对象
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典  # 遍历 ZIP 对象的文件名列表，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 用字典推导式创建文件名到数据的字典
    # 关闭 ZIP 对象  # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典  # 返回文件名到数据的字典
    return fdict
        2,  # 将数字2添加到列表word2phone中
        2,  # 将数字2添加到列表word2phone中
        2,  # 将数字2添加到列表word2phone中
        1,  # 将数字1添加到列表word2phone中
    ]

    # 计算总帧数
    total_frames = sum(word2phone)  # 计算列表word2phone中所有元素的总和，赋值给total_frames
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone列表的索引
        print(word_level_feature[i].shape)  # 打印word_level_feature中第i个元素的形状

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 将word_level_feature中第i个元素在第1维上重复word2phone[i]次，赋值给repeat_feature
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 在第0维上对phone_level_feature列表中的张量进行拼接，赋值给phone_level_feature
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状，预期为torch.Size([36, 1024])
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 从文件名读取二进制数据，并封装成字节流对象
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流内容创建 ZIP 对象，以只读模式打开
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    # 关闭 ZIP 对象
    zip.close()  # 关闭 ZIP 对象
    # 返回结果字典
    return fdict  # 返回文件名到数据的字典
```