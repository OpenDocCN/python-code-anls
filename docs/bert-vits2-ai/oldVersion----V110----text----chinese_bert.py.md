# `d:/src/tocomm/Bert-VITS2\oldVersion\V110\text\chinese_bert.py`

```
import torch  # 导入torch模块，用于深度学习任务
import sys  # 导入sys模块，用于获取系统信息
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM类

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")  # 使用预训练的tokenizer初始化一个AutoTokenizer对象

def get_bert_feature(text, word2ph, device=None):
    # 判断操作系统是否为macOS，并且torch.backends.mps可用，并且设备为cpu，则将设备设置为"mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备未指定，则将设备设置为"cuda"
    if not device:
        device = "cuda"
    # 使用预训练的模型初始化一个AutoModelForMaskedLM对象，并将其移动到指定的设备上
    model = AutoModelForMaskedLM.from_pretrained(
        "./bert/chinese-roberta-wwm-ext-large"
    ).to(device)
    # 在不计算梯度的情况下执行以下代码块
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
```
这行代码使用tokenizer对文本进行处理，将其转换为模型可以接受的输入格式。

```
        for i in inputs:
            inputs[i] = inputs[i].to(device)
```
这个循环将inputs中的每个张量都移动到指定的设备上，以便在该设备上进行模型推理。

```
        res = model(**inputs, output_hidden_states=True)
```
这行代码使用模型对输入进行推理，并返回输出结果。设置output_hidden_states=True可以获取模型的隐藏状态。

```
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```
这行代码将模型的隐藏状态拼接起来，并将结果移动到CPU上。

```
    assert len(word2ph) == len(text) + 2
```
这行代码用于断言，确保word2ph的长度等于text的长度加上2。

```
    word2phone = word2ph
```
这行代码将word2ph赋值给word2phone。

```
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
```
这个循环根据word2phone的长度，将res中的每个元素重复word2phone[i]次，并将结果添加到phone_level_feature列表中。

```
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
```
这行代码将phone_level_feature列表中的张量按照dim=0的维度进行拼接。

```
    return phone_level_feature.T
```
这行代码返回phone_level_feature的转置。

```
if __name__ == "__main__":
    import torch
```
这个条件判断语句用于判断当前脚本是否作为主程序运行，如果是，则导入torch模块。
# 创建一个38行1024列的张量，表示12个词的每个词的1024维特征
word_level_feature = torch.rand(38, 1024)  
# 创建一个列表，存储每个词对应的电话号码
word2phone = [
    1,  # 第一个词对应的电话号码
    2,  # 第二个词对应的电话号码
    1,  # 第三个词对应的电话号码
    2,  # 第四个词对应的电话号码
    2,  # 第五个词对应的电话号码
    1,  # 第六个词对应的电话号码
    2,  # 第七个词对应的电话号码
    2,  # 第八个词对应的电话号码
    1,  # 第九个词对应的电话号码
    2,  # 第十个词对应的电话号码
    2,  # 第十一个词对应的电话号码
    1,  # 第十二个词对应的电话号码
    2,  # 第十三个词对应的电话号码
    2,  # 第十四个词对应的电话号码
    2,  # 第十五个词对应的电话号码
    2,  # 第十六个词对应的电话号码
    2,  # 第十七个词对应的电话号码
    2,  # 第十八个词对应的电话号码
    ...
]
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())
    # 使用字节流里面内容创建 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')
    # 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}
    # 关闭 ZIP 对象
    zip.close()
    # 返回结果字典
    return fdict
```

需要注释的代码：

```
        1,  # 第一个语句
        1,  # 第二个语句
        2,  # 第三个语句
        2,  # 第四个语句
        1,  # 第五个语句
        2,  # 第六个语句
        2,  # 第七个语句
        2,  # 第八个语句
        2,  # 第九个语句
        1,  # 第十个语句
        2,  # 第十一个语句
        2,  # 第十二个语句
        2,  # 第十三个语句
        2,  # 第十四个语句
        2,  # 第十五个语句
        1,  # 第十六个语句
        2,  # 第十七个语句
        2,  # 第十八个语句
        2,  # 第十九个语句
        2,  # 第二十个语句
```

这些代码是一个示例，没有具体的含义。请提供具体的代码，以便我可以为每个语句添加注释。
        1,
    ]
```
这是一个列表，包含一个整数1。

```
    # 计算总帧数
    total_frames = sum(word2phone)
```
计算列表`word2phone`中所有元素的总和，并将结果赋值给变量`total_frames`。

```
    print(word_level_feature.shape)
    print(word2phone)
```
打印变量`word_level_feature`的形状和变量`word2phone`的值。

```
    phone_level_feature = []
    for i in range(len(word2phone)):
        print(word_level_feature[i].shape)

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
```
创建一个空列表`phone_level_feature`，然后遍历`word2phone`列表的索引。在每次循环中，打印`word_level_feature[i]`的形状。然后，使用`repeat`函数将`word_level_feature[i]`重复`word2phone[i]`次，并将结果添加到`phone_level_feature`列表中。

```
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    print(phone_level_feature.shape)  # torch.Size([36, 1024])
```
使用`torch.cat`函数将`phone_level_feature`列表中的所有张量按行连接起来，并将结果赋值给`phone_level_feature`变量。然后打印`phone_level_feature`的形状。
```