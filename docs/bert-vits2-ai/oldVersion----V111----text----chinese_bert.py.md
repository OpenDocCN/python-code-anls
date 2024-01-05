# `d:/src/tocomm/Bert-VITS2\oldVersion\V111\text\chinese_bert.py`

```
import torch  # 导入torch模块，用于深度学习任务
import sys  # 导入sys模块，用于获取系统信息
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM类

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")  # 使用指定的预训练模型初始化tokenizer对象

models = dict()  # 创建一个空字典，用于存储不同设备上的模型


def get_bert_feature(text, word2ph, device=None):
    # 判断当前系统是否是macOS，并且是否支持torch.backends.mps，以及是否指定了使用CPU设备
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"  # 如果满足条件，则将设备设置为"mps"
    if not device:  # 如果没有指定设备
        device = "cuda"  # 将设备设置为默认的GPU设备
    if device not in models.keys():  # 如果当前设备不在models字典的键中
        models[device] = AutoModelForMaskedLM.from_pretrained(  # 使用指定的预训练模型初始化模型对象，并将其存储在models字典中
            "./bert/chinese-roberta-wwm-ext-large"
        )
            "./bert/chinese-roberta-wwm-ext-large"
        ).to(device)
```
这段代码是使用预训练的BERT模型进行文本编码。它加载了一个预训练的BERT模型，并将其移动到指定的设备上。

```
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = models[device](**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
```
这段代码使用BERT模型对输入的文本进行编码。它首先使用tokenizer将文本转换为模型所需的输入格式，并将其移动到指定的设备上。然后，它使用模型对输入进行编码，并获取模型的隐藏状态。最后，它将隐藏状态的最后三层进行拼接，并将结果移动到CPU上。

```
    assert len(word2ph) == len(text) + 2
    word2phone = word2ph
    phone_level_feature = []
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
```
这段代码用于生成每个音素的特征表示。它首先确保音素的数量与文本的长度加2相等。然后，它将word2ph赋值给word2phone。接下来，它遍历word2phone的每个元素，并使用res[i]对其进行重复，以匹配音素的数量。最后，它将重复的特征添加到phone_level_feature列表中。

```
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    return phone_level_feature.T
```
这段代码将phone_level_feature列表中的特征拼接在一起，并返回其转置。它使用torch.cat函数将列表中的特征按行拼接起来，并将结果转置后返回。
# 如果当前脚本是直接被运行的，则执行以下代码块
if __name__ == "__main__":
    import torch  # 导入torch模块

    word_level_feature = torch.rand(38, 1024)  # 创建一个38行1024列的张量，每个元素是0到1之间的随机数，表示词级别的特征
    word2phone = [  # 创建一个列表，存储词到音素的映射关系
        1,  # 第一个词对应的音素
        2,  # 第二个词对应的音素
        1,  # 第三个词对应的音素
        2,  # 第四个词对应的音素
        2,  # 第五个词对应的音素
        1,  # 第六个词对应的音素
        2,  # 第七个词对应的音素
        2,  # 第八个词对应的音素
        1,  # 第九个词对应的音素
        2,  # 第十个词对应的音素
        2,  # 第十一个词对应的音素
        1,  # 第十二个词对应的音素
        2,  # 第十三个词对应的音素
        2,  # 第十四个词对应的音素
        ...
```

注释解释了代码的作用和功能，使得其他人能够更容易地理解代码的含义和逻辑。
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
        2,  # 读取 ZIP 文件的二进制内容并封装成字节流
        2,  # 使用字节流内容创建 ZIP 对象
        2,  # 遍历 ZIP 对象中的文件名
        1,  # 读取 ZIP 文件中的文件数据
        1,  # 组成文件名到数据的字典
        2,  # 关闭 ZIP 对象
        2,  # 返回结果字典
        1,  # 读取 ZIP 文件的二进制内容并封装成字节流
        2,  # 使用字节流内容创建 ZIP 对象
        2,  # 遍历 ZIP 对象中的文件名
        2,  # 读取 ZIP 文件中的文件数据
        2,  # 组成文件名到数据的字典
        1,  # 关闭 ZIP 对象
        2,  # 返回结果字典
        2,  # 读取 ZIP 文件的二进制内容并封装成字节流
        2,  # 使用字节流内容创建 ZIP 对象
        2,  # 遍历 ZIP 对象中的文件名
        2,  # 读取 ZIP 文件中的文件数据
        2,  # 组成文件名到数据的字典
        1,  # 关闭 ZIP 对象
        2   # 返回结果字典
```

注释解释：

- 2：表示该行代码是为了处理 ZIP 文件的操作，包括读取 ZIP 文件的二进制内容并封装成字节流、使用字节流内容创建 ZIP 对象、遍历 ZIP 对象中的文件名、读取 ZIP 文件中的文件数据、组成文件名到数据的字典、关闭 ZIP 对象、返回结果字典。
- 1：表示该行代码是为了处理 ZIP 文件的操作，包括读取 ZIP 文件的二进制内容并封装成字节流、使用字节流内容创建 ZIP 对象、遍历 ZIP 对象中的文件名、读取 ZIP 文件中的文件数据、组成文件名到数据的字典、关闭 ZIP 对象。

        2,  # 定义一个整数2
        2,  # 定义一个整数2
        2,  # 定义一个整数2
        1,  # 定义一个整数1
    ]

    # 计算总帧数
    total_frames = sum(word2phone)  # 计算列表word2phone中所有元素的和，赋值给total_frames
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone列表的索引
        print(word_level_feature[i].shape)  # 打印word_level_feature[i]的形状

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 将word_level_feature[i]在第1维度上重复word2phone[i]次，赋值给repeat_feature
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 在第0维度上将phone_level_feature列表中的张量拼接起来，赋值给phone_level_feature
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状，torch.Size([36, 1024])
```

注释解释了每个语句的作用，包括定义变量、计算总帧数、打印张量形状、遍历列表、重复张量、拼接张量等操作。
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
# 根据 ZIP 文件名读取其二进制，封装成字节流
bio = BytesIO(open(fname, 'rb').read())
```
这行代码将打开给定的 ZIP 文件，并将其内容读取为二进制数据。然后，使用BytesIO将二进制数据封装成字节流。

```
# 使用字节流里面内容创建 ZIP 对象
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用字节流里面的内容创建一个ZIP对象。该对象可以用于读取ZIP文件中的文件。

```
# 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码遍历ZIP对象中包含的文件名，并使用zip.read(n)读取每个文件的数据。然后，将文件名和数据组成一个字典。

```
# 关闭 ZIP 对象
zip.close()
```
这行代码关闭ZIP对象，释放资源。

```
# 返回结果字典
return fdict
```
这行代码返回包含文件名和数据的字典作为函数的结果。
```