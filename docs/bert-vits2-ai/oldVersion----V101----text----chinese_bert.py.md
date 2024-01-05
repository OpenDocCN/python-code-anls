# `d:/src/tocomm/Bert-VITS2\oldVersion\V101\text\chinese_bert.py`

```
import torch  # 导入torch模块，用于深度学习任务
import sys  # 导入sys模块，用于系统相关操作
from transformers import AutoTokenizer, AutoModelForMaskedLM  # 从transformers库中导入AutoTokenizer和AutoModelForMaskedLM类

device = torch.device(
    "cuda"  # 如果GPU可用，则使用cuda设备
    if torch.cuda.is_available()
    else (
        "mps"  # 如果系统是macOS且支持多进程服务，则使用mps设备
        if sys.platform == "darwin" and torch.backends.mps.is_available()
        else "cpu"  # 否则使用cpu设备
    )
)

tokenizer = AutoTokenizer.from_pretrained("./bert/chinese-roberta-wwm-ext-large")  # 从预训练模型中加载tokenizer
model = AutoModelForMaskedLM.from_pretrained("./bert/chinese-roberta-wwm-ext-large").to(
    device  # 从预训练模型中加载MaskedLM模型，并将其移动到指定设备上
)
```

这段代码主要实现了以下功能：
1. 导入了torch模块，用于深度学习任务。
2. 导入了sys模块，用于系统相关操作。
3. 从transformers库中导入了AutoTokenizer和AutoModelForMaskedLM类，用于加载预训练模型。
4. 根据系统环境和设备可用性，选择合适的设备（cuda、mps或cpu）。
5. 使用AutoTokenizer类从指定路径的预训练模型中加载tokenizer。
6. 使用AutoModelForMaskedLM类从指定路径的预训练模型中加载MaskedLM模型，并将其移动到选择的设备上。
def get_bert_feature(text, word2ph):
    # 禁用梯度计算
    with torch.no_grad():
        # 使用tokenizer对文本进行编码，返回编码后的张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用模型进行推理，输出隐藏状态
        res = model(**inputs, output_hidden_states=True)
        # 取出倒数第3层到倒数第2层的隐藏状态，并在最后一个维度上进行拼接
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

    # 断言确保word2ph的长度与text的长度加2相等
    assert len(word2ph) == len(text) + 2
    # 将word2ph赋值给word2phone
    word2phone = word2ph
    # 初始化phone_level_feature列表
    phone_level_feature = []
    # 遍历word2phone的长度
    for i in range(len(word2phone)):
        # 将res[i]在第一个维度上进行重复word2phone[i]次，得到重复特征
        repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复特征添加到phone_level_feature列表中
        phone_level_feature.append(repeat_feature)

    # 在第0维度上进行拼接，得到phone_level_feature
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回phone_level_feature的转置
    return phone_level_feature.T
if __name__ == "__main__":
    # feature = get_bert_feature('你好,我是说的道理。')
    import torch

    word_level_feature = torch.rand(38, 1024)  # 创建一个38行1024列的张量，用于存储词级别的特征
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
```

这段代码的作用是创建一个38行1024列的张量，用于存储词级别的特征，并初始化一个列表`word2phone`，其中每个元素表示对应词的电话号码。
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
- 1：表示该行代码是为了处理 ZIP 文件的操作，包括读取 ZIP 文件的二进制内容并封装成字节流、使用字节流内容创建 ZIP 对象、遍历 ZIP 对象中的文件名、读取 ZIP 文件中的文件数据、组成文件名到数据的字典。

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
这行代码将打开给定的 ZIP 文件，并将其内容读取为二进制数据。然后，使用BytesIO类将二进制数据封装成字节流对象。

```
# 使用字节流里面内容创建 ZIP 对象
zip = zipfile.ZipFile(bio, 'r')
```
这行代码使用字节流对象的内容创建一个ZIP对象。'r'参数表示以只读模式打开ZIP文件。

```
# 遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典
fdict = {n:zip.read(n) for n in zip.namelist()}
```
这行代码遍历ZIP对象中包含的所有文件名，并使用zip.read(n)读取每个文件的数据。然后，将文件名和数据组成一个字典。

```
# 关闭 ZIP 对象
zip.close()
```
这行代码关闭ZIP对象，释放资源。

```
# 返回结果字典
return fdict
```
这行代码将结果字典返回给调用者。
```