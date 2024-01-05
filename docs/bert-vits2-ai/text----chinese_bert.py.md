# `d:/src/tocomm/Bert-VITS2\text\chinese_bert.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建和训练神经网络
from transformers import AutoModelForMaskedLM, AutoTokenizer  # 从transformers模块中导入AutoModelForMaskedLM和AutoTokenizer类

from config import config  # 从config模块中导入config变量

LOCAL_PATH = "./bert/chinese-roberta-wwm-ext-large"  # 设置本地路径变量

tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)  # 使用预训练的tokenizer模型初始化tokenizer变量

models = dict()  # 创建一个空的字典变量models

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,  # 设置device变量，默认值为config.bert_gen_config.device
    style_text=None,  # 设置style_text变量，默认值为None
    style_weight=0.7,  # 设置style_weight变量，默认值为0.7
    # 检查当前操作系统是否为 macOS，并且是否支持 torch.backends.mps，以及设备是否为 CPU
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
    # 如果设备不在模型字典中，则使用预训练模型创建相应设备的模型并转移到该设备
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用分词器处理文本，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据转移到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用指定设备的模型进行推理，输出隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 将隐藏状态的倒数第三层和倒数第二层拼接，并转移到 CPU
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本
        if style_text:
            # 使用分词器处理样式文本，返回 PyTorch 张量
            style_inputs = tokenizer(style_text, return_tensors="pt")
            # 将样式输入数据转移到指定设备
            for i in style_inputs:
                # ...
                style_inputs[i] = style_inputs[i].to(device)  # 将style_inputs中的第i个元素转移到指定的设备上

            style_res = models[device](**style_inputs, output_hidden_states=True)  # 使用指定设备上的模型对style_inputs进行推理，输出隐藏状态
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()  # 将隐藏状态拼接并转移到CPU上

            style_res_mean = style_res.mean(0)  # 计算隐藏状态的均值

    assert len(word2ph) == len(text) + 2  # 断言word2ph的长度等于text的长度加2
    word2phone = word2ph  # 将word2ph赋值给word2phone
    phone_level_feature = []  # 初始化phone_level_feature列表

    for i in range(len(word2phone)):  # 遍历word2phone的长度
        if style_text:  # 如果style_text为真
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )  # 计算重复特征，根据style_text和style_weight的值进行加权计算
        else:  # 如果style_text为假
            repeat_feature = res[i].repeat(word2phone[i], 1)  # 直接重复特征
        phone_level_feature.append(repeat_feature)  # 将重复特征添加到phone_level_feature列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 将phone_level_feature列表拼接成张量

    return phone_level_feature.T  # 返回phone_level_feature的转置
if __name__ == "__main__":
    # 创建一个38行1024列的随机张量，表示12个词，每个词有1024维特征
    word_level_feature = torch.rand(38, 1024)  
    # 创建一个包含12个元素的列表，每个元素是1或2
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
        2,
# 根据 ZIP 文件名读取内容，返回其中文件名到数据的字典
def read_zip(fname):
    # 根据 ZIP 文件名读取其二进制，封装成字节流
    bio = BytesIO(open(fname, 'rb').read())  # 读取指定文件的二进制内容，并封装成字节流对象
    使用字节流里面内容创建 ZIP 对象  # 使用字节流内容创建一个 ZIP 对象
    zip = zipfile.ZipFile(bio, 'r')  # 使用字节流创建的 ZIP 对象，以只读模式打开
    遍历 ZIP 对象所包含文件的文件名，读取文件数据，组成文件名到数据的字典  # 遍历 ZIP 文件中的文件名，读取文件数据，组成文件名到数据的字典
    fdict = {n:zip.read(n) for n in zip.namelist()}  # 使用字典推导式，将 ZIP 文件中的文件名和对应的数据组成字典
    # 关闭 ZIP 对象  # 关闭 ZIP 对象，释放资源
    zip.close()
    # 返回结果字典  # 返回组成的文件名到数据的字典
    return fdict
        2,  # 定义一个包含数字2的列表
        2,  # 定义一个包含数字2的列表
        1,  # 定义一个包含数字1的列表
    ]

    # 计算总帧数
    total_frames = sum(word2phone)  # 计算word2phone列表中所有元素的总和，赋值给total_frames
    print(word_level_feature.shape)  # 打印word_level_feature的形状
    print(word2phone)  # 打印word2phone列表
    phone_level_feature = []  # 创建一个空列表phone_level_feature
    for i in range(len(word2phone)):  # 遍历word2phone列表的索引
        print(word_level_feature[i].shape)  # 打印word_level_feature中第i个元素的形状

        # 对每个词重复word2phone[i]次
        repeat_feature = word_level_feature[i].repeat(word2phone[i], 1)  # 将word_level_feature中第i个元素在第1维度上重复word2phone[i]次，赋值给repeat_feature
        phone_level_feature.append(repeat_feature)  # 将repeat_feature添加到phone_level_feature列表中

    phone_level_feature = torch.cat(phone_level_feature, dim=0)  # 在第0维度上对phone_level_feature列表中的张量进行拼接，赋值给phone_level_feature
    print(phone_level_feature.shape)  # 打印phone_level_feature的形状，预期为torch.Size([36, 1024])
```