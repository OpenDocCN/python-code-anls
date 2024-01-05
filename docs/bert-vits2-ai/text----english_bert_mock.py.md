# `d:/src/tocomm/Bert-VITS2\text\english_bert_mock.py`

```
import sys  # 导入sys模块，用于访问与Python解释器交互的变量和函数

import torch  # 导入torch模块，用于构建和训练神经网络
from transformers import DebertaV2Model, DebertaV2Tokenizer  # 从transformers模块中导入DebertaV2Model和DebertaV2Tokenizer类

from config import config  # 从config模块中导入config变量

LOCAL_PATH = "./bert/deberta-v3-large"  # 设置本地路径变量

tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)  # 使用本地路径初始化DebertaV2Tokenizer对象

models = dict()  # 创建一个空的字典对象

def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,  # 定义一个名为get_bert_feature的函数，接受text、word2ph、device和style_text等参数
    style_weight=0.7,  # 设置风格权重为0.7

):  # 函数参数列表的结束

    # 如果运行环境是 macOS，并且支持 MPS，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"

    # 如果设备未指定，则将设备设置为 "cuda"
    if not device:
        device = "cuda"

    # 如果设备不在模型字典中，则使用预训练模型创建对应设备的模型，并将其移动到设备上
    if device not in models.keys():
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)

    # 使用 torch.no_grad() 上下文管理器，避免梯度计算
    with torch.no_grad():
        # 使用分词器处理文本，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")

        # 将输入数据移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)

        # 使用模型进行推理，输出隐藏状态
        res = models[device](**inputs, output_hidden_states=True)

        # 从模型输出的隐藏状态中取出倒数第三层到倒数第二层的隐藏状态，并进行拼接
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()

        # 如果有样式文本，则使用分词器处理样式文本，返回 PyTorch 张量
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
# 遍历 style_inputs 中的每个元素，将其值转移到指定设备上
for i in style_inputs:
    style_inputs[i] = style_inputs[i].to(device)

# 使用指定设备上的模型对 style_inputs 进行推理，输出隐藏状态
style_res = models[device](**style_inputs, output_hidden_states=True)

# 从隐藏状态中选择倒数第三层到倒数第二层的结果，并在指定维度上进行拼接
style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()

# 计算 style_res 的均值
style_res_mean = style_res.mean(0)

# 断言 word2ph 的长度与 res 的行数相等，如果不相等则抛出异常
assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))

# 将 word2ph 赋值给 word2phone
word2phone = word2ph

# 初始化 phone_level_feature 列表
phone_level_feature = []

# 遍历 word2phone 的长度
for i in range(len(word2phone)):
    # 如果 style_text 存在
    if style_text:
        # 重复 res[i] 并根据权重添加 style_res_mean
        repeat_feature = (
            res[i].repeat(word2phone[i], 1) * (1 - style_weight)
            + style_res_mean.repeat(word2phone[i], 1) * style_weight
        )
    else:
        # 否则，只重复 res[i]
        repeat_feature = res[i].repeat(word2phone[i], 1)
    # 将 repeat_feature 添加到 phone_level_feature 中
    phone_level_feature.append(repeat_feature)

# 在指定维度上拼接 phone_level_feature
phone_level_feature = torch.cat(phone_level_feature, dim=0)
# 返回变量 phone_level_feature 的转置结果
```