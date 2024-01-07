# `Bert-VITS2\oldVersion\V210\text\english_bert_mock.py`

```

# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 DebertaV2Model 和 DebertaV2Tokenizer 类
from transformers import DebertaV2Model, DebertaV2Tokenizer

# 从 config 模块中导入 config 变量
from config import config

# 设置本地路径
LOCAL_PATH = "./bert/deberta-v3-large"

# 使用 DebertaV2Tokenizer 类从预训练模型中加载 tokenizer
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数 get_bert_feature，接受文本、word2ph、设备、样式文本和样式权重作为参数
def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    # 如果操作系统是 macOS，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载预训练模型并将其移动到指定设备上
    if device not in models.keys():
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 使用模型进行推理，输出隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层和倒数第二层的隐藏状态，并进行拼接和处理
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本，则对样式文本进行相同的处理
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)
    # 断言 word2ph 的长度与 res 的形状一致
    assert len(word2ph) == res.shape[0], (text, res.shape[0], len(word2ph))
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表存储每个单词对应的特征
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 如果存在样式文本，则根据样式权重对特征进行加权处理
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        # 如果不存在样式文本，则直接重复特征
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将处理后的特征添加到列表中
        phone_level_feature.append(repeat_feature)

    # 将列表中的特征拼接成张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回处理后的特征张量的转置
    return phone_level_feature.T

```