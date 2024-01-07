# `Bert-VITS2\oldVersion\V210\text\japanese_bert.py`

```

# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 AutoModelForMaskedLM 和 AutoTokenizer 类
from transformers import AutoModelForMaskedLM, AutoTokenizer
# 从 config 模块中导入 config 变量
from config import config
# 从当前目录下的 japanese 模块中导入 text2sep_kata 函数
from .japanese import text2sep_kata

# 设置本地路径
LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"
# 使用 AutoTokenizer 类从预训练模型中加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数 get_bert_feature，接受多个参数
def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    # 将文本转换为片假名
    text = "".join(text2sep_kata(text)[0])
    # 如果存在 style_text，则将其转换为片假名
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])
    # 检查系统平台和设备类型，根据条件重新设置设备类型
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    # 如果设备类型不在模型字典中，则加载对应设备类型的预训练模型并存储在模型字典中
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 使用 torch.no_grad() 上下文管理器，执行模型推理
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 执行模型推理，获取隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 对隐藏状态进行处理，得到最终结果
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在 style_text，则执行类似的操作
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    # 断言语句，用于检查条件是否为真，否则抛出异常
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表，用于存储每个音素的特征
    phone_level_feature = []
    # 遍历 word2phone 列表
    for i in range(len(word2phone)):
        # 如果存在 style_text，则根据权重计算重复特征
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        # 否则，直接重复特征
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复特征添加到列表中
        phone_level_feature.append(repeat_feature)

    # 将列表中的特征张量拼接起来
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的特征张量
    return phone_level_feature.T

```