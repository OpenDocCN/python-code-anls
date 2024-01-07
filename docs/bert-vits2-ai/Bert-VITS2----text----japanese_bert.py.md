# `Bert-VITS2\text\japanese_bert.py`

```

# 导入 sys 模块
import sys

# 导入 torch 模块
import torch
# 从 transformers 模块中导入 AutoModelForMaskedLM 和 AutoTokenizer 类
from transformers import AutoModelForMaskedLM, AutoTokenizer
# 从 config 模块中导入 config 变量
from config import config
# 从 text.japanese 模块中导入 text2sep_kata 函数
from text.japanese import text2sep_kata

# 设置本地路径常量
LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"

# 使用 AutoTokenizer 类从预训练模型路径中加载分词器
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数，用于获取 BERT 特征
def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    # 将文本转换为片假名，并连接成字符串
    text = "".join(text2sep_kata(text)[0])
    # 如果存在样式文本，将其转换为片假名，并连接成字符串
    if style_text:
        style_text = "".join(text2sep_kata(style_text)[0])
    # 检查系统平台和设备类型，进行相应的设备调整
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载对应设备的预训练模型，并将其移动到设备上
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 使用 torch.no_grad() 上下文管理器，避免梯度计算
    with torch.no_grad():
        # 使用分词器对文本进行编码，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出，并返回隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 对隐藏状态进行处理，得到特征表示
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本，进行类似的处理
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    # 断言，确保 word2ph 的长度等于文本长度加上 2
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表，用于存储每个音素级别的特征
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 如果存在样式文本，进行特征重复和加权处理
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        # 否则，直接重复特征
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复后的特征添加到列表中
        phone_level_feature.append(repeat_feature)

    # 将所有音素级别的特征拼接成一个张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的音素级别特征
    return phone_level_feature.T

```