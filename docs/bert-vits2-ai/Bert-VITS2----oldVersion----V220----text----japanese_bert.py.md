# `Bert-VITS2\oldVersion\V220\text\japanese_bert.py`

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

# 设置本地路径
LOCAL_PATH = "./bert/deberta-v2-large-japanese-char-wwm"

# 使用 AutoTokenizer 类从预训练模型中加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数 get_bert_feature，用于获取 BERT 特征
def get_bert_feature(
    text,
    word2ph,
    device=config.bert_gen_config.device,
    style_text=None,
    style_weight=0.7,
):
    # 将文本转换为片假名，并连接成字符串
    text = "".join(text2sep_kata(text)[0])
    # 检查系统平台是否为 macOS，是否支持多进程并且设备为 CPU
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则设备为 CUDA
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载预训练模型并将其移动到设备上
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，并返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入数据移动到指定设备上
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型的输出，并返回隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 拼接最后三层隐藏状态的结果，并将其移动到 CPU 上
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
        # 如果存在样式文本，则对样式文本进行相同的处理
        if style_text:
            style_inputs = tokenizer(style_text, return_tensors="pt")
            for i in style_inputs:
                style_inputs[i] = style_inputs[i].to(device)
            style_res = models[device](**style_inputs, output_hidden_states=True)
            style_res = torch.cat(style_res["hidden_states"][-3:-2], -1)[0].cpu()
            style_res_mean = style_res.mean(0)

    # 断言单词到音素的长度等于文本长度加上 2
    assert len(word2ph) == len(text) + 2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空的音素级特征列表
    phone_level_feature = []
    # 遍历 word2phone 列表
    for i in range(len(word2phone)):
        # 如果存在样式文本，则根据样式权重对特征进行加权平均
        if style_text:
            repeat_feature = (
                res[i].repeat(word2phone[i], 1) * (1 - style_weight)
                + style_res_mean.repeat(word2phone[i], 1) * style_weight
            )
        # 否则，直接重复特征
        else:
            repeat_feature = res[i].repeat(word2phone[i], 1)
        # 将重复的特征添加到音素级特征列表中
        phone_level_feature.append(repeat_feature)

    # 将音素级特征拼接成张量
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的音素级特征
    return phone_level_feature.T

```