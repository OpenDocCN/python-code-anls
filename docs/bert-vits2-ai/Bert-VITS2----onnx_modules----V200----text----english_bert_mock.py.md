# `Bert-VITS2\onnx_modules\V200\text\english_bert_mock.py`

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

# 使用本地路径初始化 DebertaV2Tokenizer 对象
tokenizer = DebertaV2Tokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数，用于获取 BERT 特征
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 如果运行平台是 macOS，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典的键中
    if device not in models.keys():
        # 将本地路径的模型添加到模型字典中，并将其移动到指定设备
        models[device] = DebertaV2Model.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 使用 tokenizer 对文本进行编码，返回 PyTorch 张量
        inputs = tokenizer(text, return_tensors="pt")
        # 将输入张量移动到指定设备
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        # 获取模型输出的隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层和倒数第二层的隐藏状态，并进行拼接和转移
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言词语到音素的长度等于文本长度加2
    # assert len(word2ph) == len(text)+2
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表，用于存储每个音素的特征
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 将 res[i] 重复 word2phone[i] 次，并添加到 phone_level_feature 列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)

    # 将 phone_level_feature 列表中的张量进行拼接
    phone_level_feature = torch.cat(phone_level_feature, dim=0)

    # 返回转置后的 phone_level_feature
    return phone_level_feature.T

```