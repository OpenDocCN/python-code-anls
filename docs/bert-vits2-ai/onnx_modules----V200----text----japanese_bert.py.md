# `Bert-VITS2\onnx_modules\V200\text\japanese_bert.py`

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

# 设置本地路径常量
LOCAL_PATH = "./bert/deberta-v2-large-japanese"

# 使用 AutoTokenizer 类从预训练模型中加载分词器
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空的模型字典
models = dict()

# 定义函数，获取 BERT 特征
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 将文本转换为片假名，并获取分词后的结果
    sep_text, _, _ = text2sep_kata(text)
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]
    # 调用 get_bert_feature_with_token 函数，获取 BERT 特征
    return get_bert_feature_with_token(sep_ids, word2ph, device)

# 定义函数，获取带有 token 的 BERT 特征
def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):
    # 如果运行平台为 macOS，且支持 MPS，且设备为 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则加载对应设备的预训练模型
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 使用 torch.no_grad() 上下文管理器，执行以下操作
    with torch.no_grad():
        # 将 tokens 转换为 tensor，并移动到指定设备上
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)
        token_type_ids = torch.zeros_like(inputs).to(device)
        attention_mask = torch.ones_like(inputs).to(device)
        inputs = {
            "input_ids": inputs,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        # 获取模型输出的隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层的隐藏状态，并转移到 CPU 上
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的 input_ids 的形状与 word2ph 的长度相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    phone_level_feature = []
    # 遍历 word2phone，重复对应位置的特征，并添加到 phone_level_feature 中
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
    # 将 phone_level_feature 拼接起来，得到 phone_level_feature 的转置
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

```