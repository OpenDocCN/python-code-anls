# `Bert-VITS2\oldVersion\V111\text\fix\japanese_bert.py`

```py
# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 导入 sys 模块
import sys
# 从当前目录下的 japanese 模块中导入 text2sep_kata 函数
from .japanese import text2sep_kata
# 从 config 模块中导入 config 对象
from config import config

# 使用预训练的 tokenizer 创建 tokenizer 对象
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# 创建空的模型字典
models = dict()

# 定义函数，获取 BERT 特征
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 将文本转换为片假名，并忽略返回的第二个值
    sep_text, _ = text2sep_kata(text)
    # 使用 tokenizer 对文本进行分词
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]
    # 将分词转换为对应的 id
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]
    # 将所有 id 拼接成一个列表
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]
    # 调用 get_bert_feature_with_token 函数，传入分词 id 列表和 word2ph 对象
    return get_bert_feature_with_token(sep_ids, word2ph, device)

# 定义函数，获取带有 token 的 BERT 特征
def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):
    # 如果运行平台是 macOS，并且支持 MPS，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则根据设备从预训练模型中加载模型，并将其移动到对应设备
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将 tokens 转换为 tensor，并移动到对应设备，然后增加一个维度
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)
        # 创建 token_type_ids 和 attention_mask
        token_type_ids = torch.zeros_like(inputs).to(device)
        attention_mask = torch.ones_like(inputs).to(device)
        inputs = {
            "input_ids": inputs,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        # 调用模型，获取输出的隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层的隐藏状态，并转移到 CPU
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的 input_ids 的最后一个维度长度与 word2ph 的长度相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表，用于存储每个词的特征
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 将 res[i] 重复 word2phone[i] 次，并添加到 phone_level_feature 列表中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
    # 将 phone_level_feature 拼接成一个 tensor，并进行转置
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T
```