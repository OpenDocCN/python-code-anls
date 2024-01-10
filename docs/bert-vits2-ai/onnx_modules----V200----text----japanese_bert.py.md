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

# 使用 AutoTokenizer 类从预训练模型中加载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(LOCAL_PATH)

# 创建空字典 models
models = dict()

# 定义函数 get_bert_feature，接受文本、word2ph 和设备参数
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 将文本转换为分隔片假名
    sep_text, _, _ = text2sep_kata(text)
    # 对分隔片假名进行分词
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]
    # 将分词转换为对应的 id
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]
    # 将分词 id 进行扁平化处理，并添加起始和结束 token
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]
    # 调用 get_bert_feature_with_token 函数，传入处理好的分词 id、word2ph 和设备参数
    return get_bert_feature_with_token(sep_ids, word2ph, device)

# 定义函数 get_bert_feature_with_token，接受 tokens、word2ph 和设备参数
def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):
    # 如果运行平台是 macOS，并且支持多进程并行计算，并且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备参数为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在 models 字典的键中
    if device not in models.keys():
        # 从本地路径加载预训练模型，并将其移动到指定设备
        models[device] = AutoModelForMaskedLM.from_pretrained(LOCAL_PATH).to(device)
    # 禁用梯度计算
    with torch.no_grad():
        # 将 tokens 转换为 tensor，并移动到指定设备，然后添加一个维度
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)
        # 创建 token_type_ids 和 attention_mask
        token_type_ids = torch.zeros_like(inputs).to(device)
        attention_mask = torch.ones_like(inputs).to(device)
        inputs = {
            "input_ids": inputs,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        # 调用模型进行推理，获取隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层和倒数第二层的隐藏状态，并拼接起来
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的 token 数量和 word2ph 的长度相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表 phone_level_feature
    phone_level_feature = []
    # 遍历 word2phone
    for i in range(len(word2phone)):
        # 将 res[i] 复制 word2phone[i] 次，并添加到 phone_level_feature 中
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
    # 将 phone_level_feature 拼接起来，并转置
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T
```