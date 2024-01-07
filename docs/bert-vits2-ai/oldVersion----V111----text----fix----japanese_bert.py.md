# `Bert-VITS2\oldVersion\V111\text\fix\japanese_bert.py`

```

# 导入 torch 库
import torch
# 从 transformers 库中导入 AutoTokenizer 和 AutoModelForMaskedLM 类
from transformers import AutoTokenizer, AutoModelForMaskedLM
# 导入 sys 模块
import sys
# 从当前目录下的 japanese 模块中导入 text2sep_kata 函数
from .japanese import text2sep_kata
# 从 config 模块中导入 config 对象
from config import config

# 使用预训练的 BERT 模型创建 tokenizer 对象
tokenizer = AutoTokenizer.from_pretrained("./bert/bert-base-japanese-v3")

# 创建空的模型字典
models = dict()

# 定义函数 get_bert_feature，接受文本、word2ph 和设备参数
def get_bert_feature(text, word2ph, device=config.bert_gen_config.device):
    # 将文本转换为片假名，并忽略返回的第二个值
    sep_text, _ = text2sep_kata(text)
    # 使用 tokenizer 对文本进行分词
    sep_tokens = [tokenizer.tokenize(t) for t in sep_text]
    # 将分词后的文本转换为 token ids
    sep_ids = [tokenizer.convert_tokens_to_ids(t) for t in sep_tokens]
    # 将 token ids 进行扁平化，并在开头和结尾添加特殊 token 的 id
    sep_ids = [2] + [item for sublist in sep_ids for item in sublist] + [3]
    # 调用 get_bert_feature_with_token 函数，传入处理好的 token ids、word2ph 和设备参数
    return get_bert_feature_with_token(sep_ids, word2ph, device)

# 定义函数 get_bert_feature_with_token，接受 tokens、word2ph 和设备参数
def get_bert_feature_with_token(tokens, word2ph, device=config.bert_gen_config.device):
    # 如果运行环境是 macOS，且支持多进程并行计算，且设备是 CPU，则将设备设置为 "mps"
    if (
        sys.platform == "darwin"
        and torch.backends.mps.is_available()
        and device == "cpu"
    ):
        device = "mps"
    # 如果设备参数为空，则将设备设置为 "cuda"
    if not device:
        device = "cuda"
    # 如果设备不在模型字典中，则根据设备参数加载对应的预训练 BERT 模型，并存入模型字典
    if device not in models.keys():
        models[device] = AutoModelForMaskedLM.from_pretrained(
            "./bert/bert-base-japanese-v3"
        ).to(device)
    # 使用 torch.no_grad() 上下文管理器，避免梯度计算
    with torch.no_grad():
        # 将 tokens 转换为 tensor，并移动到指定设备上，然后添加一个维度
        inputs = torch.tensor(tokens).to(device).unsqueeze(0)
        # 创建 token_type_ids 和 attention_mask
        token_type_ids = torch.zeros_like(inputs).to(device)
        attention_mask = torch.ones_like(inputs).to(device)
        inputs = {
            "input_ids": inputs,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        # 使用模型进行推理，获取隐藏状态
        res = models[device](**inputs, output_hidden_states=True)
        # 取出倒数第三层的隐藏状态，并转移到 CPU 上
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()
    # 断言输入的 token ids 的长度与 word2ph 的长度相等
    assert inputs["input_ids"].shape[-1] == len(word2ph)
    # 将 word2ph 赋值给 word2phone
    word2phone = word2ph
    # 创建空列表存储每个音素的特征
    phone_level_feature = []
    # 遍历 word2phone，将每个特征重复 word2phone[i] 次，并添加到列表中
    for i in range(len(word2phone)):
        repeat_feature = res[i].repeat(word2phone[i], 1)
        phone_level_feature.append(repeat_feature)
    # 将列表中的特征拼接起来，并转置
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T

```