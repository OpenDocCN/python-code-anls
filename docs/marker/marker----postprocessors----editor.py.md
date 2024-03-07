# `.\marker\marker\postprocessors\editor.py`

```py
# 导入必要的库
from collections import defaultdict, Counter
from itertools import chain
from typing import Optional

# 导入 transformers 库中的 AutoTokenizer 类
from transformers import AutoTokenizer

# 导入 settings 模块中的 settings 变量
from marker.settings import settings

# 导入 torch 库
import torch
import torch.nn.functional as F

# 导入 marker.postprocessors.t5 模块中的 T5ForTokenClassification 类和 byt5_tokenize 函数
from marker.postprocessors.t5 import T5ForTokenClassification, byt5_tokenize

# 定义加载编辑模型的函数
def load_editing_model():
    # 如果未启用编辑模型，则返回 None
    if not settings.ENABLE_EDITOR_MODEL:
        return None

    # 从预训练模型中加载 T5ForTokenClassification 模型
    model = T5ForTokenClassification.from_pretrained(
            settings.EDITOR_MODEL_NAME,
            torch_dtype=settings.MODEL_DTYPE,
        ).to(settings.TORCH_DEVICE_MODEL)
    model.eval()

    # 配置模型的标签映射
    model.config.label2id = {
        "equal": 0,
        "delete": 1,
        "newline-1": 2,
        "space-1": 3,
    }
    model.config.id2label = {v: k for k, v in model.config.label2id.items()}
    return model

# 定义编辑全文的函数
def edit_full_text(text: str, model: Optional[T5ForTokenClassification], batch_size: int = settings.EDITOR_BATCH_SIZE):
    # 如果模型为空，则直接返回原始文本和空字典
    if not model:
        return text, {}

    # 对文本进行 tokenization
    tokenized = byt5_tokenize(text, settings.EDITOR_MAX_LENGTH)
    input_ids = tokenized["input_ids"]
    char_token_lengths = tokenized["char_token_lengths"]

    # 准备 token_masks 列表
    token_masks = []
    # 遍历输入的 input_ids，按照 batch_size 进行分批处理
    for i in range(0, len(input_ids), batch_size):
        # 从 tokenized 中获取当前 batch 的 input_ids
        batch_input_ids = tokenized["input_ids"][i: i + batch_size]
        # 将 batch_input_ids 转换为 torch 张量，并指定设备为 model 的设备
        batch_input_ids = torch.tensor(batch_input_ids, device=model.device)
        # 从 tokenized 中获取当前 batch 的 attention_mask
        batch_attention_mask = tokenized["attention_mask"][i: i + batch_size]
        # 将 batch_attention_mask 转换为 torch 张量，并指定设备为 model 的设备
        batch_attention_mask = torch.tensor(batch_attention_mask, device=model.device)
        
        # 进入推理模式
        with torch.inference_mode():
            # 使用模型进行预测
            predictions = model(batch_input_ids, attention_mask=batch_attention_mask)

        # 将预测结果 logits 移动到 CPU 上
        logits = predictions.logits.cpu()

        # 如果最大概率小于阈值，则假设为不良预测
        # 我们希望保守一点，不要对文本进行过多编辑
        probs = F.softmax(logits, dim=-1)
        max_prob = torch.max(probs, dim=-1)
        cutoff_prob = max_prob.values < settings.EDITOR_CUTOFF_THRESH
        labels = logits.argmax(-1)
        labels[cutoff_prob] = model.config.label2id["equal"]
        labels = labels.squeeze().tolist()
        if len(labels) == settings.EDITOR_MAX_LENGTH:
            labels = [labels]
        labels = list(chain.from_iterable(labels))
        token_masks.extend(labels)

    # 文本中的字符列表
    flat_input_ids = list(chain.from_iterable(input_ids)

    # 去除特殊标记 0,1。保留未知标记，尽管它不应该被使用
    assert len(token_masks) == len(flat_input_ids)
    token_masks = [mask for mask, token in zip(token_masks, flat_input_ids) if token >= 2]

    # 确保 token_masks 的长度与文本编码后的长度相等
    assert len(token_masks) == len(list(text.encode("utf-8")))

    # 统计编辑次数的字典
    edit_stats = defaultdict(int)
    # 输出文本列表
    out_text = []
    # 起始位置
    start = 0
    # 遍历文本中的每个字符及其索引
    for i, char in enumerate(text):
        # 获取当前字符对应的 token 长度
        char_token_length = char_token_lengths[i]
        # 获取当前字符对应的 token 的 mask
        masks = token_masks[start: start + char_token_length]
        # 将 mask 转换为标签
        labels = [model.config.id2label[mask] for mask in masks]
        # 如果所有标签都是 "delete"，则执行删除操作
        if all(l == "delete" for l in labels):
            # 如果删除的是空格，则保留，否则忽略
            if char.strip():
                out_text.append(char)
            else:
                edit_stats["delete"] += 1
        # 如果标签为 "newline-1"，则添加换行符
        elif labels[0] == "newline-1":
            out_text.append("\n")
            out_text.append(char)
            edit_stats["newline-1"] += 1
        # 如果标签为 "space-1"，则添加空格
        elif labels[0] == "space-1":
            out_text.append(" ")
            out_text.append(char)
            edit_stats["space-1"] += 1
        # 如果标签为其他情况，则保留字符
        else:
            out_text.append(char)
            edit_stats["equal"] += 1

        # 更新下一个字符的起始位置
        start += char_token_length

    # 将处理后的文本列表转换为字符串
    out_text = "".join(out_text)
    # 返回处理后的文本及编辑统计信息
    return out_text, edit_stats
```