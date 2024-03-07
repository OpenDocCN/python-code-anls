# `.\marker\marker\benchmark\scoring.py`

```
# 导入 math 模块
import math

# 从 rapidfuzz 模块中导入 fuzz 和 distance 函数
from rapidfuzz import fuzz, distance
# 导入 re 模块
import re

# 定义最小分块字符数
CHUNK_MIN_CHARS = 25


def tokenize(text):
    # 定义正则表达式模式
    pattern = r'([^\w\s\d\'])|([\w\']+)|(\d+)|(\n+)|( +)'
    # 使用正则表达式模式匹配文本
    result = re.findall(pattern, text)
    # 将匹配结果扁平化并过滤掉空字符串
    flattened_result = [item for sublist in result for item in sublist if item]
    return flattened_result


def chunk_text(text):
    # 将文本按换行符分割成块
    chunks = text.split("\n")
    # 过滤掉空白块和长度小于最小分块字符数的块
    chunks = [c for c in chunks if c.strip() and len(c) > CHUNK_MIN_CHARS]
    return chunks


def overlap_score(hypothesis_chunks, reference_chunks):
    # 计算长度修正因子
    length_modifier = len(hypothesis_chunks) / len(reference_chunks)
    # 计算搜索距离
    search_distance = max(len(reference_chunks) // 5, 10)
    chunk_scores = []
    chunk_weights = []
    for i, hyp_chunk in enumerate(hypothesis_chunks):
        max_score = 0
        chunk_weight = 1
        i_offset = int(i * length_modifier)
        chunk_range = range(max(0, i_offset-search_distance), min(len(reference_chunks), i_offset+search_distance))
        for j in chunk_range:
            ref_chunk = reference_chunks[j]
            # 计算相似度得分
            score = fuzz.ratio(hyp_chunk, ref_chunk, score_cutoff=30) / 100
            if score > max_score:
                max_score = score
                chunk_weight = math.sqrt(len(ref_chunk))
        chunk_scores.append(max_score)
        chunk_weights.append(chunk_weight)
    chunk_scores = [chunk_scores[i] * chunk_weights[i] for i in range(len(chunk_scores))]
    return chunk_scores, chunk_weights


def score_text(hypothesis, reference):
    # 返回一个0-1的对齐分数
    hypothesis_chunks = chunk_text(hypothesis)
    reference_chunks = chunk_text(reference)
    chunk_scores, chunk_weights = overlap_score(hypothesis_chunks, reference_chunks)
    return sum(chunk_scores) / sum(chunk_weights)
```