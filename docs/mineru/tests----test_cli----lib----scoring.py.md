# `.\MinerU\tests\test_cli\lib\scoring.py`

```
# 计算相似度分数，参考 (https://github.com/VikParuchuri/marker?tab=readme-ov-file)
"""
Calculate simscore, refer to (https://github.com/VikParuchuri/marker?tab=readme-ov-file)
"""
# 导入数学库
import math

# 从 rapidfuzz 导入模糊匹配函数
from rapidfuzz import fuzz
# 导入正则表达式库
import re
# 导入高级正则表达式库
import regex
# 导入计算均值的函数
from statistics import mean

# 定义最小字符块长度
CHUNK_MIN_CHARS = 25

# 将文本分块，默认为每块500个字符
def chunk_text(text, chunk_len=500):
    # 将文本按指定长度切分为块
    chunks = [text[i:i+chunk_len] for i in range(0, len(text), chunk_len)]
    # 过滤掉空块和长度小于最小字符块的块
    chunks = [c for c in chunks if c.strip() and len(c) > CHUNK_MIN_CHARS]
    # 返回有效块
    return chunks


# 计算假设块与参考块的重叠分数
def overlap_score(hypothesis_chunks, reference_chunks):
    # 如果参考块不为空，计算长度修正因子
    if len(reference_chunks) > 0:
        length_modifier = len(hypothesis_chunks) / len(reference_chunks)
    else:
        # 否则设置修正因子为0
        length_modifier = 0
    # 设定搜索距离
    search_distance = max(len(reference_chunks) // 5, 10)
    # 初始化块分数列表
    chunk_scores = []
    # 遍历每个假设块
    for i, hyp_chunk in enumerate(hypothesis_chunks):
        # 初始化最大分数和总长度
        max_score = 0
        total_len = 0
        # 计算偏移量
        i_offset = int(i * length_modifier)
        # 计算参考块的搜索范围
        chunk_range = range(max(0, i_offset-search_distance), min(len(reference_chunks), i_offset+search_distance))
        # 遍历参考块范围
        for j in chunk_range:
            ref_chunk = reference_chunks[j]
            # 计算模糊匹配分数
            score = fuzz.ratio(hyp_chunk, ref_chunk, score_cutoff=30) / 100
            # 更新最大分数和总长度
            if score > max_score:
                max_score = score
                total_len = len(ref_chunk)
        # 添加当前块的最大分数
        chunk_scores.append(max_score)
    # 返回所有块的分数
    return chunk_scores


# 计算文本的对齐分数，范围在0到1之间
def score_text(hypothesis, reference):
    # 将假设和参考文本切分为块
    hypothesis_chunks = chunk_text(hypothesis)
    reference_chunks = chunk_text(reference)
    # 计算块的重叠分数
    chunk_scores = overlap_score(hypothesis_chunks, reference_chunks)
    # 如果存在块分数，计算平均分数并返回
    if len(chunk_scores) > 0:
        mean_score = mean(chunk_scores)
        return mean_score
    else:
        # 否则返回0
        return 0
    #return mean(chunk(scores)
```