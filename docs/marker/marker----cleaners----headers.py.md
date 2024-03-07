# `.\marker\marker\cleaners\headers.py`

```
# 导入所需的模块
import re
from collections import Counter, defaultdict
from itertools import chain
from thefuzz import fuzz
from sklearn.cluster import DBSCAN
import numpy as np
from marker.schema import Page, FullyMergedBlock
from typing import List, Tuple

# 过滤出现频率高于给定阈值的文本块
def filter_common_elements(lines, page_count):
    # 提取所有文本内容
    text = [s.text for line in lines for s in line.spans if len(s.text) > 4]
    # 统计文本内容出现的次数
    counter = Counter(text)
    # 选取出现频率高于阈值的文本内容
    common = [k for k, v in counter.items() if v > page_count * .6]
    # 获取包含常见文本内容的文本块的 span_id
    bad_span_ids = [s.span_id for line in lines for s in line.spans if s.text in common]
    return bad_span_ids

# 过滤页眉页脚文本块
def filter_header_footer(all_page_blocks, max_selected_lines=2):
    first_lines = []
    last_lines = []
    for page in all_page_blocks:
        nonblank_lines = page.get_nonblank_lines()
        first_lines.extend(nonblank_lines[:max_selected_lines])
        last_lines.extend(nonblank_lines[-max_selected_lines:])

    # 获取页眉页脚文本块的 span_id
    bad_span_ids = filter_common_elements(first_lines, len(all_page_blocks))
    bad_span_ids += filter_common_elements(last_lines, len(all_page_blocks))
    return bad_span_ids

# 对文本块进行分类
def categorize_blocks(all_page_blocks: List[Page]):
    # 提取所有非空文本块的 span
    spans = list(chain.from_iterable([p.get_nonblank_spans() for p in all_page_blocks]))
    # 构建特征矩阵
    X = np.array(
        [(*s.bbox, len(s.text)) for s in spans]
    )

    # 使用 DBSCAN 进行聚类
    dbscan = DBSCAN(eps=.1, min_samples=5)
    dbscan.fit(X)
    labels = dbscan.labels_
    label_chars = defaultdict(int)
    for i, label in enumerate(labels):
        label_chars[label] += len(spans[i].text)

    # 选择出现次数最多的类别作为主要类别
    most_common_label = None
    most_chars = 0
    for i in label_chars.keys():
        if label_chars[i] > most_chars:
            most_common_label = i
            most_chars = label_chars[i]

    # 将非主要类别标记为 1
    labels = [0 if label == most_common_label else 1 for label in labels]
    # 获取非主要类别的文本块的 span_id
    bad_span_ids = [spans[i].span_id for i in range(len(spans)) if labels[i] == 1]

    return bad_span_ids

# 替换字符串开头的数字
def replace_leading_trailing_digits(string, replacement):
    string = re.sub(r'^\d+', replacement, string)
    # 使用正则表达式替换字符串中最后的数字
    string = re.sub(r'\d+$', replacement, string)
    # 返回替换后的字符串
    return string
# 定义一个函数，用于查找重叠元素
def find_overlap_elements(lst: List[Tuple[str, int]], string_match_thresh=.9, min_overlap=.05) -> List[int]:
    # 初始化一个列表，用于存储符合条件的元素
    result = []
    # 从输入列表中提取所有元组的第一个元素，即标题
    titles = [l[0] for l in lst]

    # 遍历输入列表中的元素
    for i, (str1, id_num) in enumerate(lst):
        overlap_count = 0  # 计算至少80%重叠的元素数量

        # 再次遍历标题列表，检查元素之间的相似度
        for j, str2 in enumerate(titles):
            if i != j and fuzz.ratio(str1, str2) >= string_match_thresh * 100:
                overlap_count += 1

        # 检查元素是否与至少50%的其他元素重叠
        if overlap_count >= max(3.0, len(lst) * min_overlap):
            result.append(id_num)

    return result


# 定义一个函数，用于过滤常见标题
def filter_common_titles(merged_blocks: List[FullyMergedBlock]) -> List[FullyMergedBlock]:
    titles = []
    # 遍历合并块列表中的块
    for i, block in enumerate(merged_blocks):
        # 如果块类型为"Title"或"Section-header"
        if block.block_type in ["Title", "Section-header"]:
            text = block.text
            # 如果文本以"#"开头，则去除所有"#"
            if text.strip().startswith("#"):
                text = re.sub(r'#+', '', text)
            text = text.strip()
            # 去除文本开头和结尾的页码
            text = replace_leading_trailing_digits(text, "").strip()
            titles.append((text, i))

    # 查找重叠标题的块的索引
    bad_block_ids = find_overlap_elements(titles)

    new_blocks = []
    # 遍历合并块列表中的块
    for i, block in enumerate(merged_blocks):
        # 如果块的索引在重叠块的索引列表中，则跳过该块
        if i in bad_block_ids:
            continue
        new_blocks.append(block)

    return new_blocks
```