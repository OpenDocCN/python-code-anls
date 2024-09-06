# `.\HippoRAG\src\data_process\util.py`

```py
# 将文本语料分块成较小的部分
def chunk_corpus(corpus: list, chunk_size: int = 64) -> list:
    """
    将语料分块成较小的部分。运行以下命令下载所需的 nltk 数据：
    python -c "import nltk; nltk.download('punkt')"

    @param corpus: 格式化的语料，详见 README.md
    @param chunk_size: 每个块的大小，即每块中的词数
    @return: 分块后的语料，返回一个列表
    """
    # 从 nltk 库中导入句子和单词分词工具
    from nltk.tokenize import sent_tokenize, word_tokenize

    # 初始化一个空列表用于存储分块后的语料
    new_corpus = []
    # 遍历每个文档
    for p in corpus:
        # 提取文本内容
        text = p['text']
        # 提取索引，如果不存在 'idx' 则使用 '_id'
        idx = p['idx'] if 'idx' in p else p['_id']
        # 提取标题
        title = p['title']

        # 将文本内容分割成句子
        sentences = sent_tokenize(text)
        # 初始化当前块和当前块的大小
        current_chunk = []
        current_chunk_size = 0

        # 初始化块索引
        chunk_idx = 0
        # 遍历每个句子
        for sentence in sentences:
            # 将句子分割成单词
            words = word_tokenize(sentence)
            # 如果当前块的大小加上新句子的单词数超过块大小限制
            if current_chunk_size + len(words) > chunk_size:
                # 将当前块添加到新语料中
                new_corpus.append({
                    **p,
                    'title': title,
                    'text': " ".join(current_chunk),
                    'idx': idx + f"_{chunk_idx}",
                })
                # 重新开始新块
                current_chunk = words
                current_chunk_size = len(words)
                chunk_idx += 1
            else:
                # 将单词添加到当前块
                current_chunk.extend(words)
                current_chunk_size += len(words)

        # 如果当前块还有剩余的单词
        if current_chunk:  # there are still some words left
            # 将剩余的块添加到新语料中
            new_corpus.append({
                **p,
                'title': title,
                'text': " ".join(current_chunk),
                'idx': f"{idx}_{chunk_idx}",
            })

    # 返回分块后的语料列表
    return new_corpus


# 将分块的评分合并到原始段落中
def merge_chunk_scores(id_score: dict):
    """
    将块的评分合并到原始段落中
    @param id_score: 一个字典，键是段落 id（块 id，字符串），值是评分（浮点数）
    @return: 一个合并后的字典，键是原始段落 id（字符串），值是评分（浮点数）
    """
    # 初始化一个空字典用于存储合并后的评分
    merged_scores = {}
    # 遍历评分字典中的每个项
    for passage_id, score in id_score.items():
        # 仅保留原始段落的 id（去掉块 id 部分）
        passage_id = passage_id.split('_')[0]
        # 如果原始段落 id 不在合并后的评分字典中
        if passage_id not in merged_scores:
            # 初始化该段落的评分
            merged_scores[passage_id] = 0
        # 记录该段落的最大评分
        merged_scores[passage_id] = max(merged_scores[passage_id], score)
    # 返回合并后的评分字典
    return merged_scores


# 将分块的语料合并回原始段落
def merge_chunks(corpus: list):
    """
    将分块的语料合并回原始段落
    @param corpus: 一个段落列表
    @return: 合并后的语料，字典形式
    """

    # 初始化一个空字典用于存储合并后的段落
    new_corpus = {}
    # 遍历每个文档
    for p in corpus:
        # 提取文档的索引
        idx = p['idx']
        # 如果索引中不包含下划线，表示这是原始段落
        if '_' not in idx:
            # 直接将原始段落添加到新语料中
            new_corpus[idx] = p
        else:
            # 提取原始段落的 id
            original_idx = idx.split('_')[0]
            # 如果原始段落 id 不在新语料中
            if original_idx not in new_corpus:
                # 初始化该段落，并设置文本和索引
                new_corpus[original_idx] = {
                    **p,
                    'text': p['text'],
                    'idx': original_idx,
                }
            else:
                # 将当前块的文本追加到已存在的原始段落中
                new_corpus[original_idx]['text'] += ' ' + p['text']

    # 返回合并后的语料列表
    return list(new_corpus.values())
```