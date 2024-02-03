# `stable-diffusion-webui\modules\sd_hijack_clip_old.py`

```py
# 从 modules 模块中导入 sd_hijack_clip 和 shared 模块
from modules import sd_hijack_clip
from modules import shared

# 定义一个处理文本的函数，接受一个 FrozenCLIPEmbedderWithCustomWordsBase 对象和文本列表作为参数
def process_text_old(self: sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase, texts):
    # 获取对象中的 id_start 和 id_end 属性
    id_start = self.id_start
    id_end = self.id_end
    # 获取对象中 wrapped 属性的 max_length 属性值，赋给 maxlen
    maxlen = self.wrapped.max_length  # you get to stay at 77
    # 初始化一个空列表用于存放使用的自定义词汇
    used_custom_terms = []
    # 初始化一个空列表用于存放重新生成的批量 tokens
    remade_batch_tokens = []
    # 初始化一个空列表用于存放 hijack_comments
    hijack_comments = []
    # 初始化一个空列表用于存放 hijack_fixes
    hijack_fixes = []
    # 初始化 token_count 为 0
    token_count = 0

    # 初始化一个空字典用于缓存数据
    cache = {}
    # 调用对象的 tokenize 方法处理文本列表，返回 batch_tokens
    batch_tokens = self.tokenize(texts)
    # 初始化一个空列表用于存放 batch_multipliers
    batch_multipliers = []
    # 返回 batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

    return batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count

# 定义一个处理文本的函数，接受一个 FrozenCLIPEmbedderWithCustomWordsBase 对象和文本列表作为参数
def forward_old(self: sd_hijack_clip.FrozenCLIPEmbedderWithCustomWordsBase, texts):
    # 调用 process_text_old 函数处理文本列表，返回相应的结果
    batch_multipliers, remade_batch_tokens, used_custom_terms, hijack_comments, hijack_fixes, token_count = process_text_old(self, texts)

    # 将 hijack_comments 添加到对象的 hijack 属性中的 comments 列表中
    self.hijack.comments += hijack_comments

    # 如果 used_custom_terms 不为空
    if used_custom_terms:
        # 将 used_custom_terms 中的词汇和校验和拼接成字符串，添加到对象的 hijack 属性中的 comments 列表中
        embedding_names = ", ".join(f"{word} [{checksum}]" for word, checksum in used_custom_terms)
        self.hijack.comments.append(f"Used embeddings: {embedding_names}")

    # 将 hijack_fixes 赋给对象的 hijack 属性中的 fixes 属性
    self.hijack.fixes = hijack_fixes
    # 调用对象的 process_tokens 方法处理 remade_batch_tokens 和 batch_multipliers，返回结果
    return self.process_tokens(remade_batch_tokens, batch_multipliers)
```