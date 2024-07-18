# `.\graphrag\graphrag\index\bootstrap.py`

```py
# Copyright (c) 2024 Microsoft Corporation.
# Licensed under the MIT License

"""Bootstrap definition."""

# 导入警告模块
import warnings

# 忽略 numba 发出的警告
warnings.filterwarnings("ignore", message=".*The 'nopython' keyword.*")
warnings.filterwarnings("ignore", message=".*Use no seed for parallelism.*")

# 初始化 NLTK 标志
initialized_nltk = False

# 定义引导函数
def bootstrap():
    """Bootstrap definition."""
    # 声明全局变量
    global initialized_nltk
    # 如果 NLTK 尚未初始化
    if not initialized_nltk:
        # 导入 NLTK 库和相关模块
        import nltk
        from nltk.corpus import wordnet as wn

        # 下载 NLTK 所需的数据集
        nltk.download("punkt")
        nltk.download("averaged_perceptron_tagger")
        nltk.download("maxent_ne_chunker")
        nltk.download("words")
        nltk.download("wordnet")
        
        # 确保 WordNet 数据已加载
        wn.ensure_loaded()
        
        # 设置 NLTK 已初始化标志为 True
        initialized_nltk = True
```