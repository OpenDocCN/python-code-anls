# `D:\src\scipysrc\scikit-learn\benchmarks\bench_text_vectorizers.py`

```
"""

To run this benchmark, you will need,

 * scikit-learn
 * pandas
 * memory_profiler
 * psutil (optional, but recommended)

"""

# 导入必要的库和模块
import itertools  # 导入 itertools 库，用于迭代操作
import timeit  # 导入 timeit 库，用于性能计时

import numpy as np  # 导入 NumPy 库，用于数值计算
import pandas as pd  # 导入 pandas 库，用于数据处理
from memory_profiler import memory_usage  # 导入 memory_usage 函数，用于内存使用情况监测

# 导入 scikit-learn 的相关模块和函数
from sklearn.datasets import fetch_20newsgroups  # 导入 fetch_20newsgroups 函数，获取新闻数据集
from sklearn.feature_extraction.text import (  # 导入文本特征提取相关的类
    CountVectorizer,  # 词频向量化
    HashingVectorizer,  # 哈希向量化
    TfidfVectorizer,  # TF-IDF 向量化
)

n_repeat = 3  # 设置重复运行次数为 3


def run_vectorizer(Vectorizer, X, **params):
    """
    定义运行特定向量化器的函数

    Parameters:
    Vectorizer : class
        特定的向量化器类
    X : list
        输入的文本数据
    params : dict
        向量化器的参数

    Returns:
    function
        返回一个函数对象，用于运行向量化器
    """
    def f():
        vect = Vectorizer(**params)  # 初始化向量化器对象
        vect.fit_transform(X)  # 拟合并转换输入数据 X

    return f


text = fetch_20newsgroups(subset="train").data[:1000]  # 获取新闻数据集的前 1000 条文本数据

print("=" * 80 + "\n#" + "    Text vectorizers benchmark" + "\n" + "=" * 80 + "\n")
print("Using a subset of the 20 newsgroups dataset ({} documents).".format(len(text)))
print("This benchmarks runs in ~1 min ...")

res = []  # 初始化结果列表，用于存储性能指标

# 使用 itertools.product 进行向量化器和参数的组合迭代
for Vectorizer, (analyzer, ngram_range) in itertools.product(
    [CountVectorizer, TfidfVectorizer, HashingVectorizer],  # 向量化器列表
    [("word", (1, 1)), ("word", (1, 2)), ("char", (4, 4)), ("char_wb", (4, 4))],  # 参数组合
):
    bench = {"vectorizer": Vectorizer.__name__}  # 初始化当前向量化器的性能指标字典
    params = {"analyzer": analyzer, "ngram_range": ngram_range}  # 当前向量化器的参数
    bench.update(params)  # 更新性能指标字典

    # 计算运行时间
    dt = timeit.repeat(
        run_vectorizer(Vectorizer, text, **params), number=1, repeat=n_repeat
    )
    bench["time"] = "{:.3f} (+-{:.3f})".format(np.mean(dt), np.std(dt))  # 计算并记录平均时间和标准差

    # 计算内存使用情况
    mem_usage = memory_usage(run_vectorizer(Vectorizer, text, **params))
    bench["memory"] = "{:.1f}".format(np.max(mem_usage))  # 记录最大内存使用量

    res.append(bench)  # 将当前性能指标字典添加到结果列表中

df = pd.DataFrame(res).set_index(["analyzer", "ngram_range", "vectorizer"])  # 创建性能指标的 DataFrame

print("\n========== Run time performance (sec) ===========\n")
print(
    "Computing the mean and the standard deviation "
    "of the run time over {} runs...\n".format(n_repeat)
)
print(df["time"].unstack(level=-1))  # 打印运行时间的统计结果

print("\n=============== Memory usage (MB) ===============\n")
print(df["memory"].unstack(level=-1))  # 打印内存使用量的统计结果
```