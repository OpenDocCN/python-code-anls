# `D:\src\scipysrc\scikit-learn\examples\applications\plot_out_of_core_classification.py`

```
"""
======================================================
Out-of-core classification of text documents
======================================================

This is an example showing how scikit-learn can be used for classification
using an out-of-core approach: learning from data that doesn't fit into main
memory. We make use of an online classifier, i.e., one that supports the
partial_fit method, that will be fed with batches of examples. To guarantee
that the features space remains the same over time we leverage a
HashingVectorizer that will project each example into the same feature space.
This is especially useful in the case of text classification where new
features (words) may appear in each batch.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import itertools
import re
import sys
import tarfile
import time
from hashlib import sha256
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

from sklearn.datasets import get_data_home
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier, Perceptron, SGDClassifier
from sklearn.naive_bayes import MultinomialNB


def _not_in_sphinx():
    # Hack to detect whether we are running by the sphinx builder
    return "__file__" in globals()


# %%
# Reuters Dataset related routines
# --------------------------------
#
# The dataset used in this example is Reuters-21578 as provided by the UCI ML
# repository. It will be automatically downloaded and uncompressed on first
# run.


class ReutersParser(HTMLParser):
    """Utility class to parse a SGML file and yield documents one at a time."""

    def __init__(self, encoding="latin-1"):
        HTMLParser.__init__(self)
        self._reset()
        self.encoding = encoding

    def handle_starttag(self, tag, attrs):
        method = "start_" + tag
        getattr(self, method, lambda x: None)(attrs)

    def handle_endtag(self, tag):
        method = "end_" + tag
        getattr(self, method, lambda: None)()

    def _reset(self):
        self.in_title = 0
        self.in_body = 0
        self.in_topics = 0
        self.in_topic_d = 0
        self.title = ""
        self.body = ""
        self.topics = []
        self.topic_d = ""

    def parse(self, fd):
        self.docs = []
        for chunk in fd:
            self.feed(chunk.decode(self.encoding))
            for doc in self.docs:
                yield doc
            self.docs = []
        self.close()

    def handle_data(self, data):
        if self.in_body:
            self.body += data
        elif self.in_title:
            self.title += data
        elif self.in_topic_d:
            self.topic_d += data

    def start_reuters(self, attributes):
        pass
    # 结束 Reuters XML 解析，清除正文中多余的空白字符，将标题、正文和主题组成的字典追加到文档列表中
    def end_reuters(self):
        self.body = re.sub(r"\s+", r" ", self.body)  # 将正文中多余的空白字符替换为单个空格
        self.docs.append(
            {"title": self.title, "body": self.body, "topics": self.topics}
        )  # 将当前标题、正文和主题列表作为一个字典添加到文档列表中
        self._reset()  # 重置对象的状态，准备处理下一个文档
    
    # 开始解析标题标签，设置标志位表示当前处于标题标签内
    def start_title(self, attributes):
        self.in_title = 1
    
    # 结束解析标题标签，重置标题标签标志位
    def end_title(self):
        self.in_title = 0
    
    # 开始解析正文标签，设置标志位表示当前处于正文标签内
    def start_body(self, attributes):
        self.in_body = 1
    
    # 结束解析正文标签，重置正文标签标志位
    def end_body(self):
        self.in_body = 0
    
    # 开始解析主题标签，设置标志位表示当前处于主题标签内
    def start_topics(self, attributes):
        self.in_topics = 1
    
    # 结束解析主题标签，重置主题标签标志位
    def end_topics(self):
        self.in_topics = 0
    
    # 开始解析主题描述标签，设置标志位表示当前处于主题描述标签内
    def start_d(self, attributes):
        self.in_topic_d = 1
    
    # 结束解析主题描述标签，重置主题描述标签标志位，并将解析到的主题描述添加到主题列表中，同时重置当前主题描述字符串
    def end_d(self):
        self.in_topic_d = 0
        self.topics.append(self.topic_d)  # 将当前主题描述添加到主题列表中
        self.topic_d = ""  # 重置当前主题描述字符串
# 定义一个函数，从 Reuters 数据集中流式处理文档
def stream_reuters_documents(data_path=None):
    """Iterate over documents of the Reuters dataset.

    The Reuters archive will automatically be downloaded and uncompressed if
    the `data_path` directory does not exist.

    Documents are represented as dictionaries with 'body' (str),
    'title' (str), 'topics' (list(str)) keys.

    """

    # Reuters 数据集的下载链接
    DOWNLOAD_URL = (
        "http://archive.ics.uci.edu/ml/machine-learning-databases/"
        "reuters21578-mld/reuters21578.tar.gz"
    )
    # Reuters 数据集压缩文件的 SHA256 校验和
    ARCHIVE_SHA256 = "3bae43c9b14e387f76a61b6d82bf98a4fb5d3ef99ef7e7075ff2ccbcf59f9d30"
    # Reuters 数据集压缩文件名
    ARCHIVE_FILENAME = "reuters21578.tar.gz"

    # 如果未指定 data_path，则设置默认的数据存储路径
    if data_path is None:
        data_path = Path(get_data_home()) / "reuters"
    else:
        data_path = Path(data_path)
    
    # 如果数据路径不存在，则下载数据集
    if not data_path.exists():
        """Download the dataset."""
        print("downloading dataset (once and for all) into %s" % data_path)
        # 创建数据路径，并确保其父目录存在
        data_path.mkdir(parents=True, exist_ok=True)

        # 定义下载进度显示函数
        def progress(blocknum, bs, size):
            total_sz_mb = "%.2f MB" % (size / 1e6)
            current_sz_mb = "%.2f MB" % ((blocknum * bs) / 1e6)
            if _not_in_sphinx():
                sys.stdout.write("\rdownloaded %s / %s" % (current_sz_mb, total_sz_mb))

        # 设置下载的压缩文件路径
        archive_path = data_path / ARCHIVE_FILENAME

        # 使用 urlretrieve 下载数据集，并显示下载进度
        urlretrieve(DOWNLOAD_URL, filename=archive_path, reporthook=progress)
        if _not_in_sphinx():
            sys.stdout.write("\r")

        # 检查下载的压缩文件是否完整，通过比对 SHA256 校验和
        assert sha256(archive_path.read_bytes()).hexdigest() == ARCHIVE_SHA256

        # 解压下载的压缩文件
        print("untarring Reuters dataset...")
        with tarfile.open(archive_path, "r:gz") as fp:
            fp.extractall(data_path, filter="data")
        print("done.")

    # 创建 ReutersParser 对象用于解析数据集
    parser = ReutersParser()
    
    # 遍历数据路径下所有的 .sgm 文件，并解析每个文件中的文档
    for filename in data_path.glob("*.sgm"):
        for doc in parser.parse(open(filename, "rb")):
            yield doc


# %%
# Main
# ----
#
# 创建 HashingVectorizer 对象作为文本向量化器，并限制特征数量在合理范围内

vectorizer = HashingVectorizer(
    decode_error="ignore", n_features=2**18, alternate_sign=False
)


# 从 stream_reuters_documents 函数返回的迭代器获取数据流
data_stream = stream_reuters_documents()

# 进行二元分类，目标类为 "acq"，其它类别为负类
# "acq" 类别被选择因为它在 Reuters 文件中分布较均匀
# 对于其他数据集，需要根据实际情况创建包含足够正实例的测试集
all_classes = np.array([0, 1])
positive_class = "acq"

# 定义支持 partial_fit 方法的分类器
partial_fit_classifiers = {
    "SGD": SGDClassifier(max_iter=5),
    "Perceptron": Perceptron(),
    "NB Multinomial": MultinomialNB(alpha=0.01),
    "Passive-Aggressive": PassiveAggressiveClassifier(),
}


def get_minibatch(doc_iter, size, pos_class=positive_class):
    """Extract a minibatch of examples, return a tuple X_text, y.

    Note: size is before excluding invalid docs with no topics assigned.

    """
    # 使用列表推导式生成包含元组的列表，每个元组包含格式化后的文档标题和正文组成的字符串以及布尔值
    data = [
        ("{title}\n\n{body}".format(**doc), pos_class in doc["topics"])  # 格式化文档标题和正文，并检查正类是否在文档主题中
        for doc in itertools.islice(doc_iter, size)  # 使用迭代器切片取出指定数量的文档
        if doc["topics"]  # 过滤掉主题为空的文档
    ]
    # 如果生成的 data 列表长度为 0，返回空的 NumPy 数组作为 X_text 和 y
    if not len(data):
        return np.asarray([], dtype=int), np.asarray([], dtype=int)
    # 将 data 列表中的元组拆分为两个分别是 X_text 和 y 的列表
    X_text, y = zip(*data)
    # 返回拆分后的 X_text 和转换为 NumPy 数组的 y
    return X_text, np.asarray(y, dtype=int)
# 定义一个生成器函数，用于生成文档的小批量数据
def iter_minibatches(doc_iter, minibatch_size):
    # 调用函数获取一个小批量的文本数据和对应的标签
    X_text, y = get_minibatch(doc_iter, minibatch_size)
    # 当还有文本数据时，持续生成小批量数据
    while len(X_text):
        # 生成当前的文本数据和标签
        yield X_text, y
        # 继续获取下一个小批量的文本数据和标签
        X_text, y = get_minibatch(doc_iter, minibatch_size)

# test_stats用于存储测试数据的统计信息，初始值为零
test_stats = {"n_test": 0, "n_test_pos": 0}

# 首先从数据流中获取一定数量的测试文档，用于评估分类器的准确率
n_test_documents = 1000
tick = time.time()
# 获取测试数据的文本和标签
X_test_text, y_test = get_minibatch(data_stream, 1000)
# 计算解析数据所用的时间
parsing_time = time.time() - tick
tick = time.time()
# 将测试数据文本转换成向量表示
X_test = vectorizer.transform(X_test_text)
# 计算向量化所用的时间
vectorizing_time = time.time() - tick
# 更新测试数据的统计信息
test_stats["n_test"] += len(y_test)
test_stats["n_test_pos"] += sum(y_test)
# 打印测试集的信息，包括文档数量和正例数量
print("Test set is %d documents (%d positive)" % (len(y_test), sum(y_test)))

def progress(cls_name, stats):
    """报告分类器的训练进度信息，返回一个字符串。"""
    # 计算报告信息的持续时间
    duration = time.time() - stats["t0"]
    s = "%20s classifier : \t" % cls_name
    # 格式化输出训练文档数和正例数的报告信息
    s += "%(n_train)6d train docs (%(n_train_pos)6d positive) " % stats
    # 格式化输出测试文档数和正例数的报告信息
    s += "%(n_test)6d test docs (%(n_test_pos)6d positive) " % test_stats
    # 格式化输出准确率的报告信息
    s += "accuracy: %(accuracy).3f " % stats
    # 格式化输出训练速度的报告信息
    s += "in %.2fs (%5d docs/s)" % (duration, stats["n_train"] / duration)
    return s

# 为每个分类器初始化统计信息字典
cls_stats = {}

for cls_name in partial_fit_classifiers:
    # 初始化分类器的统计信息
    stats = {
        "n_train": 0,
        "n_train_pos": 0,
        "accuracy": 0.0,
        "accuracy_history": [(0, 0)],
        "t0": time.time(),
        "runtime_history": [(0, 0)],
        "total_fit_time": 0.0,
    }
    # 将分类器的统计信息存入cls_stats字典中
    cls_stats[cls_name] = stats

# 获取数据流中的测试文档，丢弃测试集
get_minibatch(data_stream, n_test_documents)

# 将分类器训练的批次大小设为1000
minibatch_size = 1000

# 创建一个数据流，解析Reuters SGML文件并以流的形式迭代文档
minibatch_iterators = iter_minibatches(data_stream, minibatch_size)
# 初始化向量化操作的总时间
total_vect_time = 0.0

# 主循环：迭代小批量的样例
for i, (X_train_text, y_train) in enumerate(minibatch_iterators):
    tick = time.time()
    # 将训练文本数据转换成向量表示
    X_train = vectorizer.transform(X_train_text)
    # 计算向量化操作所用的时间并累加到总时间中
    total_vect_time += time.time() - tick
   `
    for cls_name, cls in partial_fit_classifiers.items():
        tick = time.time()
        # 使用部分拟合方法，更新当前小批量中的估算器（模型）
        cls.partial_fit(X_train, y_train, classes=all_classes)

        # 累积测试精度统计信息
        cls_stats[cls_name]["total_fit_time"] += time.time() - tick
        # 累积训练样本数
        cls_stats[cls_name]["n_train"] += X_train.shape[0]
        # 累积训练样本中正例数
        cls_stats[cls_name]["n_train_pos"] += sum(y_train)
        tick = time.time()
        # 计算当前估算器（模型）在测试集上的准确率
        cls_stats[cls_name]["accuracy"] = cls.score(X_test, y_test)
        # 计算预测时间
        cls_stats[cls_name]["prediction_time"] = time.time() - tick
        # 将当前准确率和训练样本数添加到准确率历史记录中
        acc_history = (cls_stats[cls_name]["accuracy"], cls_stats[cls_name]["n_train"])
        cls_stats[cls_name]["accuracy_history"].append(acc_history)
        # 将当前准确率和总向量化时间加上总拟合时间添加到运行时历史记录中
        run_history = (
            cls_stats[cls_name]["accuracy"],
            total_vect_time + cls_stats[cls_name]["total_fit_time"],
        )
        cls_stats[cls_name]["runtime_history"].append(run_history)

        # 每处理3次小批量数据就打印进度信息
        if i % 3 == 0:
            print(progress(cls_name, cls_stats[cls_name]))
    # 每处理3次小批量数据就打印空行
    if i % 3 == 0:
        print("\n")
# %%
# Plot results
# ------------
#
# The following code block plots various metrics related to the classifier's performance:
# accuracy evolution over training examples, accuracy evolution over runtime, training times,
# and prediction times. Each plot provides insights into different aspects of the classifier's behavior.

def plot_accuracy(x, y, x_legend):
    """Plot accuracy as a function of x."""
    # Convert inputs to numpy arrays for plotting
    x = np.array(x)
    y = np.array(y)
    
    # Set up plot parameters
    plt.title("Classification accuracy as a function of %s" % x_legend)
    plt.xlabel("%s" % x_legend)
    plt.ylabel("Accuracy")
    plt.grid(True)
    
    # Plot the data
    plt.plot(x, y)


# Adjust font size for legend
rcParams["legend.fontsize"] = 10

# Get sorted list of classifier names
cls_names = list(sorted(cls_stats.keys()))

# Plot accuracy evolution over training examples
plt.figure()
for _, stats in sorted(cls_stats.items()):
    accuracy, n_examples = zip(*stats["accuracy_history"])
    plot_accuracy(n_examples, accuracy, "training examples (#)")
    ax = plt.gca()
    ax.set_ylim((0.8, 1))
plt.legend(cls_names, loc="best")

# Plot accuracy evolution over runtime
plt.figure()
for _, stats in sorted(cls_stats.items()):
    accuracy, runtime = zip(*stats["runtime_history"])
    plot_accuracy(runtime, accuracy, "runtime (s)")
    ax = plt.gca()
    ax.set_ylim((0.8, 1))
plt.legend(cls_names, loc="best")

# Plot training times
plt.figure()
fig = plt.gcf()
cls_runtime = [stats["total_fit_time"] for cls_name, stats in sorted(cls_stats.items())]
cls_runtime.append(total_vect_time)
cls_names.append("Vectorization")
bar_colors = ["b", "g", "r", "c", "m", "y"]

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5, color=bar_colors)

# Customize x-axis labels
ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=10)
ymax = max(cls_runtime) * 1.2
ax.set_ylim((0, ymax))
ax.set_ylabel("runtime (s)")
ax.set_title("Training Times")

# Function to label bars with their values
def autolabel(rectangles):
    """Attach text labels to rectangles."""
    for rect in rectangles:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            1.05 * height,
            "%.4f" % height,
            ha="center",
            va="bottom",
        )
        plt.setp(plt.xticks()[1], rotation=30)

autolabel(rectangles)
plt.tight_layout()
plt.show()

# Plot prediction times
plt.figure()
cls_runtime = []
cls_names = list(sorted(cls_stats.keys()))
for cls_name, stats in sorted(cls_stats.items()):
    cls_runtime.append(stats["prediction_time"])
cls_runtime.append(parsing_time)
cls_names.append("Read/Parse\n+Feat.Extr.")
cls_runtime.append(vectorizing_time)
cls_names.append("Hashing\n+Vect.")

ax = plt.subplot(111)
rectangles = plt.bar(range(len(cls_names)), cls_runtime, width=0.5, color=bar_colors)

# Customize x-axis labels
ax.set_xticks(np.linspace(0, len(cls_names) - 1, len(cls_names)))
ax.set_xticklabels(cls_names, fontsize=8)
plt.setp(plt.xticks()[1], rotation=30)
# 计算分类器运行时间的最大值，并将其增加20%作为纵轴上限
ymax = max(cls_runtime) * 1.2
# 设置图表的纵轴范围为0到计算得到的ymax
ax.set_ylim((0, ymax))
# 设置纵轴标签为"runtime (s)"
ax.set_ylabel("runtime (s)")
# 设置图表标题，显示测试文档实例数
ax.set_title("Prediction Times (%d instances)" % n_test_documents)
# 调用自定义函数autolabel，为每个条形图添加标签
autolabel(rectangles)
# 调整子图布局以确保所有元素适当显示
plt.tight_layout()
# 显示绘制的图表
plt.show()
```