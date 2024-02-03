# `jieba\test\extract_topic.py`

```
# 导入 sys 模块
import sys
# 将上级目录添加到 sys.path 中
sys.path.append("../")
# 导入 CountVectorizer 和 TfidfTransformer 类
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# 导入 decomposition 模块
from sklearn import decomposition

# 导入 jieba、time、glob、os、random 模块
import jieba
import time
import glob
import sys
import os
import random

# 检查命令行参数是否足够
if len(sys.argv)<2:
    print("usage: extract_topic.py directory [n_topic] [n_top_words]")
    sys.exit(0)

# 初始化主题数和关键词数
n_topic = 10
n_top_words = 25

# 根据命令行参数更新主题数
if len(sys.argv)>2:
    n_topic = int(sys.argv[2])

# 根据命令行参数更新关键词数
if len(sys.argv)>3:
    n_top_words = int(sys.argv[3])

# 创建 CountVectorizer 对象
count_vect = CountVectorizer()
# 初始化文档列表
docs = []

# 构建文件匹配模式
pattern = os.path.join(sys.argv[1],"*.txt") 
print("read "+pattern)

# 遍历匹配模式下的所有文件
for f_name in glob.glob(pattern):
    # 打开文件
    with open(f_name) as f:
        print("read file:", f_name)
        # 逐行读取文件内容
        for line in f: #one line as a document
            # 使用 jieba 分词并拼接成字符串
            words = " ".join(jieba.cut(line))
            # 将处理后的文本添加到文档列表中
            docs.append(words)

# 随机打乱文档列表
random.shuffle(docs)

print("read done.")

print("transform")
# 将文档列表转换为词频矩阵
counts = count_vect.fit_transform(docs)
# 计算 TF-IDF
tfidf = TfidfTransformer().fit_transform(counts)
print(tfidf.shape)

# 记录开始时间
t0 = time.time()
print("training...")

# 使用 NMF 进行主题建模
nmf = decomposition.NMF(n_components=n_topic).fit(tfidf)
print("done in %0.3fs." % (time.time() - t0))

# 获取特征词列表
feature_names = count_vect.get_feature_names()

# 遍历主题并输出关键词
for topic_idx, topic in enumerate(nmf.components_):
    print("Topic #%d:" % topic_idx)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print("")
```