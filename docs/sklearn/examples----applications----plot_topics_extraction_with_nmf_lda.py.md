# `D:\src\scipysrc\scikit-learn\examples\applications\plot_topics_extraction_with_nmf_lda.py`

```
# 导入所需的库和模块
from time import time
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation, MiniBatchNMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# 设定示例中使用的数据集大小和特征数
n_samples = 2000
n_features = 1000
n_components = 10
n_top_words = 20
batch_size = 128
init = "nndsvda"

# 定义函数：绘制每个主题的顶部词语
def plot_top_words(model, feature_names, n_top_words, title):
    # 创建包含10个子图的画布，每个主题一个子图
    fig, axes = plt.subplots(2, 5, figsize=(30, 15), sharex=True)
    axes = axes.flatten()
    # 对每个主题进行处理
    for topic_idx, topic in enumerate(model.components_):
        # 获取主题中权重最高的词语索引，并获取对应的词语和权重
        top_features_ind = topic.argsort()[-n_top_words:]
        top_features = feature_names[top_features_ind]
        weights = topic[top_features_ind]

        # 获取当前子图对象，绘制水平条形图
        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 30})
        ax.tick_params(axis="both", which="major", labelsize=20)
        # 隐藏图的边框
        for i in "top right left".split():
            ax.spines[i].set_visible(False)
        # 设置整体标题
        fig.suptitle(title, fontsize=40)

    # 调整子图之间的间距和整体布局
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    # 显示绘制的图形
    plt.show()

# 加载20个新闻组数据集并进行预处理
print("Loading dataset...")
t0 = time()
data, _ = fetch_20newsgroups(
    shuffle=True,
    random_state=1,
    remove=("headers", "footers", "quotes"),
    return_X_y=True,
)
data_samples = data[:n_samples]
print("done in %0.3fs." % (time() - t0))

# 使用 tf-idf 特征表示进行 NMF 模型训练
# 输出信息，表示正在为 NMF 提取 tf-idf 特征
print("Extracting tf-idf features for NMF...")
# 创建一个 TfidfVectorizer 对象，配置参数包括最大文档频率、最小文档频率、特征数目和停用词
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)
# 记录开始时间
t0 = time()
# 对数据样本进行 tf-idf 转换并拟合模型
tfidf = tfidf_vectorizer.fit_transform(data_samples)
# 输出处理时间
print("done in %0.3fs." % (time() - t0))

# 输出信息，表示正在为 LDA 提取 tf 特征
print("Extracting tf features for LDA...")
# 创建一个 CountVectorizer 对象，配置参数同样包括最大文档频率、最小文档频率、特征数目和停用词
tf_vectorizer = CountVectorizer(
    max_df=0.95, min_df=2, max_features=n_features, stop_words="english"
)
# 记录开始时间
t0 = time()
# 对数据样本进行 tf 转换并拟合模型
tf = tf_vectorizer.fit_transform(data_samples)
# 输出处理时间
print("done in %0.3fs." % (time() - t0))
print()

# 输出信息，表示正在拟合 NMF 模型（使用 Frobenius 范数）
print(
    "Fitting the NMF model (Frobenius norm) with tf-idf features, "
    "n_samples=%d and n_features=%d..." % (n_samples, n_features)
)
# 记录开始时间
t0 = time()
# 创建 NMF 模型对象并拟合，使用 tf-idf 特征矩阵作为输入
nmf = NMF(
    n_components=n_components,
    random_state=1,
    init=init,
    beta_loss="frobenius",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=1,
).fit(tfidf)
# 输出处理时间
print("done in %0.3fs." % (time() - t0))

# 获取 tf-idf 特征名称列表
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
# 调用函数绘制 NMF 模型中的前几个主题词
plot_top_words(
    nmf, tfidf_feature_names, n_top_words, "Topics in NMF model (Frobenius norm)"
)

# 输出信息，表示正在拟合 NMF 模型（使用广义 Kullback-Leibler 散度）
print(
    "\n" * 2,
    "Fitting the NMF model (generalized Kullback-Leibler "
    "divergence) with tf-idf features, n_samples=%d and n_features=%d..."
    % (n_samples, n_features),
)
# 记录开始时间
t0 = time()
# 创建 NMF 模型对象并拟合，使用 tf-idf 特征矩阵作为输入，使用广义 Kullback-Leibler 散度
nmf = NMF(
    n_components=n_components,
    random_state=1,
    init=init,
    beta_loss="kullback-leibler",
    solver="mu",
    max_iter=1000,
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=0.5,
).fit(tfidf)
# 输出处理时间
print("done in %0.3fs." % (time() - t0))

# 获取 tf-idf 特征名称列表
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
# 调用函数绘制 NMF 模型中的前几个主题词
plot_top_words(
    nmf,
    tfidf_feature_names,
    n_top_words,
    "Topics in NMF model (generalized Kullback-Leibler divergence)",
)

# 输出信息，表示正在拟合 MiniBatchNMF 模型（使用 Frobenius 范数）
print(
    "\n" * 2,
    "Fitting the MiniBatchNMF model (Frobenius norm) with tf-idf "
    "features, n_samples=%d and n_features=%d, batch_size=%d..."
    % (n_samples, n_features, batch_size),
)
# 记录开始时间
t0 = time()
# 创建 MiniBatchNMF 模型对象并拟合，使用 tf-idf 特征矩阵作为输入，使用 Frobenius 范数
mbnmf = MiniBatchNMF(
    n_components=n_components,
    random_state=1,
    batch_size=batch_size,
    init=init,
    beta_loss="frobenius",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=0.5,
).fit(tfidf)
# 输出处理时间
print("done in %0.3fs." % (time() - t0))

# 获取 tf-idf 特征名称列表
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
# 调用函数绘制 MiniBatchNMF 模型中的前几个主题词
plot_top_words(
    mbnmf,
    tfidf_feature_names,
    n_top_words,
    "Topics in MiniBatchNMF model (Frobenius norm)",
)

# 输出信息，表示正在拟合 MiniBatchNMF 模型（使用广义 Kullback-Leibler 散度）
print(
    "\n" * 2,
    "Fitting the MiniBatchNMF model (generalized Kullback-Leibler "
    "divergence) with tf-idf features, n_samples=%d and n_features=%d, "
    "batch_size=%d..." % (n_samples, n_features, batch_size),
)
# 记录开始时间
t0 = time()
# 创建 MiniBatchNMF 模型对象并拟合，使用 tf-idf 特征矩阵作为输入，使用广义 Kullback-Leibler 散度
mbnmf = MiniBatchNMF(
    n_components=n_components,
    random_state=1,
    batch_size=batch_size,
    init=init,
    beta_loss="kullback-leibler",
    alpha_W=0.00005,
    alpha_H=0.00005,
    l1_ratio=0.5,
).fit(tfidf)
# 打印程序运行时间
print("done in %0.3fs." % (time() - t0))

# 获取 TF-IDF 特征的名称列表
tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()

# 调用函数绘制显示 MiniBatchNMF 模型中的顶部词语和主题
plot_top_words(
    mbnmf,
    tfidf_feature_names,
    n_top_words,
    "Topics in MiniBatchNMF model (generalized Kullback-Leibler divergence)",
)

# 打印消息，显示正在拟合 LDA 模型，指定了样本数和特征数
print(
    "\n" * 2,
    "Fitting LDA models with tf features, n_samples=%d and n_features=%d..."
    % (n_samples, n_features),
)

# 创建 LatentDirichletAllocation 实例，配置模型参数并指定随机种子
lda = LatentDirichletAllocation(
    n_components=n_components,
    max_iter=5,
    learning_method="online",
    learning_offset=50.0,
    random_state=0,
)

# 计时开始，拟合 LDA 模型
t0 = time()
lda.fit(tf)
# 打印拟合时间
print("done in %0.3fs." % (time() - t0))

# 获取 TF 特征的名称列表
tf_feature_names = tf_vectorizer.get_feature_names_out()

# 调用函数绘制显示 LDA 模型中的顶部词语和主题
plot_top_words(lda, tf_feature_names, n_top_words, "Topics in LDA model")
```