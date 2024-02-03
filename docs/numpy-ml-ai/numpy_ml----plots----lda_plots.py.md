# `numpy-ml\numpy_ml\plots\lda_plots.py`

```py
# 禁用 flake8 检查
# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 matplotlib.pyplot 库，并使用别名 plt
import matplotlib.pyplot as plt
# 导入 seaborn 库，并使用别名 sns
import seaborn as sns

# 设置 seaborn 图表样式为白色背景
sns.set_style("white")
# 设置 seaborn 上下文为 paper，字体比例为 1
sns.set_context("paper", font_scale=1)

# 设置随机种子为 12345
np.random.seed(12345)

# 从 numpy_ml.lda 模块中导入 LDA 类
from numpy_ml.lda import LDA

# 生成语料库的函数
def generate_corpus():
    # 生成一些虚假数据
    D = 300
    T = 10
    V = 30
    N = np.random.randint(150, 200, size=D)

    # 创建三种不同类型文档的文档-主题分布
    alpha1 = np.array((20, 15, 10, 1, 1, 1, 1, 1, 1, 1))
    alpha2 = np.array((1, 1, 1, 10, 15, 20, 1, 1, 1, 1))
    alpha3 = np.array((1, 1, 1, 1, 1, 1, 10, 12, 15, 18))

    # 任意选择每个主题有 3 个非常常见的诊断词
    # 这些词几乎不与其他主题共享
    beta_probs = (
        np.ones((V, T)) + np.array([np.arange(V) % T == t for t in range(T)]).T * 19
    )
    beta_gen = np.array(list(map(lambda x: np.random.dirichlet(x), beta_probs.T))).T

    corpus = []
    theta = np.empty((D, T))

    # 从 LDA 模型生成每个文档
    for d in range(D):

        # 为文档绘制主题分布
        if d < (D / 3):
            theta[d, :] = np.random.dirichlet(alpha1, 1)[0]
        elif d < 2 * (D / 3):
            theta[d, :] = np.random.dirichlet(alpha2, 1)[0]
        else:
            theta[d, :] = np.random.dirichlet(alpha3, 1)[0]

        doc = np.array([])
        for n in range(N[d]):
            # 根据文档的主题分布绘制一个主题
            z_n = np.random.choice(np.arange(T), p=theta[d, :])

            # 根据主题-词分布绘制一个词
            w_n = np.random.choice(np.arange(V), p=beta_gen[:, z_n])
            doc = np.append(doc, w_n)

        corpus.append(doc)
    return corpus, T

# 绘制未平滑的 LDA 模型
def plot_unsmoothed():
    # 生成语料库
    corpus, T = generate_corpus()
    # 创建 LDA 对象
    L = LDA(T)
    # 训练 LDA 模型
    L.train(corpus, verbose=False)
    # 创建一个包含两个子图的图形对象和子图对象
    fig, axes = plt.subplots(1, 2)
    # 在第一个子图上绘制热力图，不显示 x 和 y 轴标签，将图形对象传递给 ax1
    ax1 = sns.heatmap(L.beta, xticklabels=[], yticklabels=[], ax=axes[0])
    # 设置第一个子图的 x 轴标签
    ax1.set_xlabel("Topics")
    # 设置第一个子图的 y 轴标签
    ax1.set_ylabel("Words")
    # 设置第一个子图的标题
    ax1.set_title("Recovered topic-word distribution")
    
    # 在第二个子图上绘制热力图，不显示 x 和 y 轴标签，将图形对象传递给 ax2
    ax2 = sns.heatmap(L.gamma, xticklabels=[], yticklabels=[], ax=axes[1])
    # 设置第二个子图的 x 轴标签
    ax2.set_xlabel("Topics")
    # 设置第二个子图的 y 轴标签
    ax2.set_ylabel("Documents")
    # 设置第二个子图的标题
    ax2.set_title("Recovered document-topic distribution")
    
    # 将图形保存为图片文件，设置分辨率为 300 dpi
    plt.savefig("img/plot_unsmoothed.png", dpi=300)
    # 关闭所有图形对象
    plt.close("all")
```