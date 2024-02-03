# `numpy-ml\numpy_ml\plots\ngram_plots.py`

```
# 禁用 flake8 的警告
# 导入 numpy 库，并使用 np 别名
import numpy as np

# 导入 matplotlib.pyplot 和 seaborn 库
import matplotlib.pyplot as plt
import seaborn as sns

# 设置 seaborn 库的样式为白色，上下文为 notebook，字体比例为 1
# 更多信息可参考：https://seaborn.pydata.org/generated/seaborn.set_context.html
# 更多信息可参考：https://seaborn.pydata.org/generated/seaborn.set_style.html
sns.set_style("white")
sns.set_context("notebook", font_scale=1)

# 从 numpy_ml.ngram 模块中导入 MLENGram、AdditiveNGram、GoodTuringNGram 类
from numpy_ml.ngram import MLENGram, AdditiveNGram, GoodTuringNGram

# 定义函数 plot_count_models，用于绘制不同模型的计数情况
def plot_count_models(GT, N):
    # 获取 GoodTuringNGram 对象中的 _num_grams_with_count 属性
    NC = GT._num_grams_with_count
    # 获取 GoodTuringNGram 对象中的 _count_models[N] 属性
    mod = GT._count_models[N]
    # 获取 GoodTuringNGram 对象中的 counts[N] 属性中的最大值
    max_n = max(GT.counts[N].values())
    # 创建一个列表，包含每个计数的计数
    emp = [NC(n + 1, N) for n in range(max_n)]
    # 创建一个列表，包含模型预测的计数
    prd = [np.exp(mod.predict(np.array([n + 1]))) for n in range(max_n + 10)]
    # 绘制散点图，展示实际值和模型预测值
    plt.scatter(range(max_n), emp, c="r", label="actual")
    plt.plot(range(max_n + 10), prd, "-", label="model")
    plt.ylim([-1, 100])
    plt.xlabel("Count ($r$)")
    plt.ylabel("Count-of-counts ($N_r$)")
    plt.legend()
    plt.savefig("test.png")
    plt.close()

# 定义函数 compare_probs，用于比较不同模型的概率
def compare_probs(fp, N):
    # 创建 MLENGram 对象 MLE
    MLE = MLENGram(N, unk=False, filter_punctuation=False, filter_stopwords=False)
    # 使用给定文件路径 fp 进行训练，编码方式为 utf-8-sig
    MLE.train(fp, encoding="utf-8-sig")

    # 初始化各种概率列表
    add_y, mle_y, gtt_y = [], [], []
    addu_y, mleu_y, gttu_y = [], [], []
    seen = ("<bol>", "the")
    unseen = ("<bol>", "asdf")

    # 创建 GoodTuringNGram 对象 GTT
    GTT = GoodTuringNGram(
        N, conf=1.96, unk=False, filter_stopwords=False, filter_punctuation=False
    )
    # 使用给定文件路径 fp 进行训练，编码方式为 utf-8-sig
    GTT.train(fp, encoding="utf-8-sig")

    # 计算已见序列 seen 的概率和未见序列 unseen 的概率
    gtt_prob = GTT.log_prob(seen, N)
    gtt_prob_u = GTT.log_prob(unseen, N)

    # 在 0 到 10 之间生成 20 个数，用于循环
    for K in np.linspace(0, 10, 20):
        # 创建 AdditiveNGram 对象 ADD
        ADD = AdditiveNGram(
            N, K, unk=False, filter_punctuation=False, filter_stopwords=False
        )
        # 使用给定文件路径 fp 进行训练，编码方式为 utf-8-sig
        ADD.train(fp, encoding="utf-8-sig")

        # 计算已见序列 seen 的概率和 MLENGram 对象 MLE 的概率
        add_prob = ADD.log_prob(seen, N)
        mle_prob = MLE.log_prob(seen, N)

        # 将计算结果添加到对应的列表中
        add_y.append(add_prob)
        mle_y.append(mle_prob)
        gtt_y.append(gtt_prob)

        # 计算未见序列 unseen 的概率
        mle_prob_u = MLE.log_prob(unseen, N)
        add_prob_u = ADD.log_prob(unseen, N)

        # 将计算结果添加到对应的列表中
        addu_y.append(add_prob_u)
        mleu_y.append(mle_prob_u)
        gttu_y.append(gtt_prob_u)
    # 绘制折线图，x轴为0到10的20个点，y轴为add_y，添加图例"Additive (seen ngram)"
    plt.plot(np.linspace(0, 10, 20), add_y, label="Additive (seen ngram)")
    # 绘制折线图，x轴为0到10的20个点，y轴为addu_y，添加图例"Additive (unseen ngram)"
    plt.plot(np.linspace(0, 10, 20), addu_y, label="Additive (unseen ngram)")
    # 注释掉的代码，不会被执行，这里是绘制Good-Turing (seen ngram)的折线图
    # plt.plot(np.linspace(0, 10, 20), gtt_y, label="Good-Turing (seen ngram)")
    # 注释掉的代码，不会被执行，这里是绘制Good-Turing (unseen ngram)的折线图
    # plt.plot(np.linspace(0, 10, 20), gttu_y, label="Good-Turing (unseen ngram)")
    # 绘制折线图，x轴为0到10的20个点，y轴为mle_y，线条为虚线，添加图例"MLE (seen ngram)"
    plt.plot(np.linspace(0, 10, 20), mle_y, "--", label="MLE (seen ngram)")
    # 设置x轴标签为"K"
    plt.xlabel("K")
    # 设置y轴标签为"log P(sequence)"
    plt.ylabel("log P(sequence)")
    # 添加图例
    plt.legend()
    # 保存图像为"img/add_smooth.png"
    plt.savefig("img/add_smooth.png")
    # 关闭所有图形窗口
    plt.close("all")
# 绘制经验频率与简单 Good Turing 平滑值的散点图，按排名顺序。依赖于 pylab 和 matplotlib。
def plot_gt_freqs(fp):
    # 创建一个最大似然估计的 1-gram 模型，不过滤标点符号和停用词
    MLE = MLENGram(1, filter_punctuation=False, filter_stopwords=False)
    # 从文件中训练模型
    MLE.train(fp, encoding="utf-8-sig")
    # 获取模型中的词频统计
    counts = dict(MLE.counts[1])

    # 创建一个 Good Turing 平滑的 1-gram 模型，不过滤停用词和标点符号
    GT = GoodTuringNGram(1, filter_stopwords=False, filter_punctuation=False)
    # 从文件中训练模型
    GT.train(fp, encoding="utf-8-sig")

    # 创建一个拉普拉斯平滑的 1-gram 模型，不过滤停用词和标点符号
    ADD = AdditiveNGram(1, 1, filter_punctuation=False, filter_stopwords=False)
    # 从文件中训练模型
    ADD.train(fp, encoding="utf-8-sig")

    # 计算总词频
    tot = float(sum(counts.values()))
    # 计算每个词的频率
    freqs = dict([(token, cnt / tot) for token, cnt in counts.items()])
    # 计算每个词的简单 Good Turing 平滑概率
    sgt_probs = dict([(tok, np.exp(GT.log_prob(tok, 1))) for tok in counts.keys()])
    # 计算每个词的拉普拉斯平滑概率
    as_probs = dict([(tok, np.exp(ADD.log_prob(tok, 1))) for tok in counts.keys()])

    # 创建 X, Y 坐标，用于绘制 MLE 的散点图
    X, Y = np.arange(len(freqs)), sorted(freqs.values(), reverse=True)
    plt.loglog(X, Y, "k+", alpha=0.25, label="MLE")

    # 创建 X, Y 坐标，用于绘制简单 Good Turing 的散点图
    X, Y = np.arange(len(sgt_probs)), sorted(sgt_probs.values(), reverse=True)
    plt.loglog(X, Y, "r+", alpha=0.25, label="simple Good-Turing")

    # 创建 X, Y 坐标，用于绘制拉普拉斯平滑的散点图
    X, Y = np.arange(len(as_probs)), sorted(as_probs.values(), reverse=True)
    plt.loglog(X, Y, "b+", alpha=0.25, label="Laplace smoothing")

    # 设置 X 轴标签
    plt.xlabel("Rank")
    # 设置 Y 轴标签
    plt.ylabel("Probability")
    # 添加图例
    plt.legend()
    # 调整布局
    plt.tight_layout()
    # 保存图像
    plt.savefig("img/rank_probs.png")
    # 关闭所有图形
    plt.close("all")
```