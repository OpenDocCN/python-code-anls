# `numpy-ml\numpy_ml\plots\hmm_plots.py`

```
# 禁用 flake8 检查
# 导入 numpy 库，并重命名为 np
import numpy as np
# 从 matplotlib 库中导入 pyplot 模块，并重命名为 plt
from matplotlib import pyplot as plt
# 从 seaborn 库中导入 sns 模块
import seaborn as sns

# 设置 seaborn 库的样式为白色
sns.set_style("white")
# 设置 seaborn 库的上下文为 notebook，字体比例为 0.8
sns.set_context("notebook", font_scale=0.8)

# 从 hmmlearn.hmm 模块中导入 MultinomialHMM 类，并重命名为 MHMM
from hmmlearn.hmm import MultinomialHMM as MHMM
# 从 numpy_ml.hmm 模块中导入 MultinomialHMM 类
from numpy_ml.hmm import MultinomialHMM

# 定义生成训练数据的函数
def generate_training_data(params, n_steps=500, n_examples=15):
    # 创建 MultinomialHMM 对象
    hmm = MultinomialHMM(A=params["A"], B=params["B"], pi=params["pi"])

    # 生成新的序列
    observations = []
    for i in range(n_examples):
        # 生成潜在状态和观测值序列
        latent, obs = hmm.generate(
            n_steps, params["latent_states"], params["obs_types"]
        )
        # 断言潜在状态和观测值序列的长度与指定长度相同
        assert len(latent) == len(obs) == n_steps
        observations.append(obs)

    # 将观测值序列转换为 numpy 数组
    observations = np.array(observations)
    return observations

# 定义默认的 HMM 模型参数
def default_hmm():
    obs_types = [0, 1, 2, 3]
    latent_states = ["H", "C"]

    # 计算派生变量
    V = len(obs_types)
    N = len(latent_states)

    # 定义一个非常简单的 HMM 模型，包含 T=3 个观测值
    O = np.array([1, 3, 1]).reshape(1, -1)
    A = np.array([[0.9, 0.1], [0.5, 0.5]])
    B = np.array([[0.2, 0.7, 0.09, 0.01], [0.1, 0.0, 0.8, 0.1]])
    pi = np.array([0.75, 0.25])

    return {
        "latent_states": latent_states,
        "obs_types": obs_types,
        "V": V,
        "N": N,
        "O": O,
        "A": A,
        "B": B,
        "pi": pi,
    }

# 绘制矩阵图
def plot_matrices(params, best, best_theirs):
    cmap = "copper"
    ll_mine, best = best
    ll_theirs, best_theirs = best_theirs

    # 创建包含 3x3 子图的图形对象
    fig, axes = plt.subplots(3, 3)
    axes = {
        "A": [axes[0, 0], axes[0, 1], axes[0, 2]],
        "B": [axes[1, 0], axes[1, 1], axes[1, 2]],
        "pi": [axes[2, 0], axes[2, 1], axes[2, 2]],
    }
    # 遍历包含参数名称和对应标题的元组列表
    for k, tt in [("A", "Transition"), ("B", "Emission"), ("pi", "Prior")]:
        # 获取真实值、估计值和第三方库估计值的轴对象
        true_ax, est_ax, est_theirs_ax = axes[k]
        true, est, est_theirs = params[k], best[k], best_theirs[k]

        # 如果参数是 "pi"，则将其形状调整为列向量
        if k == "pi":
            true = true.reshape(-1, 1)
            est = est.reshape(-1, 1)
            est_theirs = est_theirs.reshape(-1, 1)

        # 绘制真实值的热力图
        true_ax = sns.heatmap(
            true,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap=cmap,
            cbar=False,
            annot=True,
            ax=true_ax,
            xticklabels=[],
            yticklabels=[],
            linewidths=0.25,
        )

        # 绘制估计值的热力图
        est_ax = sns.heatmap(
            est,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            ax=est_ax,
            cmap=cmap,
            annot=True,
            cbar=False,
            xticklabels=[],
            yticklabels=[],
            linewidths=0.25,
        )

        # 绘制第三方库估计值的热力图
        est_theirs_ax = sns.heatmap(
            est_theirs,
            vmin=0.0,
            vmax=1.0,
            fmt=".2f",
            cmap=cmap,
            annot=True,
            cbar=False,
            xticklabels=[],
            yticklabels=[],
            linewidths=0.25,
            ax=est_theirs_ax,
        )

        # 设置真实值轴的标题
        true_ax.set_title("{} (True)".format(tt))
        # 设置估计值轴的标题
        est_ax.set_title("{} (Mine)".format(tt))
        # 设置第三方库估计值轴的标题
        est_theirs_ax.set_title("{} (hmmlearn)".format(tt))
    # 设置整体图的标题，包括自己的对数似然和第三方库的对数似然
    fig.suptitle("LL (mine): {:.2f}, LL (hmmlearn): {:.2f}".format(ll_mine, ll_theirs))
    # 调整布局
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # 保存图像
    plt.savefig("img/plot.png", dpi=300)
    # 关闭图像
    plt.close()
# 定义一个测试隐马尔可夫模型的函数
def test_HMM():
    # 设置随机种子，保证结果的可重复性
    np.random.seed(12345)
    # 设置打印选项，精度为5位小数，禁止科学计数法
    np.set_printoptions(precision=5, suppress=True)

    # 获取默认的隐马尔可夫模型参数
    P = default_hmm()
    ls, obs = P["latent_states"], P["obs_types"]

    # 生成一个新的序列
    O = generate_training_data(P, n_steps=30, n_examples=25)

    # 设置容差值
    tol = 1e-5
    n_runs = 5
    best, best_theirs = (-np.inf, []), (-np.inf, [])
    # 进行多次运行以找到最佳结果
    for _ in range(n_runs):
        # 初始化一个多项式隐马尔可夫模型
        hmm = MultinomialHMM()
        # 使用生成的序列拟合模型参数
        A_, B_, pi_ = hmm.fit(O, ls, obs, tol=tol, verbose=True)

        # 初始化一个他们自己实现的多项式隐马尔可夫模型
        theirs = MHMM(
            tol=tol,
            verbose=True,
            n_iter=int(1e9),
            transmat_prior=1,
            startprob_prior=1,
            algorithm="viterbi",
            n_components=len(ls),
        )

        # 将序列展平并拟合他们的模型
        O_flat = O.reshape(1, -1).flatten().reshape(-1, 1)
        theirs = theirs.fit(O_flat, lengths=[O.shape[1]] * O.shape[0])

        # 根据拟合的模型计算对数似然
        hmm2 = MultinomialHMM(A=A_, B=B_, pi=pi_)
        like = np.sum([hmm2.log_likelihood(obs) for obs in O])
        like_theirs = theirs.score(O_flat, lengths=[O.shape[1]] * O.shape[0])

        # 更新最佳结果
        if like > best[0]:
            best = (like, {"A": A_, "B": B_, "pi": pi_})

        if like_theirs > best_theirs[0]:
            best_theirs = (
                like_theirs,
                {
                    "A": theirs.transmat_,
                    "B": theirs.emissionprob_,
                    "pi": theirs.startprob_,
                },
            )
    # 打印最终的对数似然值
    print("Final log likelihood of sequence: {:.5f}".format(best[0]))
    print("Final log likelihood of sequence (theirs): {:.5f}".format(best_theirs[0]))
    # 绘制结果矩阵
    plot_matrices(P, best, best_theirs)
```