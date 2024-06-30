# `D:\src\scipysrc\scikit-learn\examples\decomposition\plot_ica_blind_source_separation.py`

```
# %%
# Generate sample data
# --------------------

# 导入必要的库：numpy用于数值计算，scipy中的signal模块用于生成信号
import numpy as np
from scipy import signal

# 设置随机种子以确保可复现性
np.random.seed(0)

# 定义样本数和时间向量
n_samples = 2000
time = np.linspace(0, 8, n_samples)

# 生成三个信号源
s1 = np.sin(2 * time)  # Signal 1 : sinusoidal signal
s2 = np.sign(np.sin(3 * time))  # Signal 2 : square signal
s3 = signal.sawtooth(2 * np.pi * time)  # Signal 3: saw tooth signal

# 将三个信号源堆叠成信号矩阵S
S = np.c_[s1, s2, s3]

# 向信号矩阵S中添加高斯噪声
S += 0.2 * np.random.normal(size=S.shape)

# 对信号矩阵S进行标准化处理，使每个信号源具有相同的标准差
S /= S.std(axis=0)

# 创建混合矩阵A，用于混合信号源S以生成观测数据X
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])
X = np.dot(S, A.T)  # Generate observations

# %%
# Fit ICA and PCA models
# ----------------------

# 导入sklearn库中的PCA和FastICA模块
from sklearn.decomposition import PCA, FastICA

# 计算ICA模型
ica = FastICA(n_components=3, whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # 重构信号源
A_ = ica.mixing_  # 获取估计的混合矩阵

# 通过验证反转解混过程来证明ICA模型的适用性
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# 为了对比，计算PCA模型
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # 基于正交分量重构信号

# %%
# Plot results
# ------------

# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt

plt.figure()

# 定义要绘制的模型和对应的标题
models = [X, S, S_, H]
names = [
    "Observations (mixed signal)",
    "True Sources",
    "ICA recovered signals",
    "PCA recovered signals",
]
colors = ["red", "steelblue", "orange"]

# 遍历模型和标题，绘制子图
for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

# 调整子图布局以确保美观
plt.tight_layout()
plt.show()
```