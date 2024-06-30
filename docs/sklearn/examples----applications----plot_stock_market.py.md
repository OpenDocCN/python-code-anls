# `D:\src\scipysrc\scikit-learn\examples\applications\plot_stock_market.py`

```
"""
=======================================
Visualizing the stock market structure
=======================================

This example employs several unsupervised learning techniques to extract
the stock market structure from variations in historical quotes.

The quantity that we use is the daily variation in quote price: quotes
that are linked tend to fluctuate in relation to each other during a day.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# Retrieve the data from Internet
# -------------------------------
#
# The data is from 2003 - 2008. This is reasonably calm: (not too long ago so
# that we get high-tech firms, and before the 2008 crash). This kind of
# historical data can be obtained from APIs like the
# `data.nasdaq.com <https://data.nasdaq.com/>`_ and
# `alphavantage.co <https://www.alphavantage.co/>`_.

import sys

import numpy as np
import pandas as pd

symbol_dict = {
    "TOT": "Total",
    "XOM": "Exxon",
    "CVX": "Chevron",
    "COP": "ConocoPhillips",
    "VLO": "Valero Energy",
    "MSFT": "Microsoft",
    "IBM": "IBM",
    "TWX": "Time Warner",
    "CMCSA": "Comcast",
    "CVC": "Cablevision",
    "YHOO": "Yahoo",
    "DELL": "Dell",
    "HPQ": "HP",
    "AMZN": "Amazon",
    "TM": "Toyota",
    "CAJ": "Canon",
    "SNE": "Sony",
    "F": "Ford",
    "HMC": "Honda",
    "NAV": "Navistar",
    "NOC": "Northrop Grumman",
    "BA": "Boeing",
    "KO": "Coca Cola",
    "MMM": "3M",
    "MCD": "McDonald's",
    "PEP": "Pepsi",
    "K": "Kellogg",
    "UN": "Unilever",
    "MAR": "Marriott",
    "PG": "Procter Gamble",
    "CL": "Colgate-Palmolive",
    "GE": "General Electrics",
    "WFC": "Wells Fargo",
    "JPM": "JPMorgan Chase",
    "AIG": "AIG",
    "AXP": "American express",
    "BAC": "Bank of America",
    "GS": "Goldman Sachs",
    "AAPL": "Apple",
    "SAP": "SAP",
    "CSCO": "Cisco",
    "TXN": "Texas Instruments",
    "XRX": "Xerox",
    "WMT": "Wal-Mart",
    "HD": "Home Depot",
    "GSK": "GlaxoSmithKline",
    "PFE": "Pfizer",
    "SNY": "Sanofi-Aventis",
    "NVS": "Novartis",
    "KMB": "Kimberly-Clark",
    "R": "Ryder",
    "GD": "General Dynamics",
    "RTN": "Raytheon",
    "CVS": "CVS",
    "CAT": "Caterpillar",
    "DD": "DuPont de Nemours",
}

symbols, names = np.array(sorted(symbol_dict.items())).T

quotes = []

for symbol in symbols:
    print("Fetching quote history for %r" % symbol, file=sys.stderr)
    url = (
        "https://raw.githubusercontent.com/scikit-learn/examples-data/"
        "master/financial-data/{}.csv"
    )
    quotes.append(pd.read_csv(url.format(symbol)))

close_prices = np.vstack([q["close"] for q in quotes])
open_prices = np.vstack([q["open"] for q in quotes])

# The daily variations of the quotes are what carry the most information
variation = close_prices - open_prices

# %%
# .. _stock_market:
#
# Learning a graph structure
# --------------------------
#
# We use sparse inverse covariance estimation to find which quotes are
# 导入协方差相关的条件属性，sparse inverse covariance 给出了一个图，即连接列表。对于每个symbol，它所连接的symbols是用来解释其波动的有用信息。

from sklearn import covariance

# 生成10个在-1.5到1范围内等比分布的值作为alphas参数
alphas = np.logspace(-1.5, 1, num=10)

# 创建GraphicalLassoCV对象，用于拟合数据的图形化 LASSO(Least Absolute Shrinkage and Selection Operator)模型
edge_model = covariance.GraphicalLassoCV(alphas=alphas)

# 标准化时间序列：使用相关性而不是协方差，前者对于结构恢复更有效
X = variation.copy().T
X /= X.std(axis=0)
edge_model.fit(X)

# Clustering using affinity propagation
# 使用亲和力传播进行聚类
# 在这里，我们使用聚类将表现相似的报价组合在一起。在scikit-learn中，我们使用亲和性传播作为聚类方法，因为它不强制要求等大小的簇，并且它可以从数据中自动选择簇的数量。

from sklearn import cluster

# 使用亲和力传播进行聚类，返回聚类结果和标签
_, labels = cluster.affinity_propagation(edge_model.covariance_, random_state=0)
n_labels = labels.max()

# 输出每个聚类中的股票名
for i in range(n_labels + 1):
    print(f"Cluster {i + 1}: {', '.join(names[labels == i])}")

# Embedding in 2D space
# 二维空间嵌入
# 为了可视化，我们需要在二维画布上放置不同的symbol。我们使用流形学习技术来获得2D嵌入。我们使用稠密的eigen_solver来实现可重现性(因为arpack是用我们无法控制的随机向量初始化的)。此外，我们使用大量的邻居来捕获大尺度结构。

# 为可视化寻找一个低维嵌入：找到节点(股票)在二维平面上的最佳位置

from sklearn import manifold

# 创建LocallyLinearEmbedding对象，用于将数据嵌入到一个低维空间中
node_position_model = manifold.LocallyLinearEmbedding(
    n_components=2, eigen_solver="dense", n_neighbors=6
)

# 进行嵌入并将结果转置
embedding = node_position_model.fit_transform(X.T).T

# Visualization
# 可视化
# 使用3个模型的输出组合成一个2D图，其中节点表示股票，边表示：
# - 聚类标签用于定义节点的颜色
# - 稀疏协方差模型用于显示边的强度
# - 2D嵌入用于将节点定位在平面上
# 这个示例包含大量与可视化相关的代码，因为在这里可视化对于展示图形至关重要。其中一个挑战是在最小化重叠的同时定位标签的位置。为此，我们使用了一个基于每个轴上最近邻方向的启发式方法。

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.figure(1, facecolor="w", figsize=(10, 8))
plt.clf()
ax = plt.axes([0.0, 0.0, 1.0, 1.0])
plt.axis("off")

# 绘制偏相关图表
partial_correlations = edge_model.precision_.copy()
# 计算偏相关系数的标准化因子
d = 1 / np.sqrt(np.diag(partial_correlations))
partial_correlations *= d
partial_correlations *= d[:, np.newaxis]
# 根据阈值筛选出非零偏相关系数
non_zero = np.abs(np.triu(partial_correlations, k=1)) > 0.02

# 根据嵌入坐标绘制节点散点图
plt.scatter(
    embedding[0], embedding[1], s=100 * d**2, c=labels, cmap=plt.cm.nipy_spectral
)

# 绘制边
start_idx, end_idx = np.where(non_zero)
# 创建边的序列
segments = [
    [embedding[:, start], embedding[:, stop]] for start, stop in zip(start_idx, end_idx)
]
values = np.abs(partial_correlations[non_zero])
# 创建线段集合
lc = LineCollection(
    segments, zorder=0, cmap=plt.cm.hot_r, norm=plt.Normalize(0, 0.7 * values.max())
)
lc.set_array(values)
lc.set_linewidths(15 * values)
ax.add_collection(lc)

# 给每个节点添加标签，确保标签位置不重叠
for index, (name, label, (x, y)) in enumerate(zip(names, labels, embedding.T)):
    dx = x - embedding[0]
    dx[index] = 1
    dy = y - embedding[1]
    dy[index] = 1
    this_dx = dx[np.argmin(np.abs(dy))]
    this_dy = dy[np.argmin(np.abs(dx))]
    if this_dx > 0:
        horizontalalignment = "left"
        x = x + 0.002
    else:
        horizontalalignment = "right"
        x = x - 0.002
    if this_dy > 0:
        verticalalignment = "bottom"
        y = y + 0.002
    else:
        verticalalignment = "top"
        y = y - 0.002
    # 在节点位置添加文本标签
    plt.text(
        x,
        y,
        name,
        size=10,
        horizontalalignment=horizontalalignment,
        verticalalignment=verticalalignment,
        bbox=dict(
            facecolor="w",
            edgecolor=plt.cm.nipy_spectral(label / float(n_labels)),
            alpha=0.6,
        ),
    )

# 设置 x 轴和 y 轴的显示范围
plt.xlim(
    embedding[0].min() - 0.15 * np.ptp(embedding[0]),
    embedding[0].max() + 0.10 * np.ptp(embedding[0]),
)
plt.ylim(
    embedding[1].min() - 0.03 * np.ptp(embedding[1]),
    embedding[1].max() + 0.03 * np.ptp(embedding[1]),
)

# 显示绘图
plt.show()
```