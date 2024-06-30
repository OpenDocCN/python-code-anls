# `D:\src\scipysrc\scikit-learn\examples\cluster\plot_inductive_clustering.py`

```
# ====================
# Inductive Clustering
# ====================
#
# Clustering can be expensive, especially when our dataset contains millions
# of datapoints. Many clustering algorithms are not :term:`inductive` and so
# cannot be directly applied to new data samples without recomputing the
# clustering, which may be intractable. Instead, we can use clustering to then
# learn an inductive model with a classifier, which has several benefits:
#
# - it allows the clusters to scale and apply to new data
# - unlike re-fitting the clusters to new samples, it makes sure the labelling
#   procedure is consistent over time
# - it allows us to use the inferential capabilities of the classifier to
#   describe or explain the clusters
#
# This example illustrates a generic implementation of a meta-estimator which
# extends clustering by inducing a classifier from the cluster labels.
#

# Authors: Chirag Nagpal
#          Christos Aridas

# 导入 matplotlib 中的 pyplot 模块，用于绘图
import matplotlib.pyplot as plt

# 导入必要的类和函数
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

# 设置生成数据的样本数和随机种子
N_SAMPLES = 5000
RANDOM_STATE = 42

# 检查是否可以委托一个方法给底层分类器
def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.
    
    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )

# 定义一个自定义的元估计器类 InductiveClusterer
class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    # 拟合方法，使用聚类器进行聚类并用分类器学习
    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    # 如果分类器有预测方法，则委托给底层分类器
    @available_if(_classifier_has("predict"))
    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    # 如果分类器有决策函数方法，则委托给底层分类器
    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier_.decision_function(X)

# 绘制散点图函数，展示数据的分布情况
def plot_scatter(X, color, alpha=0.5):
    return plt.scatter(X[:, 0], X[:, 1], c=color, alpha=alpha, edgecolor="k")

# 生成一些用于聚类的训练数据
X, y = make_blobs(
    n_samples=N_SAMPLES,
    cluster_std=[1.0, 1.0, 0.5],
    centers=[(-5, -5), (0, 0), (5, 5)],
    random_state=RANDOM_STATE,
)

# 在训练数据上训练一个聚类算法并获取聚类标签
clusterer = AgglomerativeClustering(n_clusters=3)
cluster_labels = clusterer.fit_predict(X)

# 创建一个图形窗口，设置图形的大小为 12x4 英寸，子图为 1 行 3 列的第一个位置
plt.figure(figsize=(12, 4))
plt.subplot(131)
# 绘制散点图，显示数据集 X 和聚类标签 cluster_labels 的关系
plot_scatter(X, cluster_labels)
# 设置图表标题为 "Ward Linkage"

# 生成新样本并将其与原始数据集一起绘制
X_new, y_new = make_blobs(
    n_samples=10, centers=[(-7, -1), (-2, 4), (3, 6)], random_state=RANDOM_STATE
)

# 在三个子图中的第二个位置绘制散点图，显示数据集 X 和黑色标记的 X_new
plt.subplot(132)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, "black", 1)
# 设置子图标题为 "Unknown instances"

# 声明归纳学习模型，用于预测未知实例的聚类成员资格
classifier = RandomForestClassifier(random_state=RANDOM_STATE)
inductive_learner = InductiveClusterer(clusterer, classifier).fit(X)

# 预测 X_new 的可能聚类
probable_clusters = inductive_learner.predict(X_new)

# 在三个子图中的第三个位置绘制散点图，显示数据集 X 和预测的聚类 probable_clusters
ax = plt.subplot(133)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, probable_clusters)

# 根据归纳学习器的预测，绘制决策边界的显示
DecisionBoundaryDisplay.from_estimator(
    inductive_learner, X, response_method="predict", alpha=0.4, ax=ax
)
# 设置子图标题为 "Classify unknown instances"

# 显示所有子图
plt.show()
```