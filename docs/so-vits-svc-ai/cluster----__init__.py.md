# `so-vits-svc\cluster\__init__.py`

```
# 导入 torch 库
import torch
# 从 sklearn 库中导入 KMeans 类
from sklearn.cluster import KMeans

# 根据给定的检查点路径获取聚类模型
def get_cluster_model(ckpt_path):
    # 加载检查点文件
    checkpoint = torch.load(ckpt_path)
    # 创建空字典用于存储 KMeans 对象
    kmeans_dict = {}
    # 遍历检查点中的每个说话者和对应的检查点
    for spk, ckpt in checkpoint.items():
        # 创建 KMeans 对象，使用检查点中的特征数
        km = KMeans(ckpt["n_features_in_"])
        # 设置 KMeans 对象的特征数属性
        km.__dict__["n_features_in_"] = ckpt["n_features_in_"]
        # 设置 KMeans 对象的线程数属性
        km.__dict__["_n_threads"] = ckpt["_n_threads"]
        # 设置 KMeans 对象的聚类中心属性
        km.__dict__["cluster_centers_"] = ckpt["cluster_centers_"]
        # 将 KMeans 对象存入字典，以说话者名为键
        kmeans_dict[spk] = km
    # 返回 KMeans 对象字典
    return kmeans_dict

# 根据给定的模型、输入数据和说话者获取聚类结果
def get_cluster_result(model, x, speaker):
    """
        x: np.array [t, 256]
        return cluster class result
    """
    # 使用指定说话者的模型预测输入数据的聚类结果
    return model[speaker].predict(x)

# 根据给定的模型、输入数据和说话者获取聚类中心结果
def get_cluster_center_result(model, x, speaker):
    """x: np.array [t, 256]"""
    # 使用指定说话者的模型预测输入数据的聚类结果
    predict = model[speaker].predict(x)
    # 返回预测聚类结果对应的聚类中心
    return model[speaker].cluster_centers_[predict]

# 根据给定的模型、聚类中心索引和说话者获取聚类中心
def get_center(model, x, speaker):
    # 返回指定说话者模型的指定聚类中心索引对应的聚类中心
    return model[speaker].cluster_centers_[x]
```