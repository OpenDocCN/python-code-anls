# `so-vits-svc\cluster\train_cluster.py`

```
# 导入必要的库
import argparse
import logging
import os
import time
from pathlib import Path

import numpy as np
import torch
import tqdm
from kmeans import KMeansGPU
from sklearn.cluster import KMeans, MiniBatchKMeans

# 配置日志记录器
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义训练聚类模型的函数
def train_cluster(in_dir, n_clusters, use_minibatch=True, verbose=False, use_gpu=False):
    # 如果输入目录以".ipynb_checkpoints"结尾，则忽略
    if str(in_dir).endswith(".ipynb_checkpoints"):
        logger.info(f"Ignore {in_dir}")

    # 记录加载特征的信息
    logger.info(f"Loading features from {in_dir}")
    features = []
    nums = 0
    # 遍历输入目录下的所有".soft.pt"文件
    for path in tqdm.tqdm(in_dir.glob("*.soft.pt")):
        # 从文件中加载特征数据，并转换为numpy数组
        features.append(torch.load(path, map_location="cpu").squeeze(0).numpy().T)
    # 将所有特征数据拼接成一个数组
    features = np.concatenate(features, axis=0)
    # 打印特征数据的大小和形状
    print(nums, features.nbytes/ 1024**2, "MB , shape:", features.shape, features.dtype)
    # 将特征数据类型转换为np.float32
    features = features.astype(np.float32)
    # 记录聚类特征的形状信息
    logger.info(f"Clustering features of shape: {features.shape}")
    t = time.time()
    # 如果不使用GPU
    if(use_gpu is False):
        # 如果使用MiniBatchKMeans
        if use_minibatch:
            kmeans = MiniBatchKMeans(n_clusters=n_clusters, verbose=verbose, batch_size=4096, max_iter=80).fit(features)
        # 如果使用普通的KMeans
        else:
            kmeans = KMeans(n_clusters=n_clusters, verbose=verbose).fit(features)
    # 如果使用GPU
    else:
        # 使用KMeansGPU进行聚类
        kmeans = KMeansGPU(n_clusters=n_clusters, mode='euclidean', verbose=2 if verbose else 0, max_iter=500, tol=1e-2)
        features = torch.from_numpy(features)
        kmeans.fit_predict(features)

    # 打印聚类所花费的时间
    print(time.time()-t, "s")

    # 构建结果字典
    x = {
            "n_features_in_": kmeans.n_features_in_ if use_gpu is False else features.shape[1],
            "_n_threads": kmeans._n_threads if use_gpu is False else 4,
            "cluster_centers_": kmeans.cluster_centers_ if use_gpu is False else kmeans.centroids.cpu().numpy(),
    }
    # 打印结束信息
    print("end")

    # 返回结果字典
    return x

# 如果作为主程序运行，则执行以下代码
if __name__ == "__main__":
    # 创建一个参数解析器对象
    parser = argparse.ArgumentParser()
    # 添加一个名为 dataset 的命令行参数，类型为 Path，默认值为 "./dataset/44k"，帮助信息为训练数据目录的路径
    parser.add_argument('--dataset', type=Path, default="./dataset/44k",
                        help='path of training data directory')
    # 添加一个名为 output 的命令行参数，类型为 Path，默认值为 "logs/44k"，帮助信息为模型输出目录的路径
    parser.add_argument('--output', type=Path, default="logs/44k",
                        help='path of model output directory')
    # 添加一个名为 gpu 的命令行参数，类型为布尔值，默认值为 False，帮助信息为是否使用 GPU
    parser.add_argument('--gpu',action='store_true', default=False ,
                        help='to use GPU')

    # 解析命令行参数
    args = parser.parse_args()

    # 从命令行参数中获取输出目录
    checkpoint_dir = args.output
    # 从命令行参数中获取数据集目录
    dataset = args.dataset
    # 从命令行参数中获取是否使用 GPU
    use_gpu = args.gpu
    # 设置聚类数为 10000
    n_clusters = 10000
    
    # 创建一个空字典用于存储模型检查点
    ckpt = {}
    # 遍历数据集目录下的子目录
    for spk in os.listdir(dataset):
        # 如果是目录
        if os.path.isdir(dataset/spk):
            # 打印训练 K 均值聚类的信息
            print(f"train kmeans for {spk}...")
            # 设置输入目录为当前子目录
            in_dir = dataset/spk
            # 调用 train_cluster 函数进行聚类训练，并将结果存储到 ckpt 字典中
            x = train_cluster(in_dir, n_clusters,use_minibatch=False,verbose=False,use_gpu=use_gpu)
            ckpt[spk] = x

    # 设置模型检查点文件路径
    checkpoint_path = checkpoint_dir / f"kmeans_{n_clusters}.pt"
    # 确保模型检查点文件的父目录存在，如果不存在则创建
    checkpoint_path.parent.mkdir(exist_ok=True, parents=True)
    # 保存 ckpt 字典到模型检查点文件
    torch.save(
        ckpt,
        checkpoint_path,
    )
```