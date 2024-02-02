# `so-vits-svc\cluster\kmeans.py`

```py
# 从 time 模块中导入 time 函数
from time import time

# 导入 numpy 模块并重命名为 np
import numpy as np
# 导入 pynvml 模块
import pynvml
# 导入 torch 模块
import torch
# 从 torch.nn.functional 模块中导入 normalize 函数
from torch.nn.functional import normalize

# 定义一个名为 _kpp 的函数，用于根据 kmeans++ 方法从数据中选择 k 个点
def _kpp(data: torch.Tensor, k: int, sample_size: int = -1):
    """ Picks k points in the data based on the kmeans++ method.

    Parameters
    ----------
    data : torch.Tensor
        Expect a rank 1 or 2 array. Rank 1 is assumed to describe 1-D
        data, rank 2 multidimensional data, in which case one
        row is one observation.
    k : int
        Number of samples to generate.
    sample_size : int
        sample data to avoid memory overflow during calculation

    Returns
    -------
    init : ndarray
        A 'k' by 'N' containing the initial centroids.

    References
    ----------
    .. [1] D. Arthur and S. Vassilvitskii, "k-means++: the advantages of
       careful seeding", Proceedings of the Eighteenth Annual ACM-SIAM Symposium
       on Discrete Algorithms, 2007.
    .. [2] scipy/cluster/vq.py: _kpp
    """
    # 获取数据的批量大小
    batch_size=data.shape[0]
    # 如果批量大小大于采样大小，则从数据中随机采样
    if batch_size>sample_size:
        data = data[torch.randint(0, batch_size,[sample_size], device=data.device)]
    # 获取数据的维度
    dims = data.shape[1] if len(data.shape) > 1 else 1
    # 初始化存储初始质心的数组
    init = torch.zeros((k, dims)).to(data.device)
    # 创建均匀分布的随机数生成器
    r = torch.distributions.uniform.Uniform(0, 1)
    # 使用 kmeans++ 方法选择初始质心
    for i in range(k):
        if i == 0:
            init[i, :] = data[torch.randint(data.shape[0], [1])]
        else:
            D2 = torch.cdist(init[:i, :][None, :], data[None, :], p=2)[0].amin(dim=0)
            probs = D2 / torch.sum(D2)
            cumprobs = torch.cumsum(probs, dim=0)
            init[i, :] = data[torch.searchsorted(cumprobs, r.sample([1]).to(data.device))]
    return init

# 定义一个名为 KMeansGPU 的类，实现了 Kmeans 聚类算法
class KMeansGPU:
  '''
  Kmeans clustering algorithm implemented with PyTorch

  Parameters:
    n_clusters: int, 
      Number of clusters

    max_iter: int, default: 100
      Maximum number of iterations

    tol: float, default: 0.0001
      Tolerance
  '''
    verbose: int, default: 0
      # 定义一个整型变量 verbose，表示详细程度，默认为 0

    mode: {'euclidean', 'cosine'}, default: 'euclidean'
      # 定义一个字符串变量 mode，表示距离度量的类型，默认为 'euclidean'

    init_method: {'random', 'point', '++'}
      # 定义一个字符串变量 init_method，表示初始化的类型

    minibatch: {None, int}, default: None
      # 定义一个整型变量 minibatch，表示MinibatchKmeans算法的批处理大小，如果为 None，则执行完整的 KMeans 算法

  Attributes:
    centroids: torch.Tensor, shape: [n_clusters, n_features]
      # 聚类中心点

  '''
  def __init__(self, n_clusters, max_iter=200, tol=1e-4, verbose=0, mode="euclidean",device=torch.device("cuda:0")):
    # 初始化方法，设置聚类数、最大迭代次数、容忍度、详细程度、距离度量类型和设备
    self.n_clusters = n_clusters
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.mode = mode
    self.device=device
    # 初始化 GPU 相关信息
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    # 计算 minibatch 的大小
    self.minibatch=int(33e6/self.n_clusters*info.free/ 1024 / 1024 / 1024)
    print("free_mem/GB:",info.free/ 1024 / 1024 / 1024,"minibatch:",self.minibatch)
    
  @staticmethod
  def cos_sim(a, b):
    """
      Compute cosine similarity of 2 sets of vectors

      Parameters:
      a: torch.Tensor, shape: [m, n_features]

      b: torch.Tensor, shape: [n, n_features]
    """
    # 计算两组向量的余弦相似度
    return normalize(a, dim=-1) @ normalize(b, dim=-1).transpose(-2, -1)

  @staticmethod
  def euc_sim(a, b):
    """
      Compute euclidean similarity of 2 sets of vectors
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    # 计算两组向量的欧氏相似度
    return 2 * a @ b.transpose(-2, -1) -(a**2).sum(dim=1)[..., :, None] - (b**2).sum(dim=1)[..., None, :]

  def max_sim(self, a, b):
    """
      Compute maximum similarity (or minimum distance) of each vector
      in a with all of the vectors in b
      Parameters:
      a: torch.Tensor, shape: [m, n_features]
      b: torch.Tensor, shape: [n, n_features]
    """
    # 计算向量 a 中每个向量与向量 b 中所有向量的最大相似度（或最小距离）
    if self.mode == 'cosine':
      sim_func = self.cos_sim
    # 如果聚类模式为欧几里得距离，则使用欧几里得相似度函数
    elif self.mode == 'euclidean':
      sim_func = self.euc_sim
    # 计算相似度
    sim = sim_func(a, b)
    # 获取最大相似度值和对应的索引
    max_sim_v, max_sim_i = sim.max(dim=-1)
    # 返回最大相似度值和对应的索引
    return max_sim_v, max_sim_i

  # 组合fit()和predict()方法，比分别调用fit()和predict()更快
  def fit_predict(self, X):
    """
      Combination of fit() and predict() methods.
      This is faster than calling fit() and predict() seperately.
      Parameters:
      X: torch.Tensor, shape: [n_samples, n_features]
      centroids: {torch.Tensor, None}, default: None
        if given, centroids will be initialized with given tensor
        if None, centroids will be randomly chosen from X
      Return:
      labels: torch.Tensor, shape: [n_samples]

            mini_=33kk/k*remain
            mini=min(mini_,fea_shape)
            offset=log2(k/1000)*1.5
            kpp_all=min(mini_*10/offset,fea_shape)
            kpp_sample=min(mini_/12/offset,fea_shape)
    """
    # 断言输入必须是torch.Tensor类型
    assert isinstance(X, torch.Tensor), "input must be torch.Tensor"
    # 断言输入必须是浮点数类型
    assert X.dtype in [torch.half, torch.float, torch.double], "input must be floating point"
    # 断言输入必须是二维张量，形状为[n_samples, n_features]
    assert X.ndim == 2, "input must be a 2d tensor with shape: [n_samples, n_features] "
    # 计算偏移量
    offset = np.power(1.5,np.log(self.n_clusters / 1000))/np.log(2)
    # 返回最接近的点
    return closest
```