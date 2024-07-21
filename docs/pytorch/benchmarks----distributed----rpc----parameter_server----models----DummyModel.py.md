# `.\pytorch\benchmarks\distributed\rpc\parameter_server\models\DummyModel.py`

```
# 导入 PyTorch 中的神经网络模块和函数模块
import torch.nn as nn
import torch.nn.functional as F

# 定义一个名为 DummyModel 的类，继承自 nn.Module 类
class DummyModel(nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        dense_input_size: int,
        dense_output_size: int,
        dense_layers_count: int,
        sparse: bool,
    ):
        r"""
        A dummy model with an EmbeddingBag Layer and Dense Layer.
        Args:
            num_embeddings (int): size of the dictionary of embeddings
                嵌入词典的大小，即嵌入向量的数量
            embedding_dim (int): the size of each embedding vector
                每个嵌入向量的维度大小
            dense_input_size (int): size of each input sample
                每个输入样本的大小
            dense_output_size (int):  size of each output sample
                每个输出样本的大小
            dense_layers_count: (int): number of dense layers in dense Sequential module
                密集层的数量，将创建这么多层的线性层
            sparse (bool): if True, gradient w.r.t. weight matrix will be a sparse tensor
                如果为 True，则权重矩阵的梯度将是稀疏张量
        """
        super().__init__()  # 调用父类的构造函数

        # 创建一个 EmbeddingBag 层，用于处理不定长的输入序列
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, sparse=sparse)

        # 创建一个 Sequential 容器，用于组合多个线性层
        self.dense = nn.Sequential(
            *[
                nn.Linear(dense_input_size, dense_output_size)
                for _ in range(dense_layers_count)
            ]
        )

    def forward(self, x):
        # 将输入 x 经过 EmbeddingBag 层处理后的输出
        x = self.embedding(x)
        # 对经过 dense Sequential 模块处理后的输出进行 softmax 操作
        return F.softmax(self.dense(x), dim=1)
```