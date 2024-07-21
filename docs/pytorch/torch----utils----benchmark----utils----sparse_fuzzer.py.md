# `.\pytorch\torch\utils\benchmark\utils\sparse_fuzzer.py`

```py
# mypy: allow-untyped-defs
# 引入需要的类型注解和模块
from typing import Optional, Tuple, Union
from numbers import Number
import torch
from torch.utils.benchmark import FuzzedTensor
import math

# 创建一个名为FuzzedSparseTensor的类，继承自FuzzedTensor类
class FuzzedSparseTensor(FuzzedTensor):
    def __init__(
        self,
        name: str,
        size: Tuple[Union[str, int], ...],  # Tensor的大小，可以是字符串或整数组成的元组
        min_elements: Optional[int] = None,  # Tensor最小元素数量的限制
        max_elements: Optional[int] = None,  # Tensor最大元素数量的限制
        dim_parameter: Optional[str] = None,  # 维度参数，用于截断size的长度
        sparse_dim: Optional[str] = None,  # 稀疏Tensor的稀疏维度数目
        nnz: Optional[str] = None,  # 非零元素的数量
        density: Optional[str] = None,  # Tensor的密度，用于生成不同稀疏度的Tensor
        coalesced: Optional[str] = None,  # 是否为紧凑的稀疏Tensor格式
        dtype=torch.float32,  # 生成Tensor的数据类型，默认为float32
        cuda=False  # 是否将Tensor放置在GPU上，默认为False
    ):
        """
        Args:
            name:
                生成Tensor的字符串标识符。
            size:
                描述生成Tensor大小的元组，可以包含整数或字符串。字符串在生成过程中将被替换为具体的整数值。
            min_elements:
                Tensor必须具有的最小参数数量，否则将重新抽样。
            max_elements:
                与`min_elements`类似，但设置了一个上限。
            dim_parameter:
                `size`的长度将被截断为此值。这允许生成具有不同维度的Tensor。
            sparse_dim:
                稀疏Tensor中的稀疏维度数目。
            density:
                此值允许生成具有不同稀疏度的Tensor。
            coalesced:
                稀疏Tensor格式允许未合并的稀疏Tensor，其中索引中可能存在重复的坐标。
            dtype:
                生成Tensor的PyTorch数据类型。
            cuda:
                是否将Tensor放置在GPU上。
        """
        # 调用父类FuzzedTensor的构造方法
        super().__init__(name=name, size=size, min_elements=min_elements,
                         max_elements=max_elements, dim_parameter=dim_parameter, dtype=dtype, cuda=cuda)
        # 设置FuzzedSparseTensor特有的属性
        self._density = density
        self._coalesced = coalesced
        self._sparse_dim = sparse_dim

    @staticmethod
    def sparse_tensor_constructor(size, dtype, sparse_dim, nnz, is_coalesced):
        """sparse_tensor_constructor creates a sparse tensor with coo format.

        Note that when `is_coalesced` is False, the number of elements is doubled but the number of indices
        represents the same amount of number of non zeros `nnz`, i.e, this is virtually the same tensor
        with the same sparsity pattern. Moreover, most of the sparse operation will use coalesce() method
        and what we want here is to get a sparse tensor with the same `nnz` even if this is coalesced or not.

        In the other hand when `is_coalesced` is True the number of elements is reduced in the coalescing process
        by an unclear amount however the probability to generate duplicates indices are low for most of the cases.
        This decision was taken on purpose to maintain the construction cost as low as possible.
        """
        # 如果 size 是一个数值，将其扩展成一个长度为 sparse_dim 的列表
        if isinstance(size, Number):
            size = [size] * sparse_dim
        # 断言所有维度的尺寸大于 0 或者 nnz 为 0，否则抛出异常 'invalid arguments'
        assert all(size[d] > 0 for d in range(sparse_dim)) or nnz == 0, 'invalid arguments'
        # 设置 v_size，作为稀疏张量 v 的大小，将 nnz 放在最前面
        v_size = [nnz] + list(size[sparse_dim:])
        # 根据数据类型，生成随机数据 v
        if dtype.is_floating_point:
            v = torch.rand(size=v_size, dtype=dtype, device="cpu")
        else:
            v = torch.randint(1, 127, size=v_size, dtype=dtype, device="cpu")

        # 生成随机的索引 i，形状为 (sparse_dim, nnz)
        i = torch.rand(sparse_dim, nnz, device="cpu")
        # 缩放索引 i，以匹配 size 的尺寸
        i.mul_(torch.tensor(size[:sparse_dim]).unsqueeze(1).to(i))
        i = i.to(torch.long)

        # 如果 is_coalesced 为 False，复制 v 和 i 以增加元素数量
        if not is_coalesced:
            v = torch.cat([v, torch.randn_like(v)], 0)
            i = torch.cat([i, i], 1)

        # 使用索引 i 和数据 v 创建稀疏 COO 格式的张量 x
        x = torch.sparse_coo_tensor(i, v, torch.Size(size))
        # 如果 is_coalesced 为 True，对张量 x 进行 coalesce 操作
        if is_coalesced:
            x = x.coalesce()
        return x

    def _make_tensor(self, params, state):
        # 获取张量的大小和步长
        size, _, _ = self._get_size_and_steps(params)
        # 计算非零元素的数量 nnz
        density = params['density']
        nnz = math.ceil(sum(size) * density)
        assert nnz <= sum(size)

        # 获取是否进行 coalesce 操作的标志
        is_coalesced = params['coalesced']
        # 获取稀疏维度的数量
        sparse_dim = params['sparse_dim'] if self._sparse_dim else len(size)
        sparse_dim = min(sparse_dim, len(size))
        # 使用 sparse_tensor_constructor 函数创建稀疏张量 tensor
        tensor = self.sparse_tensor_constructor(size, self._dtype, sparse_dim, nnz, is_coalesced)

        # 如果需要在 GPU 上运行，将张量移动到 CUDA 设备上
        if self._cuda:
            tensor = tensor.cuda()
        # 获取稀疏张量的稀疏维度和稠密维度
        sparse_dim = tensor.sparse_dim()
        dense_dim = tensor.dense_dim()
        # 确定张量是否为混合类型
        is_hybrid = len(size[sparse_dim:]) > 0

        # 构建属性字典
        properties = {
            "numel": int(tensor.numel()),  # 张量元素的总数
            "shape": tensor.size(),        # 张量的形状
            "is_coalesced": tensor.is_coalesced(),  # 张量是否已经 coalesced
            "density": density,            # 张量的密度
            "sparsity": 1.0 - density,     # 张量的稀疏度
            "sparse_dim": sparse_dim,      # 稀疏张量的稀疏维度
            "dense_dim": dense_dim,        # 稀疏张量的稠密维度
            "is_hybrid": is_hybrid,        # 张量是否为混合类型
            "dtype": str(self._dtype),     # 张量的数据类型
        }
        return tensor, properties
```