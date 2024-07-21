# `.\pytorch\torch\ao\pruning\_experimental\data_sparsifier\data_norm_sparsifier.py`

```py
# mypy: allow-untyped-defs
# 引入PyTorch库
import torch
# 引入torch.nn.functional模块并重命名为F
from torch.nn import functional as F
# 引入reduce函数
from functools import reduce
# 引入类型提示相关的模块
from typing import Any, List, Optional, Tuple

# 引入base_data_sparsifier模块中的BaseDataSparsifier类
from .base_data_sparsifier import BaseDataSparsifier
# 引入operator模块
import operator

# 定义一个列表，包含当前模块中公开的所有对象名
__all__ = ['DataNormSparsifier']

# 定义一个继承自BaseDataSparsifier的类DataNormSparsifier
class DataNormSparsifier(BaseDataSparsifier):
    # L1-Norm Sparsifier类的文档字符串，描述其功能和操作
    r"""L1-Norm Sparsifier
    This sparsifier computes the *L1-norm* of every sparse block and "zeroes-out" the
    ones with the lowest norm. The level of sparsity defines how many of the
    blocks is removed.
    This sparsifier is controlled by three variables:
    1. `sparsity_level` defines the number of *sparse blocks* that are zeroed-out
    2. `sparse_block_shape` defines the shape of the sparse blocks. Note that
        the sparse blocks originate at the zero-index of the tensor.
    3. `zeros_per_block` is the number of zeros that we are expecting in each
        sparse block. By default we assume that all elements within a block are
        zeroed-out. However, setting this variable sets the target number of
        zeros per block. The zeros within each block are chosen as the *smallest
        absolute values*.
    Args:
        sparsity_level: The target level of sparsity
        sparse_block_shape: The shape of a sparse block
        zeros_per_block: Number of zeros in a sparse block
    Note::
        All arguments to the DataNormSparsifier constructor are "default"
        arguments and could be overriden by the configuration provided in the
        `add_data` step.
    """
    # DataNormSparsifier类的初始化方法
    def __init__(self, data_list: Optional[List[Tuple[str, Any]]] = None, sparsity_level: float = 0.5,
                 sparse_block_shape: Tuple[int, int] = (1, 4),
                 zeros_per_block: Optional[int] = None, norm: str = 'L1'):
        # 如果zeros_per_block参数为None，则计算出默认值，即sparse_block_shape中所有元素的乘积
        if zeros_per_block is None:
            zeros_per_block = reduce(operator.mul, sparse_block_shape)

        # 断言norm参数为'L1'或'L2'，否则抛出异常
        assert norm in ['L1', 'L2'], "only L1 and L2 norm supported at the moment"

        # 定义一个字典，包含默认的参数值
        defaults = {'sparsity_level': sparsity_level, 'sparse_block_shape': sparse_block_shape,
                    'zeros_per_block': zeros_per_block}
        # 将norm参数赋值给self.norm
        self.norm = norm
        # 调用父类BaseDataSparsifier的构造方法，传入data_list参数和defaults字典中的参数
        super().__init__(data_list=data_list, **defaults)

    # 定义一个私有方法__get_scatter_folded_mask，用于生成折叠后的掩码
    def __get_scatter_folded_mask(self, data, dim, indices, output_size, sparse_block_shape):
        # 创建一个与data形状相同的全1张量，并命名为mask
        mask = torch.ones_like(data)
        # 根据给定的维度dim、索引indices，将mask张量的对应位置置为0（即掩码生成）
        mask.scatter_(dim=dim, index=indices, value=0)  # zeroing out
        # 使用F.fold函数将mask张量折叠为指定的output_size形状，kernel_size为sparse_block_shape
        mask = F.fold(mask, output_size=output_size, kernel_size=sparse_block_shape,
                      stride=sparse_block_shape)
        # 将mask张量的数据类型转换为torch.int8类型，并返回生成的掩码张量
        mask = mask.to(torch.int8)
        return mask
    def __get_block_level_mask(self, data,
                               sparse_block_shape, zeros_per_block):
        # 假设数据是一个被压缩的张量（squeeze 过的）
        height, width = data.shape[-2], data.shape[-1]
        block_height, block_width = sparse_block_shape
        values_per_block = block_height * block_width

        # 如果每个块都需要清零，则直接返回一个与 data 形状相同的全零张量
        if values_per_block == zeros_per_block:
            return torch.zeros_like(data, dtype=torch.int8)

        # 计算需要的额外填充高度和宽度以支持块状处理
        dh = (block_height - height % block_height) % block_height
        dw = (block_width - width % block_width) % block_width

        # 创建一个新的填充张量，与 data 类型和设备相同，用 NaN 填充，也可以用 0 替换以保留边缘数据
        padded_data = torch.ones(height + dh, width + dw, dtype=data.dtype, device=data.device)
        padded_data = padded_data * torch.nan  # 这里用 NaN 填充，可以替换为 0 以保留边缘数据
        padded_data[0:height, 0:width] = data

        # 对填充后的张量进行展开操作，以匹配 sparse_block_shape 的内核大小和步长
        unfolded_data = F.unfold(padded_data[None, None, :], kernel_size=sparse_block_shape,
                                 stride=sparse_block_shape)

        # 对展开后的数据按第一维进行排序，获取排序后的索引
        _, sorted_idx = torch.sort(unfolded_data, dim=1)
        sorted_idx = sorted_idx[:, :zeros_per_block, :]  # 清零指定数量的元素

        # 获取 scatter 和 fold 合并后的掩码，对 data 进行块级遮罩处理
        mask = self.__get_scatter_folded_mask(data=unfolded_data, dim=1, indices=sorted_idx, output_size=padded_data.shape,
                                              sparse_block_shape=sparse_block_shape)

        # 去除填充部分并保证连续性，然后压缩掩码的维度
        mask = mask.squeeze(0).squeeze(0)[:height, :width].contiguous()
        return mask
    def __get_data_level_mask(self, data, sparsity_level,
                              sparse_block_shape):
        # 获取数据的高度和宽度
        height, width = data.shape[-2], data.shape[-1]
        # 获取稀疏块的高度和宽度
        block_height, block_width = sparse_block_shape
        # 计算高度和宽度的补齐量
        dh = (block_height - height % block_height) % block_height
        dw = (block_width - width % block_width) % block_width

        # 对数据进行二维平均池化，使用稀疏块的大小作为池化核和步长，启用向上取整模式
        data_norm = F.avg_pool2d(data[None, None, :], kernel_size=sparse_block_shape,
                                 stride=sparse_block_shape, ceil_mode=True)

        # 计算每个稀疏块中的值的数量
        values_per_block = reduce(operator.mul, sparse_block_shape)

        # 展开数据进行处理，以使其具有与稀疏块相似的形状
        data_norm = data_norm.flatten()
        num_blocks = len(data_norm)

        # 将数据进行复制以得到类似展开后的形状
        data_norm = data_norm.repeat(1, values_per_block, 1)  # get similar shape after unfold

        # 对数据进行排序，返回排序后的索引
        _, sorted_idx = torch.sort(data_norm, dim=2)

        # 计算要移除的稀疏块数量的阈值索引
        threshold_idx = round(sparsity_level * num_blocks)
        sorted_idx = sorted_idx[:, :, :threshold_idx]

        # 生成折叠的散射掩码，使用指定的索引和输出大小
        mask = self.__get_scatter_folded_mask(data=data_norm, dim=2, indices=sorted_idx,
                                              output_size=(height + dh, width + dw),
                                              sparse_block_shape=sparse_block_shape)

        # 压缩掩码的维度，只保留高度和宽度的部分
        mask = mask.squeeze(0).squeeze(0)[:height, :width]
        # 返回生成的掩码
        return mask
    # 更新掩码函数，用于更新稀疏掩码
    def update_mask(self, name, data, sparsity_level,
                    sparse_block_shape, zeros_per_block, **kwargs):

        # 计算每个块中的元素个数
        values_per_block = reduce(operator.mul, sparse_block_shape)
        # 检查每个块中零的数量不能超过块中元素的总数
        if zeros_per_block > values_per_block:
            raise ValueError("Number of zeros per block cannot be more than "
                             "the total number of elements in that block.")
        # 检查每个块中零的数量必须是正数
        if zeros_per_block < 0:
            raise ValueError("Number of zeros per block should be positive.")

        # 根据设置的归一化方式计算数据的归一化值
        if self.norm == 'L1':
            data_norm = torch.abs(data).squeeze()  # 基于绝对值的L1范数
        else:
            data_norm = (data * data).squeeze()  # 对每个元素求平方得到L2范数

        # 如果数据维度大于2，目前只支持2维数据
        if len(data_norm.shape) > 2:
            raise ValueError("only supports 2-D at the moment")

        # 如果数据是一维的，将其扩展为二维（用于偏置或一维数据的情况）
        elif len(data_norm.shape) == 1:
            data_norm = data_norm[None, :]

        # 获取当前名称对应的掩码
        mask = self.get_mask(name)

        # 根据稀疏度级别和每个块中零的数量来设置掩码的值
        if sparsity_level <= 0 or zeros_per_block == 0:
            mask.data = torch.ones_like(mask)
        elif sparsity_level >= 1.0 and (zeros_per_block == values_per_block):
            mask.data = torch.zeros_like(mask)

        # 获取数据级别的掩码，将整个块置零
        data_lvl_mask = self.__get_data_level_mask(data=data_norm, sparsity_level=sparsity_level,
                                                   sparse_block_shape=sparse_block_shape)

        # 获取块级别的掩码，将每个块内的一定数量元素置零
        block_lvl_mask = self.__get_block_level_mask(data=data_norm, sparse_block_shape=sparse_block_shape,
                                                     zeros_per_block=zeros_per_block)

        # 根据数据级别掩码和块级别掩码，将掩码中相应位置置零
        mask.data = torch.where(data_lvl_mask == 1, data_lvl_mask, block_lvl_mask)
```