# `.\pytorch\functorch\dim\delayed_mul_tensor.py`

```
# 导入 torch 库，用于张量操作
import torch

# 导入本地模块 _Tensor 和 Tensor，以及 reference 模块中的 _dims, _enable_layers, llist, ltuple
from . import _Tensor, Tensor
from .reference import _dims, _enable_layers, llist, ltuple

# 定义 DelayedMulTensor 类，继承自 _Tensor 类
class DelayedMulTensor(_Tensor):
    def __init__(self, lhs, rhs):
        # 初始化 DelayedMulTensor 对象，接收 lhs 和 rhs 作为输入参数
        self._lhs, self._rhs = lhs, rhs
        self._data = None  # 初始化数据属性 _data
        self._levels_data = None  # 初始化数据属性 _levels_data
        # 判断是否有设备属性，如果 lhs 或 rhs 中有一个有设备，则设为 True
        self._has_device = lhs._has_device or rhs._has_device
        self._batchtensor_data = None  # 初始化数据属性 _batchtensor_data
        self._tensor_data = None  # 初始化数据属性 _tensor_data

    @property
    def _levels(self):
        # 返回延迟乘积张量的层级
        if self._levels_data is None:
            # 如果 _levels_data 为空，则生成新的层级列表，包括 lhs 和 rhs 的所有层级
            levels = llist(self._lhs._levels)
            for l in self._rhs._levels:
                if l not in levels:
                    levels.append(l)
            self._levels_data = ltuple(levels)
        return self._levels_data

    @property
    def _batchtensor(self):
        # 返回批张量的乘积
        if self._batchtensor_data is None:
            # 如果 _batchtensor_data 为空，则启用层级列表，计算 _lhs 和 _rhs 的批张量乘积
            with _enable_layers(self._levels):
                print("bt multiply fallback")  # 打印信息，表示使用默认乘积计算方法
                self._batchtensor_data = self._lhs._batchtensor * self._rhs._batchtensor
        return self._batchtensor_data

    @property
    def _tensor(self):
        # 返回最终的张量数据
        if self._tensor_data is None:
            # 如果 _tensor_data 为空，则根据批张量数据创建 Tensor 对象，并获取其底层张量数据
            self._tensor_data = Tensor.from_batched(
                self._batchtensor, self._has_device
            )._tensor
        return self._tensor_data

    @property
    def ndim(self):
        # 返回批张量的维度
        return self._batchtensor.ndim

    @property
    def dims(self):
        # 返回延迟乘积张量的维度元组
        return ltuple(super().dims)

    def sum(self, dim):
        # 对指定维度进行求和操作
        dims = _dims(dim, 0, False, False)  # 获取指定维度的相关信息
        n = ord("a")  # 获取 ASCII 字符 'a' 的编码值
        all_levels = self._levels  # 获取所有层级信息

        def to_char(d):
            # 将层级字符转换为指定维度字符
            return chr(n + all_levels.index(d))

        plhs, levelslhs = self._lhs._tensor, self._lhs._levels  # 获取左操作数的张量和层级
        prhs, levelsrhs = self._rhs._tensor, self._rhs._levels  # 获取右操作数的张量和层级
        new_dims = tuple(d for d in self.dims if d not in dims)  # 计算新的维度元组
        new_levels = [l for l in self._levels if l not in dims]  # 计算新的层级列表
        fmt = "".join(
            [
                *(to_char(d) for d in levelslhs),  # 左操作数的层级转换为字符
                ",",
                *(to_char(d) for d in levelsrhs),  # 右操作数的层级转换为字符
                "->",
                *(to_char(d) for d in new_levels),  # 新层级转换为字符
            ]
        )
        result_data = torch.einsum(fmt, (plhs, prhs))  # 使用 Einstein 求和约定计算结果张量
        return Tensor.from_positional(result_data, new_levels, True)  # 返回结果张量对象
```