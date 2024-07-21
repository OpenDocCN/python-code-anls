# `.\pytorch\torch\_inductor\codegen\triton_split_scan.py`

```
# mypy: allow-untyped-defs
import functools  # 导入 functools 模块，用于高阶函数的支持

from typing import Optional, Set  # 导入类型提示相关的模块

import torch._inductor.runtime.hints  # 导入 Torch 的归纳器运行时提示模块
from torch._inductor import config  # 导入 Torch 归纳器的配置模块
from torch._inductor.codegen.simd import IterationRangesRoot  # 导入 SIMD 相关的迭代范围根节点类

from torch._inductor.codegen.triton import triton_compute_type, TritonKernel  # 导入 Triton 相关的计算类型和 TritonKernel 类

from torch._prims_common import prod  # 导入 Torch 公共基元模块中的 prod 函数

from torch.utils._sympy.functions import CeilDiv  # 导入 Torch 工具中的 CeilDiv 函数


class TritonSplitScanKernel(TritonKernel):
    """Generates a triton kernel that supports ops.scan calls while also splitting
    the reduction dimension over multiple triton programs.

    For this kernel, loop numels will always take the form ``(xdim, rdim)``
    and the grid has the shape ``(CeilDiv(rdim, RBLOCK), xdim)``. Communication
    between blocks occurs within a global memory workspace buffer, which
    must be zero-filled before launching the kernel.

    Note that generation for ``ops.reduction`` is not supported.

    For details of the communication strategy, see
    https://research.nvidia.com/publication/2016-03_single-pass-parallel-prefix-scan-decoupled-look-back

    """

    def __init__(
        self,
        *groups,  # 可变位置参数，接收一系列分组参数
        index_dtype: str,  # 索引数据类型，字符串类型
        mutations: Optional[Set[str]] = None,  # 可选的变异集合参数，默认为 None
        reduction_hint=torch._inductor.runtime.hints.ReductionHint.DEFAULT,  # 归纳提示，默认为 Torch 归纳器的默认提示
        min_elem_per_thread=0,  # 每个线程的最小元素数，默认为 0
    ):
        super().__init__(
            *groups,  # 调用父类 TritonKernel 的构造函数，传递所有分组参数
            index_dtype=index_dtype,  # 设置索引数据类型
            mutations=mutations,  # 设置变异集合参数
            pid_cache=None,  # 进程 ID 缓存设置为 None
            reduction_hint=reduction_hint,  # 设置归纳提示
            min_elem_per_thread=min_elem_per_thread,  # 设置每个线程的最小元素数
        )
        self.no_x_dim = True  # 初始化 TritonSplitScanKernel 实例的 no_x_dim 属性为 True

    def initialize_range_tree(self, pid_cache):
        prefixes = "yxr"  # 设置前缀字符串
        assert len(self.numels) <= len(
            prefixes
        ), "z dimension not supported for split scan"  # 断言确保 numel 数量不超过前缀字符串长度，否则抛出异常提示 z 维度不支持拆分扫描

        active_prefixes = prefixes[len(prefixes) - len(self.numels) :]  # 获取有效前缀，长度与 numel 数量对应

        grid_dims = "rxy"  # 设置网格维度字符串
        for numel, prefix in zip(self.numels, active_prefixes):
            is_reduction = prefix == "r"  # 判断是否为归纳维度
            tensor_dim = 0 if is_reduction else None  # 如果是归纳维度则设置张量维度为 0，否则为 None
            grid_dim = grid_dims.find(prefix)  # 获取前缀在网格维度字符串中的索引
            self.range_trees.append(
                IterationRangesRoot(
                    f"{prefix}index",  # 迭代范围的名称
                    numel,  # 迭代范围的元素数
                    prefix,  # 迭代范围的前缀
                    grid_dim,  # 迭代范围的网格维度
                    self,  # TritonSplitScanKernel 实例本身
                    pid_cache=pid_cache,  # 进程 ID 缓存
                    is_loop=False,  # 不是循环
                    tensor_dim=tensor_dim,  # 张量维度
                    grid_dim=grid_dim,  # 网格维度
                    has_zdim=False,  # 没有 z 维度
                )
            )
        for tree in self.range_trees:
            self.iteration_ranges_codegen_header(tree, self.body)  # 为每棵迭代范围树生成代码头部

    def reduction(self, dtype, src_dtype, reduction_type, value):
        raise NotImplementedError("NYI TritonSplitDimKernel reductions")  # 抛出未实现异常，提示 TritonSplitDimKernel 不支持归纳操作

    def _get_heuristic(self):
        return "split_scan"  # 返回启发式方法名称 "split_scan"

    def _get_grid_fn(self):
        return "split_scan_grid"  # 返回网格函数名称 "split_scan_grid"
```