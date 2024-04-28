# `.\models\esm\openfold_utils\chunk_utils.py`

```
# 版权声明和许可证信息
# 2021年AlQuraishi实验室所有
#
# 根据Apache许可证2.0版（“许可证”）的规定授权；
# 除非符合许可证的规定，否则不能使用此文件。
# 您可以获取许可证的副本，网址如上所示
#
# 除非法律另有规定或得到书面同意，否则根据许可证分发的软件基于“现状”分发，
# 没有任何明示或暗示的担保或条件。
# 请查阅许可证了解具体的语言权限和限制。
import logging  # 导入logging模块
import math  # 导入math模块
from functools import partial  # 从functools模块中导入partial函数
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union  # 导入类型提示

import torch  # 导入torch模块

from .tensor_utils import tensor_tree_map, tree_map  # 从自定义模块中导入tensor_tree_map和tree_map函数


def _fetch_dims(tree: Union[dict, list, tuple, torch.Tensor]) -> List[Tuple[int, ...]]:
    """从树形结构中获取维度信息"""
    shapes = []  # 创建一个空列表，用于存储维度信息
    if isinstance(tree, dict):  # 如果tree是字典类型
        for v in tree.values():  # 遍历tree中的值
            shapes.extend(_fetch_dims(v))  # 递归调用_fetch_dims函数，将结果合并到shapes中
    elif isinstance(tree, (list, tuple)):  # 如果tree是列表或元组类型
        for t in tree:  # 遍历tree中的元素
            shapes.extend(_fetch_dims(t))  # 递归调用_fetch_dims函数，将结果合并到shapes中
    elif isinstance(tree, torch.Tensor):  # 如果tree是torch张量类型
        shapes.append(tree.shape)  # 将张量的形状信息添加到shapes中
    else:  # 如果tree不是以上类型
        raise ValueError("Not supported")  # 抛出数值错误异常，提示不支持的类型

    return shapes  # 返回维度信息的列表


@torch.jit.ignore
def _flat_idx_to_idx(flat_idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    """将扁平化的索引转换为索引元组"""
    idx = []  # 创建一个空列表，用于存储索引
    for d in reversed(dims):  # 遍历反转后的维度元组
        idx.append(flat_idx % d)  # 计算当前维度上的索引并添加到idx中
        flat_idx = flat_idx // d  # 更新flat_idx为除以当前维度的结果

    return tuple(reversed(idx))  # 返回逆序的索引元组


@torch.jit.ignore
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: Sequence[int],
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> List[Tuple[slice, ...]]:
    """
    获取最小切片集合，用于提取包含指定范围内所有叶子节点的子张量
    """
    # 如果start_edges和end_edges为None，则将其初始化为True/False的列表
    def reduce_edge_list(l: List[bool]) -> None:
        tally = True  # 初始化tally为True
        for i in range(len(l)):  # 遍历列表l的索引
            reversed_idx = -1 * (i + 1)  # 计算反转后的索引
            l[reversed_idx] &= tally  # 通过与运算更新列表中的值
            tally = l[reversed_idx]  # 更新tally为当前值

    if start_edges is None:  # 如果start_edges为None
        start_edges = [s == 0 for s in start]  # 将start中的元素是否为0的结果赋值给start_edges
        reduce_edge_list(start_edges)  # 调用reduce_edge_list函数
    if end_edges is None:  # 如果end_edges为None
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]  # 将end中的元素等于(d-1)的结果赋值给end_edges
        reduce_edge_list(end_edges)  # 调用reduce_edge_list函数

    # 基本情况。如果start/end为空，或者维度是一维的，则直接返回空切片
    if len(start) == 0:  # 如果start为空
        return [()]  # 返回空切片的列表
    # 如果起始索引和结束索引长度相等，说明只有一个维度，直接返回一个包含切片的元组列表，每个切片从起始索引到结束索引
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    # 初始化切片列表和路径列表
    slices: List[Tuple[slice, ...]] = []
    path_list: List[slice] = []

    # 对于起始索引和结束索引中共同的维度，可以直接选择
    for s, e in zip(start, end):
        # 如果起始索引和结束索引相等，直接将切片加入路径列表
        if s == e:
            path_list.append(slice(s, s + 1))
        else:
            # 如果不相等，退出循环
            break

    # 将路径列表转换为元组
    path: Tuple[slice, ...] = tuple(path_list)
    # 记录分歧索引位置
    divergence_idx = len(path)

    # 如果起始索引和结束索引完全相等，直接返回路径列表
    if divergence_idx == len(dims):
        return [path]

    # 定义一个函数，用于计算上界
    def upper() -> Tuple[Tuple[slice, ...], ...]:
        assert start_edges is not None
        assert end_edges is not None

        sdi = start[divergence_idx]
        return tuple(
            path + (slice(sdi, sdi + 1),) + s
            for s in _get_minimal_slice_set(
                start[divergence_idx + 1 :],
                [d - 1 for d in dims[divergence_idx + 1 :]],
                dims[divergence_idx + 1 :],
                start_edges=start_edges[divergence_idx + 1 :],
                end_edges=[True for _ in end_edges[divergence_idx + 1 :]],
            )
        )

    # 定义一个函数，用于计算下界
    def lower() -> Tuple[Tuple[slice, ...], ...]:
        assert start_edges is not None
        assert end_edges is not None

        edi = end[divergence_idx]
        return tuple(
            path + (slice(edi, edi + 1),) + s
            for s in _get_minimal_slice_set(
                [0 for _ in start[divergence_idx + 1 :]],
                end[divergence_idx + 1 :],
                dims[divergence_idx + 1 :],
                start_edges=[True for _ in start_edges[divergence_idx + 1 :]],
                end_edges=end_edges[divergence_idx + 1 :],
            )
        )

    # 如果起始和结束索引都处于 divergence_idx 指定的子树的边界上，可以一次性选择整个子树
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    # 如果只有起始索引处于边界上，可以几乎选择整个子树，只需将最后一个不完整的边界视为边界情况
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    # 类似于上一个情况，但这次是顶部不完整
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    # 如果范围两侧都不完整，则需要分别处理两侧，如果它们之间有连续的元素，可以将其视为一个大块进行索引
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())

    return slices
# 标记为忽略 Torch JIT 编译器
@torch.jit.ignore
# 定义一个函数，根据给定的参数切分输入张量，并返回切分后的张量
def _chunk_slice(t: torch.Tensor, flat_start: int, flat_end: int, no_batch_dims: int) -> torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be memory-intensive in certain situations. The only
    reshape operations in this function are performed on sub-tensors that scale with (flat_end - flat_start), the chunk
    size.
    """

    # 获取批次维度
    batch_dims = t.shape[:no_batch_dims]
    # 转换扁平化的起始索引
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # _get_minimal_slice_set 相当于是包容的，所以这里 flat_end - 1
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # 获取按顺序执行的切片列表
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    # 对切片后的张量进行切片操作
    sliced_tensors = [t[s] for s in slices]

    # 将切片后的张量拼接起来
    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


# 定义一个函数，实现对输入的层进行分块处理
def chunk_layer(
    layer: Callable,
    inputs: Dict[str, Any],
    chunk_size: int,
    no_batch_dims: int,
    low_mem: bool = False,
    _out: Any = None,
    _add_into_out: bool = False,
) -> Any:
    """
    Implements the "chunking" procedure described in section 1.11.8.

    Layer outputs and inputs are assumed to be simple "pytrees," consisting only of (arbitrarily nested) lists, tuples,
    and dicts with torch.Tensor leaves.

    Args:
        layer:
            The layer to be applied chunk-wise
        inputs:
            A (non-nested) dictionary of keyworded inputs. All leaves must be tensors and must share the same batch
            dimensions.
        chunk_size:
            The number of sub-batches per chunk. If multiple batch dimensions are specified, a "sub-batch" is defined
            as a single indexing of all batch dimensions simultaneously (s.t. the number of sub-batches is the product
            of the batch dimensions).
        no_batch_dims:
            How many of the initial dimensions of each input tensor can be considered batch dimensions.
        low_mem:
            Avoids flattening potentially large input tensors. Unnecessary in most cases, and is ever so slightly
            slower than the default setting.
    Returns:
        The reassembled output of the layer on the inputs.
    """
    # 判断是否有输入
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    # 获取输入张量的初始维度
    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    # 准备输入张量进行分块处理
    def _prep_inputs(t: torch.Tensor) -> torch.Tensor:
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    # 遍历所有输入，准备输入数据
    prepped_inputs: Dict[str, Any] = tensor_tree_map(_prep_inputs, inputs)
    # 初始化一个变量为None
    prepped_outputs = None
    # 如果_out不为空，则对_out中的各个张量进行操作
    if _out is not None:
        prepped_outputs = tensor_tree_map(lambda t: t.view([-1] + list(t.shape[no_batch_dims:])), _out)
    
    # 初始化扁平化的批次维度
    flat_batch_dim = 1
    # 计算扁平化批次维度的大小
    for d in orig_batch_dims:
        flat_batch_dim *= d
    
    # 计算总的块数
    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)
    
    # 定义一个函数，用于选择块
    def _select_chunk(t: torch.Tensor) -> torch.Tensor:
        return t[i : i + chunk_size] if t.shape[0] != 1 else t
    
    # 初始化i为0
    i = 0
    # 初始化输出为prepped_outputs
    out = prepped_outputs
    # 对每个块进行操作
    for _ in range(no_chunks):
        # 如果low_mem为false，则选择整块
        if not low_mem:
            select_chunk = _select_chunk
        # 如果low_mem为true，则选择部分块
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )
    
        # 对输入数据进行分块处理
        chunks: Dict[str, Any] = tensor_tree_map(select_chunk, prepped_inputs)
    
        # 对每个块运行层操作
        output_chunk = layer(**chunks)
    
        # 为输出分配空间
        if out is None:
            out = tensor_tree_map(lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:]), output_chunk)
    
        # 将块放入其预先分配的空间
        if isinstance(output_chunk, dict):
            
            # 递归地将第二个字典的值加到第一个字典中，或者替换第一个字典中的值
            def assign(d1: dict, d2: dict) -> None:
                for k, v in d1.items():
                    if isinstance(v, dict):
                        assign(v, d2[k])
                    else:
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]
    
            assign(out, output_chunk)
        elif isinstance(output_chunk, tuple):
            # 对每一对元组进行操作
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif isinstance(output_chunk, torch.Tensor):
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            raise ValueError("Not supported")
    
        # 更新i的值，指向下一个块的起始位置
        i += chunk_size
    
    # 将out恢复成原始形状
    out = tensor_tree_map(lambda t: t.view(orig_batch_dims + t.shape[1:]), out)
    
    # 返回处理后的输出
    return out
class ChunkSizeTuner:
    # 初始化函数，设置最大的块大小，缓存的块大小和参数数据
    def __init__(
        self,
        # 根据经验，大多数网络中的模块在我运行模型时在所有 GPU 上的运行时间都比这个值早达到平稳期
        max_chunk_size: int = 512,
    ):
        self.max_chunk_size = max_chunk_size
        self.cached_chunk_size: Optional[int] = None
        self.cached_arg_data: Optional[tuple] = None

    # 内部方法，确定最有利的块大小
    def _determine_favorable_chunk_size(self, fn: Callable, args: tuple, min_chunk_size: int) -> int:
        logging.info("Tuning chunk size...")

        # 如果最小块大小大于等于最大块大小，则直接返回最小块大小
        if min_chunk_size >= self.max_chunk_size:
            return min_chunk_size

        # 生成候选的块大小列表
        candidates: List[int] = [2**l for l in range(int(math.log(self.max_chunk_size, 2)) + 1)]
        candidates = [c for c in candidates if c > min_chunk_size]
        candidates = [min_chunk_size] + candidates
        candidates[-1] += 4

        # 测试块大小的方法，判断是否能成功运行
        def test_chunk_size(chunk_size: int) -> bool:
            try:
                with torch.no_grad():
                    fn(*args, chunk_size=chunk_size)
                return True
            except RuntimeError:
                return False

        # 初始化指针和最小有效块大小索引
        min_viable_chunk_size_index = 0
        i = len(candidates) - 1
        # 在候选列表中二分搜索能够成功运行的最大块大小
        while i > min_viable_chunk_size_index:
            viable = test_chunk_size(candidates[i])
            if not viable:
                i = (min_viable_chunk_size_index + i) // 2
            else:
                min_viable_chunk_size_index = i
                i = (i + len(candidates) - 1) // 2

        return candidates[min_viable_chunk_size_index]

    # 比较参数缓存
    def _compare_arg_caches(self, ac1: Iterable, ac2: Iterable) -> bool:
        consistent = True
        # 遍历参数缓存
        for a1, a2 in zip(ac1, ac2):
            assert type(ac1) == type(ac2)
            # 如果参数是列表或元组，递归比较
            if isinstance(ac1, (list, tuple)):
                consistent &= self._compare_arg_caches(a1, a2)
            # 如果参数是字典，按 key 排序后递归比较
            elif isinstance(ac1, dict):
                a1_items = [v for _, v in sorted(a1.items(), key=lambda x: x[0])]
                a2_items = [v for _, v in sorted(a2.items(), key=lambda x: x[0])]
                consistent &= self._compare_arg_caches(a1_items, a2_items)
            else:
                consistent &= a1 == a2

        return consistent

    # 调整块大小的方法
    def tune_chunk_size(
        self,
        representative_fn: Callable,
        args: tuple,
        min_chunk_size: int,
    # 定义函数返回值类型为整数
    ) -> int:
        # 初始化consistent为True
        consistent = True
        # 使用tree_map函数遍历args，如果元素是torch.Tensor类型则返回其shape，否则返回原始元素
        arg_data: tuple = tree_map(lambda a: a.shape if isinstance(a, torch.Tensor) else a, args, object)
        # 如果缓存的参数数据不为空
        if self.cached_arg_data is not None:
            # 如果参数的形状/数值发生了改变，则需要重新调整
            assert len(self.cached_arg_data) == len(arg_data)
            consistent = self._compare_arg_caches(self.cached_arg_data, arg_data)
        else:
            # 否则可以重用预计算的数值
            consistent = False

        # 如果参数发生了改变
        if not consistent:
            # 确定有利的块大小并缓存之
            self.cached_chunk_size = self._determine_favorable_chunk_size(
                representative_fn,
                args,
                min_chunk_size,
            )
            # 缓存参数数据
            self.cached_arg_data = arg_data

        # 检查缓存的块大小不为空
        assert self.cached_chunk_size is not None

        # 返回缓存的块大小
        return self.cached_chunk_size
```