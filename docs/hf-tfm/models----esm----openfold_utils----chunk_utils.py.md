# `.\models\esm\openfold_utils\chunk_utils.py`

```py
# 导入日志和数学库
import logging
import math
# 导入偏函数模块
from functools import partial
# 导入类型提示
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

# 导入 PyTorch 库
import torch

# 导入自定义模块中的函数
from .tensor_utils import tensor_tree_map, tree_map

# 定义一个函数，根据树形结构获取张量的维度信息
def _fetch_dims(tree: Union[dict, list, tuple, torch.Tensor]) -> List[Tuple[int, ...]]:
    # 初始化空列表，用于存储所有维度信息
    shapes = []
    # 如果输入是字典，则递归获取每个值的维度信息
    if isinstance(tree, dict):
        for v in tree.values():
            shapes.extend(_fetch_dims(v))
    # 如果输入是列表或元组，则递归获取每个元素的维度信息
    elif isinstance(tree, (list, tuple)):
        for t in tree:
            shapes.extend(_fetch_dims(t))
    # 如果输入是 PyTorch 张量，则获取其维度信息并添加到列表中
    elif isinstance(tree, torch.Tensor):
        shapes.append(tree.shape)
    else:
        # 如果输入类型不支持，则抛出 ValueError
        raise ValueError("Not supported")

    # 返回所有获取到的维度信息列表
    return shapes


# 使用 Torch 的 JIT 功能忽略该函数，不进行 JIT 编译
@torch.jit.ignore
# 定义一个函数，将扁平索引转换为多维索引
def _flat_idx_to_idx(flat_idx: int, dims: Tuple[int, ...]) -> Tuple[int, ...]:
    # 初始化空列表，用于存储多维索引
    idx = []
    # 从后向前遍历维度元组
    for d in reversed(dims):
        # 将当前扁平索引对应的维度索引加入到列表中
        idx.append(flat_idx % d)
        # 更新扁平索引，准备处理下一个维度
        flat_idx = flat_idx // d

    # 返回反转后的多维索引元组
    return tuple(reversed(idx))


# 使用 Torch 的 JIT 功能忽略该函数，不进行 JIT 编译
@torch.jit.ignore
# 定义一个函数，获取最小的切片集合
def _get_minimal_slice_set(
    start: Sequence[int],
    end: Sequence[int],
    dims: Sequence[int],
    start_edges: Optional[Sequence[bool]] = None,
    end_edges: Optional[Sequence[bool]] = None,
) -> List[Tuple[slice, ...]]:
    """
    Produces an ordered sequence of tensor slices that, when used in sequence on a tensor with shape dims, yields
    tensors that contain every leaf in the contiguous range [start, end]. Care is taken to yield a short sequence of
    slices, and perhaps even the shortest possible (I'm pretty sure it's the latter).

    end is INCLUSIVE.
    """

    # 如果未提供起始边缘信息，则初始化为从顶部开始的边缘
    if start_edges is None:
        start_edges = [s == 0 for s in start]
        # 减少边缘列表，确保每个维度是否为顶部边缘
        reduce_edge_list(start_edges)
    
    # 如果未提供结束边缘信息，则初始化为从底部结束的边缘
    if end_edges is None:
        end_edges = [e == (d - 1) for e, d in zip(end, dims)]
        # 减少边缘列表，确保每个维度是否为底部边缘
        reduce_edge_list(end_edges)

    # 基本情况：如果起始索引为空，则返回空切片元组
    if len(start) == 0:
        return [()]
    # 如果起始和结束的维度长度为1，直接返回包含该范围的切片元组的列表
    elif len(start) == 1:
        return [(slice(start[0], end[0] + 1),)]

    # 初始化空列表用于存储切片元组
    slices: List[Tuple[slice, ...]] = []
    # 初始化空列表用于存储路径切片
    path_list: List[slice] = []

    # 遍历起始和结束的维度，找出可以直接选择的公共路径
    for s, e in zip(start, end):
        if s == e:
            path_list.append(slice(s, s + 1))  # 如果起始和结束相同，直接选择这一维度的切片
        else:
            break

    # 将路径切片转换为元组
    path: Tuple[slice, ...] = tuple(path_list)
    # 确定分歧点的索引
    divergence_idx = len(path)

    # 如果起始和结束完全相同，直接返回路径切片的列表
    if divergence_idx == len(dims):
        return [path]

    # 定义用于处理上界情况的函数
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

    # 定义用于处理下界情况的函数
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

    # 如果起始和结束都在分叉点的子树边缘上，直接选择整个子树的切片
    if start_edges[divergence_idx] and end_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx] + 1),))
    # 如果只有起始在边缘上，选择几乎整个子树，最后一个边缘情况单独处理
    elif start_edges[divergence_idx]:
        slices.append(path + (slice(start[divergence_idx], end[divergence_idx]),))
        slices.extend(lower())
    # 如果只有结束在边缘上，选择上半部分子树，最后一个边缘情况单独处理
    elif end_edges[divergence_idx]:
        slices.extend(upper())
        slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx] + 1),))
    # 如果起始和结束都不在边缘上，需要分别处理两边，中间部分可以一次性索引
    else:
        slices.extend(upper())
        middle_ground = end[divergence_idx] - start[divergence_idx]
        if middle_ground > 1:
            slices.append(path + (slice(start[divergence_idx] + 1, end[divergence_idx]),))
        slices.extend(lower())

    return slices
@torch.jit.ignore
# 标记此函数在Torch的即时编译中应被忽略
def _chunk_slice(t: torch.Tensor, flat_start: int, flat_end: int, no_batch_dims: int) -> torch.Tensor:
    """
    Equivalent to

        t.reshape((-1,) + t.shape[no_batch_dims:])[flat_start:flat_end]

    but without the need for the initial reshape call, which can be memory-intensive in certain situations. The only
    reshape operations in this function are performed on sub-tensors that scale with (flat_end - flat_start), the chunk
    size.
    """
    # 将输入张量的批处理维度保存到batch_dims中
    batch_dims = t.shape[:no_batch_dims]
    # 将flat_start转换为索引，_flat_idx_to_idx返回的是生成器，将其转换为列表
    start_idx = list(_flat_idx_to_idx(flat_start, batch_dims))
    # flat_end - 1转换为索引
    end_idx = list(_flat_idx_to_idx(flat_end - 1, batch_dims))

    # 获取一个有序的切片列表以执行
    slices = _get_minimal_slice_set(
        start_idx,
        end_idx,
        batch_dims,
    )

    # 对切片后的张量列表进行操作
    sliced_tensors = [t[s] for s in slices]

    # 拼接切片后的张量，并重新调整形状，以匹配原始批处理维度之后的形状
    return torch.cat([s.view((-1,) + t.shape[no_batch_dims:]) for s in sliced_tensors])


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
    # 如果没有提供输入，则引发值错误
    if not (len(inputs) > 0):
        raise ValueError("Must provide at least one input")

    # 从输入中提取初始维度，并确定原始批处理维度
    initial_dims = [shape[:no_batch_dims] for shape in _fetch_dims(inputs)]
    orig_batch_dims = tuple([max(s) for s in zip(*initial_dims)])

    def _prep_inputs(t: torch.Tensor) -> torch.Tensor:
        # 如果low_mem为False，扩展输入张量的形状以匹配原始批处理维度，并重新调整形状
        if not low_mem:
            if not sum(t.shape[:no_batch_dims]) == no_batch_dims:
                t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
            t = t.reshape(-1, *t.shape[no_batch_dims:])
        else:
            # 否则，仅扩展输入张量的形状以匹配原始批处理维度
            t = t.expand(orig_batch_dims + t.shape[no_batch_dims:])
        return t

    # 对输入进行准备处理，并应用于输入字典的所有叶子张量
    prepped_inputs: Dict[str, Any] = tensor_tree_map(_prep_inputs, inputs)
    # 初始化预处理输出为 None
    prepped_outputs = None
    # 如果 _out 不为 None，则对 _out 中的每个张量应用 lambda 函数，将其展平并保留其余维度
    if _out is not None:
        prepped_outputs = tensor_tree_map(lambda t: t.view([-1] + list(t.shape[no_batch_dims:])), _out)

    # 计算扁平化批次维度的初始值为 1
    flat_batch_dim = 1
    # 遍历原始批次维度列表，计算总的扁平化批次维度
    for d in orig_batch_dims:
        flat_batch_dim *= d

    # 计算需要的块数，即扁平化批次维度除以块大小，如果有余数则增加一个块
    no_chunks = flat_batch_dim // chunk_size + (flat_batch_dim % chunk_size != 0)

    # 定义一个函数，用于从张量中选择一个块
    def _select_chunk(t: torch.Tensor) -> torch.Tensor:
        return t[i : i + chunk_size] if t.shape[0] != 1 else t

    # 初始化块的起始索引 i 为 0，输出 out 为预处理输出
    i = 0
    out = prepped_outputs
    # 对于每个块的迭代
    for _ in range(no_chunks):
        # 如果不使用低内存选项，则选择的块为 _select_chunk 函数，否则为 _chunk_slice 函数的部分应用
        if not low_mem:
            select_chunk = _select_chunk
        else:
            select_chunk = partial(
                _chunk_slice,
                flat_start=i,
                flat_end=min(flat_batch_dim, i + chunk_size),
                no_batch_dims=len(orig_batch_dims),
            )

        # 对预处理输入的每个张量应用 select_chunk 函数，得到块的字典 chunks
        chunks: Dict[str, Any] = tensor_tree_map(select_chunk, prepped_inputs)

        # 在当前块上运行层操作，得到输出块 output_chunk
        output_chunk = layer(**chunks)

        # 如果输出 out 为 None，则根据 output_chunk 的形状创建全零张量作为 out
        if out is None:
            out = tensor_tree_map(lambda t: t.new_zeros((flat_batch_dim,) + t.shape[1:]), output_chunk)

        # 将 output_chunk 中的数据放入预先分配的空间中
        if isinstance(output_chunk, dict):
            # 如果 output_chunk 是字典，则递归地将其值分配给 out 中对应的项
            def assign(d1: dict, d2: dict) -> None:
                for k, v in d1.items():
                    if isinstance(v, dict):
                        assign(v, d2[k])
                    else:
                        # 根据 _add_into_out 标志选择是否加法或赋值操作
                        if _add_into_out:
                            v[i : i + chunk_size] += d2[k]
                        else:
                            v[i : i + chunk_size] = d2[k]

            assign(out, output_chunk)
        elif isinstance(output_chunk, tuple):
            # 如果 output_chunk 是元组，则对应元素逐一处理
            for x1, x2 in zip(out, output_chunk):
                if _add_into_out:
                    x1[i : i + chunk_size] += x2
                else:
                    x1[i : i + chunk_size] = x2
        elif isinstance(output_chunk, torch.Tensor):
            # 如果 output_chunk 是张量，则根据 _add_into_out 标志选择是否加法或赋值操作
            if _add_into_out:
                out[i : i + chunk_size] += output_chunk
            else:
                out[i : i + chunk_size] = output_chunk
        else:
            # 如果 output_chunk 类型不支持，则引发错误
            raise ValueError("Not supported")

        # 更新块的起始索引 i
        i += chunk_size

    # 将 out 中的每个张量重新调整形状，恢复原始批次维度
    out = tensor_tree_map(lambda t: t.view(orig_batch_dims + t.shape[1:]), out)

    # 返回最终的输出结果 out
    return out
    # 定义一个用于调整块大小的类
    class ChunkSizeTuner:
        def __init__(
            self,
            # 最大块大小，默认为512，基于实验观察到大多数模型在所有GPU上的运行时会在此之前达到平台期。
            max_chunk_size: int = 512,
        ):
            self.max_chunk_size = max_chunk_size
            self.cached_chunk_size: Optional[int] = None  # 缓存的块大小，初始为None
            self.cached_arg_data: Optional[tuple] = None  # 缓存的参数数据，初始为None

        def _determine_favorable_chunk_size(self, fn: Callable, args: tuple, min_chunk_size: int) -> int:
            # 记录调整块大小的过程
            logging.info("Tuning chunk size...")

            # 如果最小块大小已经大于等于最大块大小，直接返回最小块大小
            if min_chunk_size >= self.max_chunk_size:
                return min_chunk_size

            # 创建候选块大小列表，从最小块大小开始到不超过最大块大小，以2的指数增长
            candidates: List[int] = [2**l for l in range(int(math.log(self.max_chunk_size, 2)) + 1)]
            candidates = [c for c in candidates if c > min_chunk_size]
            candidates = [min_chunk_size] + candidates
            candidates[-1] += 4

            # 测试每个候选块大小是否可行
            def test_chunk_size(chunk_size: int) -> bool:
                try:
                    with torch.no_grad():
                        fn(*args, chunk_size=chunk_size)
                    return True
                except RuntimeError:
                    return False

            # 初始化最小可行块大小的索引
            min_viable_chunk_size_index = 0
            i = len(candidates) - 1
            # 二分搜索找到最小的可行块大小
            while i > min_viable_chunk_size_index:
                viable = test_chunk_size(candidates[i])
                if not viable:
                    i = (min_viable_chunk_size_index + i) // 2
                else:
                    min_viable_chunk_size_index = i
                    i = (i + len(candidates) - 1) // 2

            # 返回最小的可行块大小
            return candidates[min_viable_chunk_size_index]

        def _compare_arg_caches(self, ac1: Iterable, ac2: Iterable) -> bool:
            # 比较两个参数缓存是否一致
            consistent = True
            for a1, a2 in zip(ac1, ac2):
                assert type(ac1) == type(ac2)
                if isinstance(ac1, (list, tuple)):
                    consistent &= self._compare_arg_caches(a1, a2)
                elif isinstance(ac1, dict):
                    # 将字典按键排序后比较值是否一致
                    a1_items = [v for _, v in sorted(a1.items(), key=lambda x: x[0])]
                    a2_items = [v for _, v in sorted(a2.items(), key=lambda x: x[0])]
                    consistent &= self._compare_arg_caches(a1_items, a2_items)
                else:
                    consistent &= a1 == a2

            return consistent

        def tune_chunk_size(
            self,
            representative_fn: Callable,
            args: tuple,
            min_chunk_size: int,
        ) -> int:
        # 定义一个方法，其返回类型为整数
        consistent = True
        # 初始化一个布尔变量 consistent 为 True
        arg_data: tuple = tree_map(lambda a: a.shape if isinstance(a, torch.Tensor) else a, args, object)
        # 使用 tree_map 函数，将参数 args 中的每个元素转换成其形状（如果是 torch.Tensor 对象）或原始值，并存储在 arg_data 中
        if self.cached_arg_data is not None:
            # 如果已经有缓存的参数数据
            assert len(self.cached_arg_data) == len(arg_data)
            # 断言缓存的参数数据与当前参数数据的长度相等
            consistent = self._compare_arg_caches(self.cached_arg_data, arg_data)
            # 调用 _compare_arg_caches 方法比较缓存的参数数据和当前参数数据，更新 consistent 变量
        else:
            # 如果没有缓存的参数数据
            # 此时需要重新计算
            consistent = False

        if not consistent:
            # 如果参数数据不一致
            self.cached_chunk_size = self._determine_favorable_chunk_size(
                representative_fn,
                args,
                min_chunk_size,
            )
            # 调用 _determine_favorable_chunk_size 方法计算出合适的块大小，并存储在 cached_chunk_size 中
            self.cached_arg_data = arg_data
            # 更新缓存的参数数据为当前参数数据

        assert self.cached_chunk_size is not None
        # 断言 cached_chunk_size 不为 None

        return self.cached_chunk_size
        # 返回 cached_chunk_size
```