# `.\lucidrains\meshgpt-pytorch\meshgpt_pytorch\data.py`

```
# 导入必要的库
from pathlib import Path
from functools import partial
import torch
from torch import Tensor
from torch import is_tensor
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from numpy.lib.format import open_memmap

from einops import rearrange, reduce

from beartype import beartype
from beartype.typing import Tuple, List, Union, Optional, Callable, Dict, Callable

from torchtyping import TensorType

from pytorch_custom_utils.utils import pad_or_slice_to

# 定义辅助函数

def exists(v):
    return v is not None

def identity(t):
    return t

# 定义常量

Vertices = TensorType['nv', 3, float]   # 3个坐标
Faces = TensorType['nf', 3, int]        # 3个顶点

# 用于自动缓存文本到文本嵌入的装饰器

# 你可以用这个装饰器装饰你的 Dataset 类
# 然后改变你的 `data_kwargs = ["text_embeds", "vertices", "faces"]`

@beartype
def cache_text_embeds_for_dataset(
    embed_texts_fn: Callable[[List[str]], Tensor],
    max_text_len: int,
    cache_path: str = './text_embed_cache'
):
    # 创建缓存文件夹路径

    path = Path(cache_path)
    path.mkdir(exist_ok = True, parents = True)
    assert path.is_dir()

    # 全局内存映射句柄

    text_embed_cache = None
    is_cached = None

    # 缓存函数

    def get_maybe_cached_text_embed(
        idx: int,
        dataset_len: int,
        text: str,
        memmap_file_mode = 'w+'
    ):
        nonlocal text_embed_cache
        nonlocal is_cached

        # 在第一次调用时初始化缓存

        if not exists(text_embed_cache):
            test_embed = embed_texts_fn(['test'])
            feat_dim = test_embed.shape[-1]
            shape = (dataset_len, max_text_len, feat_dim)

            text_embed_cache = open_memmap(str(path / 'cache.text_embed.memmap.npy'), mode = memmap_file_mode, dtype = 'float32', shape = shape)
            is_cached = open_memmap(str(path / 'cache.is_cached.memmap.npy'), mode = memmap_file_mode, dtype = 'bool', shape = (dataset_len,))

        # 确定是从缓存中获取还是调用文本模型

        if is_cached[idx]:
            text_embed = torch.from_numpy(text_embed_cache[idx])
        else:
            # 缓存

            text_embed = get_text_embed(text)
            text_embed = pad_or_slice_to(text_embed, max_text_len, dim = 0, pad_value = 0.)

            is_cached[idx] = True
            text_embed_cache[idx] = text_embed.cpu().numpy()

        mask = ~reduce(text_embed == 0, 'n d -> n', 'all')
        return text_embed[mask]

    # 获取文本嵌入

    def get_text_embed(text: str):
        text_embeds = embed_texts_fn([text])
        return text_embeds[0]

    # 内部函数
    # 定义一个装饰器函数，接受一个数据集类作为参数
    def inner(dataset_klass):
        # 断言数据集类是 Dataset 类的子类
        assert issubclass(dataset_klass, Dataset)

        # 保存原始的 __init__ 和 __getitem__ 方法
        orig_init = dataset_klass.__init__
        orig_get_item = dataset_klass.__getitem__

        # 定义新的 __init__ 方法
        def __init__(
            self,
            *args,
            cache_memmap_file_mode = 'w+',
            **kwargs
        ):
            # 调用原始的 __init__ 方法
            orig_init(self, *args, **kwargs)

            # 设置缓存内存映射文件的模式
            self._cache_memmap_file_mode = cache_memmap_file_mode

            # 如果数据集类有 data_kwargs 属性，则将其中的 'texts' 替换为 'text_embeds'
            if hasattr(self, 'data_kwargs'):
                self.data_kwargs = [('text_embeds' if data_kwarg == 'texts' else data_kwarg) for data_kwarg in self.data_kwargs]

        # 定义新的 __getitem__ 方法
        def __getitem__(self, idx):
            # 调用原始的 __getitem__ 方法
            items = orig_get_item(self, idx)

            # 定义局部函数 get_text_embed_，用于获取可能缓存的文本嵌入
            get_text_embed_ = partial(get_maybe_cached_text_embed, idx, len(self), memmap_file_mode = self._cache_memmap_file_mode)

            # 如果 items 是字典
            if isinstance(items, dict):
                # 如果字典中包含 'texts' 键
                if 'texts' in items:
                    # 获取文本嵌入并替换 'texts' 键为 'text_embeds'
                    text_embed = get_text_embed_(items['texts'])
                    items['text_embeds'] = text_embed
                    del items['texts']

            # 如果 items 是元组
            elif isinstance(items, tuple):
                new_items = []

                # 遍历元组中的每个元素
                for maybe_text in items:
                    # 如果元素不是字符串，则直接添加到新列表中
                    if not isinstance(maybe_text, str):
                        new_items.append(maybe_text)
                        continue

                    # 如果元素是字符串，则获取文本嵌入并添加到新列表中
                    new_items.append(get_text_embed_(maybe_text))

                # 更新 items 为新的元组
                items = tuple(new_items)

            # 返回处理后的 items
            return items

        # 替换数据集类的 __init__ 和 __getitem__ 方法为新定义的方法
        dataset_klass.__init__ = __init__
        dataset_klass.__getitem__ = __getitem__

        # 返回修改后的数据集类
        return dataset_klass

    # 返回装饰器函数 inner
    return inner
# 用于自动缓存面边缘的装饰器

# 你可以用这个函数装饰你的 Dataset 类
# 然后改变你的 `data_kwargs = ["vertices", "faces", "face_edges"]`

@beartype
def cache_face_edges_for_dataset(
    max_edges_len: int,
    cache_path: str = './face_edges_cache',
    assert_edge_len_lt_max: bool = True,
    pad_id = -1
):
    # 创建缓存文件夹路径

    path = Path(cache_path)
    path.mkdir(exist_ok = True, parents = True)
    assert path.is_dir()

    # 全局 memmap 句柄

    face_edges_cache = None
    is_cached = None

    # 缓存函数

    def get_maybe_cached_face_edges(
        idx: int,
        dataset_len: int,
        faces: Tensor,
        memmap_file_mode = 'w+'
    ):
        nonlocal face_edges_cache
        nonlocal is_cached

        if not exists(face_edges_cache):
            # 在第一次调用时初始化缓存

            shape = (dataset_len, max_edges_len, 2)
            face_edges_cache = open_memmap(str(path / 'cache.face_edges_embed.memmap.npy'), mode = memmap_file_mode, dtype = 'float32', shape = shape)
            is_cached = open_memmap(str(path / 'cache.is_cached.memmap.npy'), mode = memmap_file_mode, dtype = 'bool', shape = (dataset_len,))

        # 确定是从缓存中获取还是调用派生面边缘函数

        if is_cached[idx]:
            face_edges = torch.from_numpy(face_edges_cache[idx])
        else:
            # 缓存

            face_edges = derive_face_edges_from_faces(faces, pad_id = pad_id)

            edge_len = face_edges.shape[0]
            assert not assert_edge_len_lt_max or (edge_len <= max_edges_len), f'mesh #{idx} has {edge_len} which exceeds the cache length of {max_edges_len}'

            face_edges = pad_or_slice_to(face_edges, max_edges_len, dim = 0, pad_value = pad_id)

            is_cached[idx] = True
            face_edges_cache[idx] = face_edges.cpu().numpy()

        mask = reduce(face_edges != pad_id, 'n d -> n', 'all')
        return face_edges[mask]

    # 内部函数

    def inner(dataset_klass):
        assert issubclass(dataset_klass, Dataset)

        orig_init = dataset_klass.__init__
        orig_get_item = dataset_klass.__getitem__

        def __init__(
            self,
            *args,
            cache_memmap_file_mode = 'w+',
            **kwargs
        ):
            orig_init(self, *args, **kwargs)

            self._cache_memmap_file_mode = cache_memmap_file_mode

            if hasattr(self, 'data_kwargs'):
                self.data_kwargs.append('face_edges')

        def __getitem__(self, idx):
            items = orig_get_item(self, idx)

            get_face_edges_ = partial(get_maybe_cached_face_edges, idx, len(self), memmap_file_mode = self._cache_memmap_file_mode)

            if isinstance(items, dict):
                face_edges = get_face_edges_(items['faces'])
                items['face_edges'] = face_edges

            elif isinstance(items, tuple):
                _, faces, *_ = items
                face_edges = get_face_edges_(faces)
                items = (*items, face_edges)

            return items

        dataset_klass.__init__ = __init__
        dataset_klass.__getitem__ = __getitem__

        return dataset_klass

    return inner

# 数据集

class DatasetFromTransforms(Dataset):
    @beartype
    def __init__(
        self,
        folder: str,
        transforms: Dict[str, Callable[[Path], Tuple[Vertices, Faces]]],
        data_kwargs: Optional[List[str]] = None,
        augment_fn: Callable = identity
    ):
        folder = Path(folder)
        assert folder.exists and folder.is_dir()
        self.folder = folder

        exts = transforms.keys()
        self.paths = [p for ext in exts for p in folder.glob(f'**/*.{ext}')]

        print(f'{len(self.paths)} training samples found at {folder}')
        assert len(self.paths) > 0

        self.transforms = transforms
        self.data_kwargs = data_kwargs
        self.augment_fn = augment_fn
    # 返回路径列表的长度，即数据集中的样本数量
    def __len__(self):
        return len(self.paths)

    # 根据索引获取数据集中指定位置的样本
    def __getitem__(self, idx):
        # 获取指定索引位置的路径
        path = self.paths[idx]
        # 获取路径的文件扩展名
        ext = path.suffix[1:]
        # 根据文件扩展名获取对应的转换函数
        fn = self.transforms[ext]

        # 使用转换函数处理路径对应的数据
        out = fn(path)
        # 对处理后的数据进行增强处理
        return self.augment_fn(out)
# tensor helper functions

# 从面数据中推导出面的边
def derive_face_edges_from_faces(
    faces: TensorType['b', 'nf', 3, int],  # 输入的面数据，形状为 [batch_size, num_faces, 3, int]
    pad_id = -1,  # 填充值，默认为 -1
    neighbor_if_share_one_vertex = False,  # 如果共享一个顶点则为邻居，默认为 False
    include_self = True  # 是否包括自身，默认为 True
) -> TensorType['b', 'e', 2, int]:  # 返回的面边数据，形状为 [batch_size, num_edges, 2, int]

    is_one_face, device = faces.ndim == 2, faces.device  # 判断是否只有一个面，获取设备信息

    if is_one_face:
        faces = rearrange(faces, 'nf c -> 1 nf c')  # 如果只有一个面，则重排维度为 [1, num_faces, 3, int]

    max_num_faces = faces.shape[1]  # 获取最大面数
    face_edges_vertices_threshold = 1 if neighbor_if_share_one_vertex else 2  # 根据是否共享一个顶点设置阈值

    all_edges = torch.stack(torch.meshgrid(
        torch.arange(max_num_faces, device = device),
        torch.arange(max_num_faces, device = device),
    indexing = 'ij'), dim = -1)  # 创建所有可能的边的组合

    face_masks = reduce(faces != pad_id, 'b nf c -> b nf', 'all')  # 根据填充值生成面的掩码
    face_edges_masks = rearrange(face_masks, 'b i -> b i 1') & rearrange(face_masks, 'b j -> b 1 j')  # 生成面边的掩码

    face_edges = []  # 存储面边数据的列表

    for face, face_edge_mask in zip(faces, face_edges_masks):

        shared_vertices = rearrange(face, 'i c -> i 1 c 1') == rearrange(face, 'j c -> 1 j 1 c')  # 判断是否共享顶点
        num_shared_vertices = shared_vertices.any(dim = -1).sum(dim = -1)  # 统计共享顶点的数量

        is_neighbor_face = (num_shared_vertices >= face_edges_vertices_threshold) & face_edge_mask  # 判断是否为邻居面

        if not include_self:
            is_neighbor_face &= num_shared_vertices != 3  # 排除自身面

        face_edge = all_edges[is_neighbor_face]  # 获取邻居面的边
        face_edges.append(face_edge)  # 添加到面边列表中

    face_edges = pad_sequence(face_edges, padding_value = pad_id, batch_first = True)  # 对面边进行填充

    if is_one_face:
        face_edges = rearrange(face_edges, '1 e ij -> e ij')  # 如果只有一个面，则重排维度

    return face_edges  # 返回面边数据

# custom collater

# 获取列表中的第一个元素
def first(it):
    return it[0]

# 自定义数据集拼接函数
def custom_collate(data, pad_id = -1):
    is_dict = isinstance(first(data), dict)  # 判断数据是否为字典类型

    if is_dict:
        keys = first(data).keys()  # 获取字典的键
        data = [d.values() for d in data]  # 获取字典的值

    output = []  # 存储输出数据的列表

    for datum in zip(*data):
        if is_tensor(first(datum)):
            datum = pad_sequence(datum, batch_first = True, padding_value = pad_id)  # 如果是张量，则进行填充
        else:
            datum = list(datum)  # 否则转换为列表

        output.append(datum)  # 添加到输出列表中

    output = tuple(output)  # 转换为元组

    if is_dict:
        output = dict(zip(keys, output))  # 如果是字典类型，则重新组合为字典

    return output  # 返回拼接后的数据
```