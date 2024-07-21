# `.\pytorch\torch\sparse\semi_structured.py`

```
# 启用类型未标记的函数声明
mypy: allow-untyped-defs

# 导入警告模块
import warnings
# 导入命名元组类
from collections import namedtuple
# 导入类型提示相关的类和函数
from typing import Any, Optional, Tuple, List, Callable, Dict

# 导入PyTorch库
import torch
# 导入稀疏张量的相关转换函数
from torch.sparse._semi_structured_conversions import (
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass
)
# 导入稀疏张量的相关操作函数
from torch.sparse._semi_structured_ops import (
    fallback_dispatcher,
    semi_sparse_values,
    semi_sparse_indices,
    semi_sparse_detach,
    semi_sparse_t,
    semi_sparse_view,
    semi_sparse_mm,
    semi_sparse_addmm,
    semi_sparse_linear,
)

# 公开的模块成员列表
__all__ = [
    "SparseSemiStructuredTensor",
    "SparseSemiStructuredTensorCUTLASS",
    "SparseSemiStructuredTensorCUSPARSELT",
    "to_sparse_semi_structured",
]

# 命名元组用于存储稀疏结构配置
_SEMI_STRUCTURED_SPARSE_CONFIG = namedtuple(
    "_SEMI_STRUCTURED_SPARSE_CONFIG",
    "sparse_min_rows sparse_min_cols dense_min_rows dense_min_cols",
)

# 稀疏半结构化张量类，继承自torch.Tensor
class SparseSemiStructuredTensor(torch.Tensor):
    """
    This class implementes semi-structured sparsity as a Tensor subclass.

    Semi-structured sparsity describes a sparsity pattern where n in every 2n elements are sparse,
    depending on the datatype. It is also referred to as 2:4 sparsity or fine-grained
    structured sparsity.

    There are two backends available for semi_structred sparsity, either cuSPARSELt or CUTLASS.
    This class is meant to serve as a base class for both implementations. SparseSemiStructuredCUTLASS
    and SparseSemiStructuredCUSPARSELT both inherit from this class and define three backend-specific items.
    Note that as such, this class cannot be insantiated directly.

    -`_DTYPE_SHAPE_CONSTRAINTS` - A dictionary holding backend specific dense/sparse min shape constraints
    - `def from_dense()` - backend specific compression routines
    - `def _mm()` - backend specifc mm op (either torch._cslt_sparse_mm or torch._sparse_semi_structured_(mm|addmm))
    """

    # 默认算法标识符
    _DEFAULT_ALG_ID: int = 0
    # 不同数据类型的形状约束字典
    _DTYPE_SHAPE_CONSTRAINTS: Dict[torch.dtype, _SEMI_STRUCTURED_SPARSE_CONFIG]
    # 强制使用Cutlass后端的标志
    _FORCE_CUTLASS: bool = True
    # 是否融合转置操作的标志
    _FUSE_TRANSPOSE: bool = False
    # 原型警告是否已显示的标志
    _PROTOTYPE_WARNING_SHOWN: bool = False

    # 后端类型
    BACKEND: str
    # 稀疏操作的调度函数映射
    SPARSE_DISPATCH: Dict[Callable, Callable]

    # 打包后的张量数据
    packed: Optional[torch.Tensor]
    # 元数据张量
    meta: Optional[torch.Tensor]
    # 转置后的打包张量
    packed_t: Optional[torch.Tensor]
    # 转置后的元数据张量
    meta_t: Optional[torch.Tensor]
    # 压缩的交错位掩码张量
    compressed_swizzled_bitmask: Optional[torch.Tensor]
    # 是否融合转置操作的cuSPARSELt标志
    fuse_transpose_cusparselt: bool
    # cuSPARSELt特定的算法标识符
    alg_id_cusparselt: int

    # 仅限属性列表，减少内存使用
    __slots__ = ["packed", "meta", "packed_t", "meta_t", "compressed_swizzled_bitmask"]

    @staticmethod
    def __new__(  # noqa: PYI034
        cls,
        shape: torch.Size,
        packed: Optional[torch.Tensor],
        meta: Optional[torch.Tensor],
        packed_t: Optional[torch.Tensor],
        meta_t: Optional[torch.Tensor],
        compressed_swizzled_bitmask: Optional[torch.Tensor],
        fuse_transpose_cusparselt: bool = False,
        alg_id_cusparselt: int = 0,
        requires_grad: bool = False,
    def __repr__(self) -> str:  # type: ignore[override]
        # 确保对象有属性 "shape"
        assert hasattr(self, "shape")
        # 返回对象的字符串表示，包括类名和 shape 属性
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __tensor_flatten__(
        self,
    ) -> Tuple[List[str], Tuple[torch.Size, bool, int, bool]]:
        # 过滤出非空的内部张量名称列表
        inner_tensors = list(
            filter(lambda x: getattr(self, x) is not None, self.__slots__)
        )
        # 构建张量的元信息元组
        tensor_meta = (
            self.shape,
            self.fuse_transpose_cusparselt,
            self.alg_id_cusparselt,
            self.requires_grad,
        )
        return inner_tensors, tensor_meta

    @classmethod
    def __tensor_unflatten__(
        cls,
        inner_tensors,
        tensor_meta: Tuple[torch.Size, bool, int, bool],
        outer_size,
        outer_stride,
    ) -> torch.Tensor:
        # 解包张量元信息
        shape, fuse_transpose_cusparselt, alg_id_cusparselt, requires_grad = tensor_meta
        # 返回解包后的类实例
        return cls(
            shape=shape,
            packed=inner_tensors.get("packed", None),
            meta=inner_tensors.get("meta", None),
            packed_t=inner_tensors.get("packed_t", None),
            meta_t=inner_tensors.get("meta_t", None),
            compressed_swizzled_bitmask=inner_tensors.get(
                "compressed_swizzled_bitmask", None
            ),
            fuse_transpose_cusparselt=fuse_transpose_cusparselt,
            alg_id_cusparselt=alg_id_cusparselt,
            requires_grad=requires_grad,
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs) -> Any:
        # 检查是否支持请求的操作
        if func._overloadpacket not in cls.SPARSE_DISPATCH:
            raise NotImplementedError(
                f"{cls.__name__} only supports a specific set of operations, "
                f"can't perform requested op ({func.__name__})"
            )
        # 调用对应的稀疏分派函数并返回结果
        return cls.SPARSE_DISPATCH[func._overloadpacket](func, types, args, kwargs)

    @classmethod
    def _load_dispatch_table(cls, custom_dispatch_table=None) -> None:
        """
        Loads the op overload sparse dispatch table for the current class.
        """
        # 检查当前类是否已经定义了 SPARSE_DISPATCH 属性，如果没有则创建一个空字典
        if getattr(cls, "SPARSE_DISPATCH", None) is None:
            # 定义一个包含了操作重载函数的稀疏分发表
            cls.SPARSE_DISPATCH = {
                torch.ops.aten.values: semi_sparse_values,  # 将 torch.ops.aten.values 映射到 semi_sparse_values 函数
                torch.ops.aten.indices: semi_sparse_indices,  # 将 torch.ops.aten.indices 映射到 semi_sparse_indices 函数
                torch.ops.aten.is_same_size: fallback_dispatcher,  # 将 torch.ops.aten.is_same_size 映射到 fallback_dispatcher 函数
                torch.ops.aten.detach_: fallback_dispatcher,  # 将 torch.ops.aten.detach_ 映射到 fallback_dispatcher 函数
                torch.ops.aten.detach: semi_sparse_detach,  # 将 torch.ops.aten.detach 映射到 semi_sparse_detach 函数
                torch.ops.aten.t: semi_sparse_t,  # 将 torch.ops.aten.t 映射到 semi_sparse_t 函数
                torch.ops.aten.view: semi_sparse_view,  # 将 torch.ops.aten.view 映射到 semi_sparse_view 函数
                torch.ops.aten.mm: semi_sparse_mm,  # 将 torch.ops.aten.mm 映射到 semi_sparse_mm 函数
                torch.ops.aten.matmul: semi_sparse_mm,  # 将 torch.ops.aten.matmul 映射到 semi_sparse_mm 函数
                torch.ops.aten.addmm: semi_sparse_addmm,  # 将 torch.ops.aten.addmm 映射到 semi_sparse_addmm 函数
                torch.ops.aten.linear: semi_sparse_linear,  # 将 torch.ops.aten.linear 映射到 semi_sparse_linear 函数
                torch.ops.aten._to_copy: fallback_dispatcher,  # 将 torch.ops.aten._to_copy 映射到 fallback_dispatcher 函数
            }
            # 如果提供了自定义的分发表，则更新 SPARSE_DISPATCH 属性
            if custom_dispatch_table is not None:
                cls.SPARSE_DISPATCH.update(custom_dispatch_table)

    @classmethod
    def _validate_device_dim_dtype_shape(cls, original_tensor : torch.Tensor) -> None:
        """
        Assert that the given tensor is valid for semi-structured sparse compression.
        """
        # 检查设备类型是否为 CUDA
        if not original_tensor.is_cuda:
            raise RuntimeError(
                f"Error original_tensor.device= {original_tensor.device} is not supported! "
                "Only CUDA tensors are currently supported."
            )

        # 检查张量维度是否为 2
        if original_tensor.dim() != 2:
            raise RuntimeError(
                f"Error original_tensor.dim = {original_tensor.dim()} is not supported! "
                "Only 2d tensors are currently supported."
            )

        # 检查张量是否是连续的
        if not original_tensor.is_contiguous():
            raise RuntimeError(
                "Error original_tensor is not contiguous!"
                "Only contiguous tensors are currently supported."
            )

        # 检查张量数据类型是否在支持的范围内
        if original_tensor.dtype not in cls._DTYPE_SHAPE_CONSTRAINTS:
            raise RuntimeError(
                f"Error original_tensor.dtype {original_tensor.dtype} is not a supported dtype! "
                "dtype must be one of: {cls._DTYPE_SHAPE_CONSTRAINTS}"
            )

        # 检查张量形状是否满足特定约束
        m, n = original_tensor.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[original_tensor.dtype].sparse_min_cols
        if m < min_rows or m % min_rows or n < min_cols or n % min_cols:
            # TODO 在未来可以添加填充来支持不是完美倍数的稀疏维度
            raise RuntimeError(
                f"Error original_tensor.shape {original_tensor.shape} is not supported! "
                f"Both dimensions must be larger or equal than and a multiple of ({min_rows}, {min_cols})"
            )

    @classmethod
    def _pad_dense_input(cls, dense_input: torch.Tensor) -> torch.Tensor:
        """
        Calculates padding for dense tensor and pads tensor if necessary.
        If padding is not required, this function returns the original tensor.
        """
        # 断言输入张量是二维的
        assert dense_input.dim() == 2

        # 检查张量形状
        m, n = dense_input.shape
        min_rows = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_rows
        min_cols = cls._DTYPE_SHAPE_CONSTRAINTS[dense_input.dtype].dense_min_cols

        # 计算需要填充的量
        to_pad_m = -m % min_rows if m < min_rows or m % min_rows else 0
        to_pad_n = -n % min_cols if n < min_cols or n % min_rows else 0
        if to_pad_m or to_pad_n:
            return torch.nn.functional.pad(dense_input, (0, to_pad_n, 0, to_pad_m))
        else:
            return dense_input

    def to_dense(self):
        col = self.shape[-1]
        # 将稀疏矩阵转换为密集矩阵
        return torch.mm(self, torch.eye(col, dtype=self.dtype, device=self.device))

    @classmethod
    # 定义一个类方法，用于从稠密的 torch.Tensor 转换为 SparseSemiStructuredTensor
    def from_dense(cls, original_tensor: torch.Tensor) -> "SparseSemiStructuredTensor":
        # 抛出未实现错误，表明该方法需要在子类中实现
        raise NotImplementedError

    # 定义一个私有方法 _mm，用于执行矩阵乘法操作
    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        # 抛出未实现错误，表明该方法需要在子类中实现
        raise NotImplementedError
# 导入函数所需的torch库，torch是PyTorch的主要库
import torch

# 导入警告模块，用于发出关于已弃用参数的警告
import warnings

# 引入SparseSemiStructuredTensor类，这是一个torch.Tensor的子类，表示稀疏半结构化张量
from typing import SparseSemiStructuredTensor

# 将密集张量转换为稀疏半结构化张量的函数
def to_sparse_semi_structured(
    original_tensor: torch.Tensor,
    transposed: bool = False,
) -> SparseSemiStructuredTensor:
    """
    This function converts a dense tensor into a sparse semi-structured tensor.
    It will return a SparseSemiStructuredTensor, a subclass of torch.Tensor.

    This function will check to ensure the dense tensor has the right dtype, size, dims, and device.
    We currently only support semi-structured sparse tensors for 2d CUDA tensors.
    Additionally, your tensor must be a positive multiple of the mininum sparse block size, given in
    `_DTYPE_TO_SHAPE_CONSTRAINTS` for each dtype (float32, float16, bfloat16, int8).

    Args:
        original_tensor (Tensor): the dense tensor to convert
        transposed (bool, optional): deprecated arg to be removed in another release. Do not use.
    Returns:
        SparseSemiStructuredTensor: A sparse semi-structured tensor created from the given original_tensor
    Raises:
        None
    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
        >>> A = torch.Tensor([0, 0, 1, 1]).tile((128, 32)).half().cuda()
        tensor([[0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                ...,
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.],
                [0., 0., 1.,  ..., 0., 1., 1.]], device='cuda:0', dtype=torch.float16)
        >>> A_sparse = to_sparse_semi_structured(A)
        SparseSemiStructuredTensor(shape=torch.Size([128, 128]))
        >>> A_sparse.values()
        tensor([[1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                ...,
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.],
                [1., 1., 1.,  ..., 1., 1., 1.]], device='cuda:0', dtype=torch.float16),
        >>> A_sparse.indices()
        tensor([[-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                ...,
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370],
                [-4370, -4370, -4370,  ..., -4370, -4370, -4370]], device='cuda:0', dtype=torch.int16))
    """
    # 如果使用了已弃用的参数transposed，则发出警告
    if transposed:
        warnings.warn(
            "Setting transpose from `to_sparse_semi_structured` is deprecated "
            "and will be removed in a future release. "
            "`SparseSemiStructuredTensor` only support contiguous input tensors.",
            FutureWarning,
            stacklevel=2,
        )

    # 根据_FORCE_CUTLASS标志设置
    # 如果 _FORCE_CUTLASS 为真，则选择 Cutlass 实现的稀疏半结构化张量
    SPARSE_SUBCLASS = (
        torch.sparse.SparseSemiStructuredTensorCUTLASS
        if SparseSemiStructuredTensor._FORCE_CUTLASS
        else torch.sparse.SparseSemiStructuredTensorCUSPARSELT
    )

    # 将原始密集张量转换为稀疏张量，使用选择的子类
    return SPARSE_SUBCLASS.from_dense(original_tensor)
# 实现了基于 CUTLASS 后端的半结构稀疏张量的类，继承自 SparseSemiStructuredTensor。

class SparseSemiStructuredTensorCUTLASS(SparseSemiStructuredTensor):
    """
    This class implements semi-structured sparsity for the CUTLASS backend.

    在 CUTLASS 后端实现了半结构稀疏性。

    In this implementation, the specified elements and metadata are stored seprately,
    in packed and meta respectively.

    在这个实现中，指定的元素和元数据分别存储在 packed 和 meta 中。

    When _FORCE_CUTLASS is set, or when cuSPARSELt is not available, this subclass calls into _sparse_semi_structured_(mm|addmm) and
    sparse_semi_structured_from_dense for conversion to the compressed format.

    当设置了 _FORCE_CUTLASS 或者 cuSPARSELt 不可用时，此子类调用 _sparse_semi_structured_(mm|addmm) 和
    sparse_semi_structured_from_dense 函数，将数据转换为压缩格式。
    """

    BACKEND = "cutlass"
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 128, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 64, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 4, 4),
    }

    @classmethod
    def from_dense(
        cls, original_tensor: torch.Tensor
    ) -> "SparseSemiStructuredTensorCUTLASS":
        # 验证原始张量的设备、维度和数据类型约束
        cls._validate_device_dim_dtype_shape(original_tensor)
        # 调用 sparse_semi_structured_from_dense_cutlass 函数，将原始张量转换为压缩格式的稀疏张量和元数据张量
        (
            sparse_tensor_cutlass,
            meta_tensor_cutlass,
        ) = sparse_semi_structured_from_dense_cutlass(original_tensor)
        # 返回 SparseSemiStructuredTensorCUTLASS 的实例，使用转换后的数据和元数据
        return cls(
            original_tensor.shape,
            packed=sparse_tensor_cutlass,
            meta=meta_tensor_cutlass,
            packed_t=None,
            meta_t=None,
            compressed_swizzled_bitmask=None,
            requires_grad=original_tensor.requires_grad,
        )

    def to_dense(self):
        # 断言稀疏张量的 meta 和 packed 属性不为 None
        assert self.meta is not None and self.packed is not None
        # 如果 meta 的维度为 2，则调用 sparse_semi_structured_to_dense_cutlass 函数将稀疏张量转换为密集张量
        return sparse_semi_structured_to_dense_cutlass(
            self.packed,
            self.meta,
        ) if self.meta.ndim == 2 else super().to_dense()

    @classmethod
    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        # 如果 B 是 SparseSemiStructuredTensor 的实例，则抛出 ValueError
        if isinstance(B, SparseSemiStructuredTensor):
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        # 获取当前类名
        cls_name = self.__class__.__name__
        # 如果当前对象或者 B 的维度不是 2，则抛出 NotImplementedError
        if self.ndim != 2 or B.ndim != 2:
            raise NotImplementedError(
                f"`{cls_name}` matmul: Broadcasting is not implemented"
            )
        # 如果 packed 或者 meta 属性为 None，则抛出 NotImplementedError
        if self.packed is None or self.meta is None:
            raise NotImplementedError(
                f"`{cls_name}` matmul: operation is not supported"
            )
        else:
            # 如果 bias 为 None，则调用 torch._sparse_semi_structured_mm 函数执行稀疏矩阵乘法
            if bias is None:
                res = torch._sparse_semi_structured_mm(
                    self.packed, self.meta, B
                )
            else:
                # 否则调用 torch._sparse_semi_structured_addmm 函数执行带有偏置的稀疏矩阵乘法
                res = torch._sparse_semi_structured_addmm(
                    bias, self.packed, self.meta, B
                )
            # 返回乘法结果，截取前 self.shape[0] 行
            return res[: self.shape[0]]


class SparseSemiStructuredTensorCUSPARSELT(SparseSemiStructuredTensor):
    """
    Placeholder for implementing sparse tensor operations using the CUSPARSELt library.
    """
    """
    cuSPARSELt backend expects the specified elements and the metadata to be stored in a single tensor:
    packed = [ specified elements of original tensor | metadata ]
    For an original tensor of size (m, k) we expect the first m * k // 2 elements to be the kept elements
    The rest of the tensor is metadata. Since there is only one tensor, we only use the packed and packed_t
    attributes respectively.

    cuSPARSELt also supports transposition fusion, which is necessary for performant 2:4 sparse training, as well
    as specifying alg_id, a config that affects the performance of the matmul depending on matmul sizes.
    """

    # 定义后端为 "cusparselt"
    BACKEND = "cusparselt"

    # 不同数据类型的稀疏张量对应的形状约束配置
    _DTYPE_SHAPE_CONSTRAINTS = {
        torch.int8: _SEMI_STRUCTURED_SPARSE_CONFIG(32, 32, 16, 16),
        torch.float16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.bfloat16: _SEMI_STRUCTURED_SPARSE_CONFIG(16, 16, 8, 8),
        torch.float32: _SEMI_STRUCTURED_SPARSE_CONFIG(8, 8, 4, 4),
    }

    @classmethod
    def from_dense(cls, original_tensor: torch.Tensor) -> "SparseSemiStructuredTensorCUSPARSELT":
        # 验证原始张量的设备、维度和数据类型是否符合要求
        cls._validate_device_dim_dtype_shape(original_tensor)
        # 返回一个 SparseSemiStructuredTensorCUSPARSELT 类的实例，使用原始张量的相关属性初始化
        return cls(
            shape=original_tensor.shape,  # 设置稀疏张量的形状
            packed=torch._cslt_compress(original_tensor),  # 压缩原始张量为稀疏形式，存储在 packed 属性中
            meta=None,  # 元数据设置为 None
            packed_t=None,  # 稀疏张量转置后的数据，初始设置为 None
            meta_t=None,  # 转置后的元数据，初始设置为 None
            compressed_swizzled_bitmask=None,  # 压缩后的位掩码，初始设置为 None
            fuse_transpose_cusparselt=SparseSemiStructuredTensor._FUSE_TRANSPOSE,  # 是否融合转置操作
            alg_id_cusparselt=SparseSemiStructuredTensor._DEFAULT_ALG_ID,  # 指定的算法 ID
            requires_grad=original_tensor.requires_grad,  # 是否需要梯度信息
        )

    @classmethod
    def prune_dense_static_sort(cls, original_tensor : torch.Tensor, algorithm="") -> "SparseSemiStructuredTensor":
        """
        This function prunes a dense tensor to a sparse semi-structured tensor using the specified algorithm.

        The equivalent PyTorch code to create the same outputs from the dense tensor is provided below.

        Args:
        - cls: Class reference for SparseSemiStructuredTensor.
        - original_tensor: Input dense tensor to be pruned.
        - algorithm: Optional algorithm parameter for pruning.

        Returns:
        - SparseSemiStructuredTensor: A sparse semi-structured tensor object.

        The following code performs the equivalent operations in PyTorch:
        ```
        from torch.sparse import SparseSemiStructuredTensorCUSPARSELT
        from torch.sparse._semi_structured_conversions import _sparse_semi_structured_tile, _compute_compressed_swizzled_bitmask

        pruned = _sparse_semi_structured_tile(original_tensor)
        packed_cusparselt = torch._cslt_compress(pruned)
        packed_t_cusparselt = torch._cslt_compress(pruned.t().contiguous())
        bitmask = _compute_compressed_swizzled_bitmask(pruned)

        SparseSemiStructuredTensorCUSPARSELT(original_tensor.shape, packed_cutlass, None, packed_t_cutlass, None, bitmask)
        ```
        """
        # Call internal function to tile and prune the original dense tensor
        (packed, meta, packed_t, meta_t, compressed_swizzled_bitmask) = torch._sparse_semi_structured_tile(
            original_tensor,
            algorithm=algorithm,
            use_cutlass=False)

        # Return an instance of SparseSemiStructuredTensor using the computed results
        return cls(
            original_tensor.shape,
            packed=packed,
            meta=meta,
            packed_t=packed_t,
            meta_t=meta_t,
            compressed_swizzled_bitmask=compressed_swizzled_bitmask,
            requires_grad=False,
        )

    def _mm(
        self,
        B: torch.Tensor,
        *,
        bias: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        if isinstance(B, SparseSemiStructuredTensor):
            # 如果输入的 B 是 SparseSemiStructuredTensor 类型，则抛出错误，因为硬件不支持这种类型的乘法运算
            raise ValueError(
                "`SparseSemiStructuredTensor @ SparseSemiStructuredTensor` is not supported by the hardware"
            )
        if self.ndim != 2 or B.ndim != 2:
            # 如果 self 或者 B 的维度不是 2，则抛出错误，因为当前版本未实现广播功能
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: Broadcasting is not implemented"
            )
        if B.dtype != self.dtype:
            # 如果 B 的数据类型与 self 的数据类型不同，则抛出错误，因为只有当 A 和 B 具有相同的数据类型时才支持此操作
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)}`, "
                f"with A.dtype={self.dtype} and B.dtype={B.dtype}. "
                "This operation is only supported when A and B have the same data type."
            )
        if bias is not None and bias.dtype != self.dtype:
            # 如果 bias 不为 None，并且其数据类型与 self 的数据类型不同，则抛出错误，
            # 因为只有当 A、B 和 C 具有相同的数据类型时才支持此操作
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: trying to do `A={tuple(self.shape)} @ B={tuple(B.shape)} + C`, "
                "with A.dtype=B.dtype={self.dtype} and C.dtype={B.dtype}. "
                "This operation is only supported when A, B and C have the same data type."
            )
        if self.packed is None:
            # 如果 self.packed 为 None，则抛出错误，因为当前版本不支持该操作
            raise NotImplementedError(
                f"`{self.__class__.__name__}` matmul: operation is not supported"
            )
        else:
            # 调用底层的稀疏矩阵乘法函数 torch._cslt_sparse_mm 进行矩阵乘法运算
            res = torch._cslt_sparse_mm(
                self.packed,
                B,
                bias=bias,
                transpose_result=self.fuse_transpose_cusparselt,
                alg_id=self.alg_id_cusparselt,
            )
            # 如果设置了 transpose_result=self.fuse_transpose_cusparselt，则对结果进行转置
            return res.t() if self.fuse_transpose_cusparselt else res
```