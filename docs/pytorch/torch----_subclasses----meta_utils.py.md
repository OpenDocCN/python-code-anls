# `.\pytorch\torch\_subclasses\meta_utils.py`

```
# mypy: allow-untyped-defs
# 启用 mypy，允许未类型化的函数定义

from __future__ import annotations
# 在 Python 3.7 及更早版本中，使用 __future__ 导入标注的类型提示支持

import contextlib
# 导入 contextlib 模块，用于支持上下文管理器

import dataclasses
# 导入 dataclasses 模块，用于数据类的支持

import warnings
# 导入 warnings 模块，用于警告处理

import weakref
# 导入 weakref 模块，用于弱引用支持

from dataclasses import dataclass
# 从 dataclasses 中导入 dataclass 装饰器，用于声明数据类

from typing import (
    Any,
    Callable,
    ClassVar,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    Union,
)
# 导入多种类型提示，用于静态类型检查

from typing_extensions import TypeAlias
# 从 typing_extensions 导入 TypeAlias，用于类型别名的支持

import torch
# 导入 PyTorch 库

from torch._C._autograd import CreationMeta
# 从 torch._C._autograd 导入 CreationMeta，用于自动求导元数据

from torch._C._functorch import (
    _add_batch_dim,
    _unwrap_functional_tensor,
    _wrap_functional_tensor,
    get_unwrapped,
    is_batchedtensor,
    is_functorch_wrapped_tensor,
    is_gradtrackingtensor,
    is_legacy_batchedtensor,
    maybe_get_bdim,
    maybe_get_level,
    peek_interpreter_stack,
)
# 从 torch._C._functorch 导入多个函数和类，用于 Functorch 功能

from torch._logging import trace_structured
# 从 torch._logging 导入 trace_structured，用于结构化日志追踪

from torch.utils._mode_utils import no_dispatch
# 从 torch.utils._mode_utils 导入 no_dispatch，用于模式管理

from torch.utils._python_dispatch import is_traceable_wrapper_subclass
# 从 torch.utils._python_dispatch 导入 is_traceable_wrapper_subclass，用于判断是否为可追踪包装器子类

from torch.utils.weak import WeakIdKeyDictionary
# 从 torch.utils.weak 导入 WeakIdKeyDictionary，用于弱引用字典支持

if TYPE_CHECKING:
    from torch._C._functorch import CInterpreter
    from torch._guards import Source
    # 在类型检查中导入，避免循环导入

    from torch._subclasses.fake_tensor import FakeTensorMode
    # 在类型检查中导入，用于虚假张量模式

    from torch.fx.experimental.symbolic_shapes import ShapeEnv, SymbolicContext
    # 在类型检查中导入，用于符号形状实验特性的支持

DimList = List
# 定义 DimList 类型别名为 List

def safe_is_leaf(t):
    try:
        return t.is_leaf
    except RuntimeError:
        # 推理模式可能触发此异常
        return False
    # 返回张量 t 是否为叶子节点的安全检查函数

def safe_grad(t):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "The .grad attribute of a Tensor")
        return t.grad
    # 获取张量 t 的梯度属性的安全访问函数

def assert_eq(a, b):
    assert a == b, f"{a} != {b}"
    # 断言函数，判断 a 和 b 是否相等，若不相等则抛出异常

def assert_metadata_eq(
    assert_eq,
    m1: Union[MetaTensorDesc, torch.Tensor],
    m2: torch.Tensor,
    *,
    skip_symbolic=False,
    skip_leaf=False,
):
    if isinstance(m1, torch.Tensor):
        m1 = MetaTensorDescriber().describe_tensor(m1)
    # 如果 m1 是 torch.Tensor 类型，则通过 MetaTensorDescriber 描述器描述张量 m1 的元数据
    # 定义一个名为 go 的函数，用于比较两个输入的 MetaTensor 对象 m1 和 m2
    def go(m1, m2):
        # 断言 m1 和 m2 的数据类型相同
        assert_eq(m1.dtype, m2.dtype)
        # 如果不跳过符号检查，断言 m1 和 m2 的形状相同
        if not skip_symbolic:
            assert_eq(m1.shape, m2.shape)
        # 断言 m1 和 m2 是否需要梯度计算的属性相同
        assert_eq(m1.requires_grad, m2.requires_grad)
        # 如果不跳过叶子节点检查，断言 m1 和 m2 是否为叶子节点
        if not skip_leaf:
            assert_eq(m1.is_leaf, m2.is_leaf)
        # MetaTensorDesc 不存储 grad_fn；从叶子节点推断
        # 断言 m1 和 m2 的 grad_fn 是否为 None
        # assert_eq(m1.grad_fn is None, m2.grad_fn is None)
        assert_eq(m1.is_sparse, m2.is_sparse)
        # 断言 m1 和 m2 是否处于推断状态
        assert_eq(m1.is_inference, m2.is_inference())
        # 断言 m1 和 m2 是否为共轭
        assert_eq(m1.is_conj, m2.is_conj())
        # 断言 m1 和 m2 是否为负数
        assert_eq(m1.is_neg, m2.is_neg())
        # 断言 m1 的梯度是否不为 None，并且 m2 的 safe_grad 不为 None
        assert_eq(m1.grad is not None, safe_grad(m2) is not None)
        # 如果 m1 的梯度不为 None，则递归调用 go 函数比较 m1.grad 和 m2 的 safe_grad(m2)
        if m1.grad is not None:
            go(m1.grad, safe_grad(m2))
        # 如果 m1 是稀疏张量
        if m1.is_sparse:
            # 断言 m1 的稠密维度等于 m2 的稠密维度
            assert_eq(m1.dense_dim, m2.dense_dim())
            # 断言 m1 的稀疏维度等于 m2 的稀疏维度
            assert_eq(m1.sparse_dim, m2.sparse_dim())
            # 断言 m1 是否已经聚合
            assert_eq(m1.is_coalesced, m2.is_coalesced())
        else:
            # 如果不跳过符号检查，断言 m1 的步幅等于 m2 的步幅
            if not skip_symbolic:
                assert_eq(m1.stride, m2.stride())
                # 断言 m1 的存储偏移量等于 m2 的存储偏移量
                assert_eq(m1.storage_offset, m2.storage_offset())
            # 断言 m1 是否为视图
            assert_eq(m1.is_view, m2._is_view())
            # 如果 m1 是视图，则递归调用 go 函数比较 m1.base 和 m2 的 m2._base
            if m1.is_view:
                go(m1.base, m2._base)
        # TODO: 测试是否可调整大小（当前无直接查询）
        # TODO: 审计 AutogradMeta 看是否匹配
        # TODO: 测试前向自动微分（AD）

    # 返回调用 go 函数的结果，比较 m1 和 m2
    return go(m1, m2)
# 检查给定对象是否为稀疏的 COO（坐标格式）张量
def is_sparse_coo(t):
    return isinstance(t, torch.Tensor) and t.layout is torch.sparse_coo


# 检查给定布局是否为压缩稀疏格式之一
def is_sparse_compressed_layout(layout):
    return layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }


# 检查给定对象是否为稀疏压缩格式的张量
def is_sparse_compressed(t):
    return isinstance(t, torch.Tensor) and is_sparse_compressed_layout(t.layout)


# 检查给定对象是否为任意稀疏格式的张量
def is_sparse_any(t):
    return is_sparse_coo(t) or is_sparse_compressed(t)


# 类型别名，用于表示元数据中的存储 ID 和张量 ID
MetaStorageId: TypeAlias = int
MetaTensorId: TypeAlias = int


# 全局变量，用于跟踪描述器的下一个 ID
DESCRIBER_NEXT_ID = 0


class MetaTensorDescriber:
    """
    给定一个张量/存储，生成对应的 MetaTensorDesc/MetaStorageDesc，
    包含足够的信息以尽可能忠实地重构一个元张量/伪张量。
    
    这是一个有状态的转换对象，因为我们跟踪了传递给我们的张量/存储的 ID，
    这样我们可以在看到相同的张量/存储时始终给出相同的 ID。
    """

    def __init__(self, *, copy_data=False):
        global DESCRIBER_NEXT_ID
        self.id = DESCRIBER_NEXT_ID
        DESCRIBER_NEXT_ID += 1
        self.next_tensor_id: MetaTensorId = 0
        self.next_storage_id: MetaStorageId = 0
        # 弱引用字典，将张量映射到整数 ID
        self.lookup_tensor = WeakIdKeyDictionary()
        # 弱引用字典，将存储映射到整数 ID
        self.lookup_storage = WeakIdKeyDictionary()
        self.copy_data = copy_data
        self.traced_tensors = set()
        self.traced_storages = set()

    def get_tensor_id(self, t: torch.Tensor):
        # 如果张量不在查找表中，将其映射到下一个张量 ID
        if t not in self.lookup_tensor:
            self.lookup_tensor[t] = self.next_tensor_id
            self.next_tensor_id += 1
        return self.lookup_tensor[t]

    def get_storage_id(self, s: torch.UntypedStorage):
        # 如果存储不在查找表中，将其映射到下一个存储 ID
        if s not in self.lookup_storage:
            self.lookup_storage[s] = self.next_storage_id
            self.next_storage_id += 1
        return self.lookup_storage[s]

    def describe_storage(self, s: torch.UntypedStorage, *, trace: bool = False):
        # 创建一个 MetaStorageDesc 对象，描述给定存储的 ID、大小和数据（如果需要复制）
        r = MetaStorageDesc(
            id=self.get_storage_id(s),
            size=s.size(),
            # 注意：我们还未复制数据；复制将在创建新存储时进行
            data=s if self.copy_data else None,
        )
        # 如果需要追踪描述操作，并且 ID 尚未在追踪集合中，则追踪此操作
        if trace and r.id not in self.traced_storages:
            trace_structured(
                "describe_storage",
                metadata_fn=lambda: r.as_json(self.id),
            )
            self.traced_storages.add(r.id)
        return r

    def describe_tensor(
        self, t: torch.Tensor, *, recurse: bool = True, trace: bool = False
    ):
        # 此方法未完整提供，应继续阅读后续代码以获得完整的注释信息
    # 将对象的属性转换为 JSON 格式的字典
    def as_json(self, describer_id):
        # 返回包含对象 id、描述者 id 和大小的字典
        return {
            "id": self.id,  # 对象的唯一标识符
            "describer_id": describer_id,  # 描述者的标识符
            "size": self.size if isinstance(self.size, int) else repr(self.size),  # 如果大小是整数则直接使用，否则转换为字符串表示
        }
@dataclass(frozen=True)
class MetaTensorDesc:
    id: MetaTensorId
    ndim: int
    dtype: torch.dtype
    device: torch.device

    # NB: Sometimes, size, stride and storage_offset contain SymInt, in which
    # case this is NOT serializable.  That only happens when you're
    # re-fakeifying a fake tensor with an existing ShapeEnv... maybe we
    # can get rid of this use case entirely.  Notably, even if we are
    # fakeifying a real tensor into a fake tensor with symbolic shapes, the
    # size here is NOT dynamic
    # NB: These also contain SymInt because wrap_meta_outputs_with_default_device_logic
    # goes through this codepath.  But it really should not LOL.
    # NB: size could potentially be None as you can override it and make it
    # throw an error, but we don't currently have any subclasses that do this
    # except C++ nested tensor but we're going to have nested int to make this
    # defined on NJT
    size: Tuple[int, ...]  # Tuple storing the dimensions of the tensor
    dynamo_dynamic_indices: List[int]  # List of dynamic indices

    layout: torch.layout = torch.strided  # Default layout type is strided
    is_inference: bool = False  # Default to False, indicating not in inference mode
    is_leaf: bool = False  # Default to False, indicating not a leaf tensor
    requires_grad: bool = False  # Default to False, indicating no gradient computation needed
    is_sparse: bool = False  # Default to False, indicating not a sparse tensor
    is_mkldnn: bool = False  # Default to False, indicating not an MKL-DNN tensor
    is_functorch_wrapped: bool = False  # Default to False, indicating not wrapped by functorch
    is_batchedtensor: bool = False  # Default to False, indicating not a batched tensor
    is_legacy_batchedtensor: bool = False  # Default to False, indicating not a legacy batched tensor
    is_gradtrackingtensor: bool = False  # Default to False, indicating not a gradient tracking tensor
    is_view: bool = False  # Default to False, indicating not a view tensor
    is_nested: bool = False  # Default to False, indicating not a nested tensor
    is_traceable_wrapper_subclass: bool = False  # Default to False, indicating not a traceable wrapper subclass
    is_functional: bool = False  # Default to False, indicating not a functional tensor
    is_conj: bool = False  # Default to False, indicating not a conjugate tensor
    is_neg: bool = False  # Default to False, indicating not a negative tensor
    is_parameter: bool = False  # Default to False, indicating not a parameter tensor
    stride: Optional[Tuple[int, ...]] = None  # Optional tuple storing stride information
    storage_offset: int = 0  # Default storage offset is 0

    # NB: We have a choice whether or not to store the id or a direct pointer
    # to the data structure.  For ease of use, we store the data structure,
    # but this means that when we serialize, we have to swizzle these pointers
    # back into ids (so we have accurate aliasing relationships)
    storage: Optional[MetaStorageDesc] = None  # Optional storage descriptor

    sparse_dim: Optional[int] = None  # Optional sparse dimension count (for sparse tensors)
    dense_dim: Optional[int] = None  # Optional dense dimension count (for sparse tensors)
    is_coalesced: Optional[bool] = None  # Optional coalesced flag (for sparse tensors)
    crow_indices: Optional[MetaTensorDesc] = None  # Optional crow indices (for sparse compressed tensors)
    col_indices: Optional[MetaTensorDesc] = None  # Optional column indices (for sparse compressed tensors)
    ccol_indices: Optional[MetaTensorDesc] = None  # Optional compressed column indices (for sparse compressed tensors)
    row_indices: Optional[MetaTensorDesc] = None  # Optional row indices (for sparse compressed tensors)
    values: Optional[MetaTensorDesc] = None  # Optional values (for sparse compressed tensors)
    unwrapped: Optional[MetaTensorDesc] = None  # Optional unwrapped tensor descriptor (for functorch wrapped tensors)
    bdim: Optional[int] = None  # Optional batch dimension (for functorch wrapped tensors)
    base: Optional[MetaTensorDesc] = None  # Optional base tensor descriptor (for view tensors)
    attrs: Optional[Dict[str, MetaTensorDesc]] = None  # Optional attributes dictionary (for traceable wrapper subclasses)
    creation_meta: Optional[CreationMeta] = None  # Optional creation meta information
    grad: Optional[MetaTensorDesc] = None  # Optional gradient descriptor

    # Everything below is NOT serializable, need some more work
    # ClassVar annotation to define a class-level variable _UNSERIALIZABLE which is a list of strings representing
    # attributes that are not serializable.
    _UNSERIALIZABLE: ClassVar[List[str]] = [
        "ctx",                      # Context object associated with tracing
        "type",                     # Type information associated with tracing
        "fake_mode",                # Mode of FakeTensorMode
        "view_func",                # Callable function for viewing tensor
        "level",                    # Indicates functorch wrapping level
        "current_level",            # Current level in the context
        "functorch_stack",          # Stack of CInterpreter objects for functorch
        "autograd_meta_from",       # Tensor from which autograd meta information is derived
    ]
    
    ctx: Optional[object] = None   # Context object associated with tracing (is_traceable_wrapper_subclass)
    type: Optional[Type] = None    # Type information associated with tracing (is_traceable_wrapper_subclass)
    fake_mode: Optional[FakeTensorMode] = None  # Mode of FakeTensorMode
    view_func: Optional[Callable[[torch.Tensor, Callable[[int], int], Callable[[torch.Tensor], torch.Tensor]], torch.Tensor]] = None
        # Callable function for viewing tensor with specific signatures
    
    # level looks serializable, but actually it is meaningless without
    # the functorch_stack below
    level: Optional[int] = None   # Indicates functorch wrapping level (is_functorch_wrapped)
    current_level: Optional[int] = None  # Current level in the context
    
    functorch_stack: Optional[List[CInterpreter]] = None
        # Stack of CInterpreter objects used by functorch
    
    autograd_meta_from: Optional[torch.Tensor] = None
        # Tensor from which autograd meta information is derived
    
    # This is only populated on copy_data, and typically is not used at all,
    # except for some of our meta-ification paths that don't properly use
    # storage (pro-tip: you should use storage)
    data: Optional[torch.Tensor] = None
        # Optional tensor data, usually not used and considered for meta-ification
    
    # Faithfully serializing functorch tensors will not be too difficult.
    # We only need to consider grad/vmap interpreters, and their internal
    # state is only bools (mostly what the grad enabled/disabled state
    # should be in the lower layer).  Beyond that, tensors just need to
    # precisely indicate which particular interpreter they correspond
    # to (we then replace level with a pointer to the interpreter stack.)
    # However, this use of functorch is very "non-lexical" so it's not
    # entirely clear how to make it all lexical again, so we haven't done
    # it for now.
    
    # NB: This will reference numeric IDs, and it is assumed that you've
    # already serialized everything this recursively references
    # 定义一个内部函数 json，用于将对象转换为 JSON 可序列化的形式
    def as_json(self, describer_id):
        def json(k, v):
            # 对于一些无法序列化的字段，进行最佳努力的调试序列化处理
            # （可以根据需要添加其他特殊情况）
            if k in ["data", "autograd_meta_from"]:
                return None  # 这些字段不进行 repr 处理
            if k in set(MetaTensorDesc._UNSERIALIZABLE):
                return repr(v)  # 使用 repr 将无法序列化的字段转换为字符串
            if isinstance(v, (torch.device, torch.dtype, torch.layout)):
                return repr(v)  # 将特定的 torch 类型对象转换为字符串
            if isinstance(v, torch.SymInt):
                return repr(v)  # 将 torch 的 SymInt 类型对象转换为字符串
            if isinstance(v, (tuple, list)):
                return [json(k, v1) for v1 in v]  # 递归处理元组和列表中的每个元素
            if isinstance(v, (MetaStorageDesc, MetaTensorDesc)):
                return v.id  # 对于 MetaStorageDesc 和 MetaTensorDesc 对象，返回其 id 属性
            if isinstance(v, CreationMeta):
                return str(v)  # 对于 CreationMeta 对象，返回其字符串表示
            if k == "attrs" and isinstance(v, dict):
                return {k1: v1.id for k1, v1 in v.items()}  # 将 attrs 字典中的值转换为 id 形式
            return v  # 默认情况下直接返回值 v

        # 构建 JSON 对象
        r = {
            field.name: json(field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)  # 遍历数据类的每个字段
            if not (
                getattr(self, field.name) is field.default
                or (
                    field.name == "dynamo_dynamic_indices"
                    and not getattr(self, field.name)
                )
            )
        }
        r.update({"describer_id": describer_id})  # 添加 describer_id 到 JSON 对象中
        return r  # 返回构建的 JSON 对象

    @property
    def shape(self):
        return self.size  # 返回对象的 size 属性作为 shape 属性的值
# 这个函数用于在安全模式下复制张量数据到目标张量，避免对假张量进行复制操作。
def _safe_copy(dst, src):
    # 如果 src 不是 torch.Tensor 类型，则直接返回，不进行复制操作
    if type(src) is not torch.Tensor:
        return
    # 使用 dst 的 copy_ 方法复制 src 的数据到 dst 中
    dst.copy_(src)


# 这个函数用于在安全模式下克隆张量，避免对假张量进行克隆操作。
def _safe_clone(src):
    # 如果 src 不是 torch.Tensor 类型，则返回 None
    if type(src) is not torch.Tensor:
        return None
    # 返回 src 的克隆张量
    return src.clone()


# 这是一个用于将多个张量转换为共享相同视图/存储结构的元张量的类。
# 操作模型是：分配一个 MetaConverter 实例，并重复调用它来转换所有需要的张量。
# 使用相同的对象来处理共享存储的张量非常重要，因为这决定了如何将共享存储对应到相同的元存储。
# 这个类会持有对缓存张量和张量存储的弱引用。
class MetaConverter:
    def __init__(self, *, copy_data: bool = False):
        # 将 MetaStorageId 映射到 UntypedStorage 的弱引用字典
        self.storage_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        # 将 MetaTensorId 映射到 torch.Tensor 的弱引用字典（通常是元张量或 FakeTensor）
        self.tensor_memo: weakref.WeakValueDictionary = weakref.WeakValueDictionary()
        self.hit = 0  # 命中缓存的次数
        self.miss = 0  # 未命中缓存的次数
        self.del_hook = None  # 删除钩子函数
        self.arg_cnt = 0  # 参数计数
        # 确保在生成的元存储/张量中填充 real_storage/real_tensor 属性。
        # 这个属性的命名具有重要意义：FakeTensor 依赖于 real_tensor 正好被设置为这个值。
        self.copy_data = copy_data  # 控制是否复制数据
        self.describer = MetaTensorDescriber(copy_data=copy_data)  # 元张量描述器的实例化

    # 返回是否至少有一次命中缓存且未发生未命中的情况
    def successful(self):
        return self.hit > 0 and self.miss == 0

    # 根据 MetaTensorDesc 获取张量的缓存记录
    def get_tensor_memo(self, t: MetaTensorDesc):
        return self.tensor_memo.get(t.id, None)

    # 设置 MetaTensorDesc 对应的张量缓存记录
    def set_tensor_memo(self, t: MetaTensorDesc, v):
        self.tensor_memo[t.id] = v

    # 根据 MetaStorageDesc 获取存储的缓存记录
    def get_storage_memo(self, s: MetaStorageDesc):
        return self.storage_memo.get(s.id, None)

    # 设置 MetaStorageDesc 对应的存储缓存记录
    def set_storage_memo(self, s: MetaStorageDesc, v):
        self.storage_memo[s.id] = v
    # 处理元数据存储的函数，接受元数据存储描述和回调函数作为参数
    def meta_storage(self, s: MetaStorageDesc, callback):
        # 如果要伪造一个具有秘密零大小存储的张量，
        # 需要确保调整元数据存储的大小。
        if self.get_storage_memo(s) is None:
            # 使用回调函数创建一个空的元数据存储张量
            r_s = callback(
                lambda: torch.empty(s.size, dtype=torch.uint8, device="meta"),
            ).untyped_storage()
            # 如果需要复制数据
            if self.copy_data:
                # 注意：需要使用 no_dispatch，因为内部的存储复制是作为张量操作实现的
                with torch.no_grad(), no_dispatch():
                    assert s.data is not None
                    # 将原始数据复制到新创建的存储中
                    r_s.real_storage = s.data.clone()
            # 将处理后的存储信息保存到内存中
            self.set_storage_memo(s, r_s)
            return r_s
        else:
            # 如果存储信息已经存在于内存中，则直接返回保存的存储信息
            return self.get_storage_memo(s)

    # 该函数假设可以执行数据转换
    # 注意：这里的 name 在 Dynamo 中被传统方式使用；
    # 它对应于我们伪造的张量的 Source.name()，并且对应于有效的 Python 表达式。
    # 在构造子名称作为这个过程的一部分时，我们将保持此不变性！
    # （尽管这个属性对于这个的其他用户可能不需要保持不变。）
    def meta_tensor(
        self,
        t: MetaTensorDesc,
        shape_env: Optional[ShapeEnv] = None,
        callback=lambda t: t(),
        source: Optional[Source] = None,
        symbolic_context: Optional[SymbolicContext] = None,
    ):
        # 函数定义部分，处理元数据张量相关逻辑
        pass

    # 对象的调用方法，用于处理张量数据的元数据信息
    def __call__(
        self,
        t,
        shape_env=None,
        *,
        callback=lambda t: t(),
        source=None,
        symbolic_context=None,
        # 控制是否在源不为空时将张量元数据转储到结构化日志中。
        # 因为在 Dynamo 完成后我们进行再伪造时，不希望从 AOTAutograd 再次转储信息，这是多余的。
        trace=True,
    ):
        # 函数定义部分，处理对象调用时的各种参数和控制逻辑
        pass
        # TODO: zero tensors?  We appear to have eliminated them by
        # excluding complex for now

        # Filter out cases we don't support
        # TODO: This can probably be simplified quite a bit
        if isinstance(t, torch.Tensor) or is_traceable_wrapper_subclass(t):
            # Check if the tensor is on a device type that is "lazy",
            # which is not supported, or if it is quantized, which is also not supported,
            # or if it's a view out of a sparse tensor (plain sparse is supported).
            if (
                t.device.type == "lazy"
                or
                t.is_quantized
                or
                (t._is_view() and t._base is not None and t._base.is_sparse)
            ):
                # Increase miss count and return NotImplemented for unsupported cases
                self.miss += 1
                return NotImplemented
            else:
                # Increase hit count for supported cases
                self.hit += 1
        elif torch.overrides.is_tensor_like(t):
            # Increase miss count and return NotImplemented for tensor-like objects
            self.miss += 1
            return NotImplemented
        else:
            # non-Tensor types don't count as hit or miss, return the object directly
            return t

        if source is None:
            # If source is not provided, set trace to False
            trace = False

        # Describe the tensor using self.describer.describe_tensor method,
        # considering trace flag if it is True
        t_desc = self.describer.describe_tensor(t, trace=trace)

        if trace:
            # If trace is True, trace the description of the source tensor
            trace_structured(
                "describe_source",
                metadata_fn=lambda: {
                    "describer_id": self.describer.id,
                    "id": t_desc.id,
                    "source": source.name(),
                },
            )

        # Perform meta-fication of the tensor t with specified contexts and settings
        # Here, contextlib.ExitStack is used to manage multiple contexts
        with contextlib.ExitStack() as exit_stack:
            exit_stack.enter_context(torch._dispatch.python.suspend_functionalization())
            st = peek_interpreter_stack()
            if st is not None:
                exit_stack.enter_context(
                    torch._functorch.pyfunctorch.temporarily_clear_interpreter_stack()
                )

            # Call self.meta_tensor with specified arguments to get the result r
            r = self.meta_tensor(
                t_desc,
                shape_env=shape_env,
                callback=callback,
                source=source,
                symbolic_context=symbolic_context,
            )

        if type(t) is torch.nn.Parameter:
            # If t is a torch.nn.Parameter, mark r as a parameter tensor
            r._is_param = True

        # TODO: return the description for later use
        return r
# 导入名为 torch._prims_common 的模块 utils
import torch._prims_common as utils
```