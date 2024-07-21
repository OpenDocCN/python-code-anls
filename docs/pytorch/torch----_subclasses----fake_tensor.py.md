# `.\pytorch\torch\_subclasses\fake_tensor.py`

```py
# mypy: allow-untyped-defs
# 引入需要的模块和库
import contextlib                      # 上下文管理模块
import functools                       # 函数工具模块
import logging                         # 日志记录模块
import os                              # 操作系统功能模块
import traceback                       # 追踪异常模块
import weakref                         # 弱引用模块
from collections import defaultdict    # 默认字典模块
from dataclasses import dataclass      # 数据类装饰器
from typing import (                   # 类型提示模块，包括各种类型注解
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TYPE_CHECKING,
    TypeVar,
    Union,
)
from weakref import ReferenceType     # 引入弱引用类型

import torch                          # PyTorch核心库
import torch._custom_op               # PyTorch自定义操作
import torch._logging                 # PyTorch日志模块
from torch._C._functorch import is_functorch_wrapped_tensor, is_legacy_batchedtensor  # 引入函数检查Functorch包装的张量和旧批量张量
from torch._guards import Source      # 引入Source守卫
from torch._ops import OpOverload     # 引入操作重载
from torch._prims_common import suggest_memory_format  # 推荐内存格式
from torch._subclasses.meta_utils import (  # 元类工具函数引入
    assert_eq,
    assert_metadata_eq,
    is_sparse_any,
    is_sparse_compressed,
    MetaConverter,
)
from torch._utils import render_call  # 引入调用渲染工具函数
from torch.fx.operator_schemas import normalize_function  # 标准化函数
from torch.multiprocessing.reductions import StorageWeakRef  # 存储弱引用
from torch.overrides import TorchFunctionMode  # Torch函数模式
from torch.utils._mode_utils import no_dispatch  # 无分派工具
from torch.utils._python_dispatch import (  # Python分派工具
    is_traceable_wrapper_subclass,
    TorchDispatchMode,
)
from torch.utils._pytree import PyTree, tree_map, tree_map_  # PyTree相关工具
from torch.utils._stats import count  # 统计工具
from torch.utils._traceback import CapturedTraceback  # 捕获追踪

if TYPE_CHECKING:
    from torch.fx.experimental.symbolic_shapes import ShapeEnv  # 符号形状环境类型引入
    from torch.types import _bool  # 布尔类型引入


class _Unassigned:  # 定义未分配类
    pass


def _is_plain_tensor(t):  # 判断是否为普通张量函数定义
    return (
        type(t) is torch.Tensor  # 张量类型判断
        and t.layout == torch.strided  # 张量布局判断为strided
        and not (  # 不满足以下条件
            t.is_sparse  # 稀疏张量判断
            or t.is_nested  # 嵌套张量判断
            or is_functorch_wrapped_tensor(t)  # Functorch包装张量判断
            or is_legacy_batchedtensor(t)  # 旧批量张量判断
            or torch._is_functional_tensor(t)  # 功能张量判断
        )
    )


_UNASSIGNED = _Unassigned()  # 创建未分配实例

DimList = List  # 维度列表定义

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器实例

# TODO: Hack to unblock https://github.com/pytorch/pytorch/pull/108186
# Proper fix tracked by https://github.com/pytorch/pytorch/issues/120105
try:
    not_implemented_log = torch._logging.getArtifactLogger(__name__, "not_implemented")  # 获取'not_implemented'的日志记录器
except ValueError as e:
    if "'not_implemented' not registered" in str(e):
        import logging as not_implemented_log  # 若'not_implemented'未注册则使用标准日志模块
    else:
        raise e

pytree = torch.utils._pytree  # PyTree工具模块别名定义
T = TypeVar("T")  # 类型变量定义
TensorWeakRef = Any  # 张量弱引用定义

aten = torch._ops.ops.aten  # ATen操作定义

CONSTANT_NUMEL_LIMIT = 1  # 常量尺寸限制定义

RECURSION_COUNT = 0  # 递归计数初始化为0


# Small helper that increments recursion count, and
# resets it when the object goes out of scope.  Useful
# if you don't want to increase indentation which is
# what a context manager would do.
class IncrementRecursionCount:  # 递增递归计数的帮助类定义
    def __init__(self):
        global RECURSION_COUNT
        RECURSION_COUNT += 1  # 递归计数增加

    def __del__(self):
        global RECURSION_COUNT
        RECURSION_COUNT -= 1  # 对象销毁时递归计数减少


@dataclass
class UnsupportedFakeTensorException(RuntimeError):  # 不支持的伪张量异常定义
    reason: str  # 原因属性


@dataclass
class DynamicOutputShapeException(RuntimeError):  # 动态输出形状异常定义
    func: OpOverload  # 操作重载属性
# 自定义异常类，用于表示当数据依赖性输出时引发的异常
class DataDependentOutputException(RuntimeError):
    func: OpOverload  # 异常类包含一个名为 func 的属性，类型为 OpOverload


# 使用 @dataclass 装饰器，定义一个自定义异常类 UnsupportedOperatorException
@dataclass
class UnsupportedOperatorException(RuntimeError):
    func: OpOverload  # 异常类包含一个名为 func 的属性，类型为 OpOverload


# 定义函数 ordered_set，接收多个参数并返回一个字典，其中参数作为字典的键，值为 True
def ordered_set(*items):
    return dict.fromkeys(items, True)


# 使用 @contextlib.contextmanager 装饰器定义上下文管理器函数 unset_fake_temporarily
def unset_fake_temporarily():
    # 调用 torch._C._unset_dispatch_mode 方法，临时取消 FAKE 模式并保存旧的模式
    old = torch._C._unset_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE)
    try:
        yield old  # 返回旧的模式
    finally:
        if old is not None:
            torch._C._set_dispatch_mode(old)  # 恢复旧的模式


# 定义函数 get_plain_tensors，接收一个参数 subclass，确保其为可追踪包装器的子类
def get_plain_tensors(subclass):
    assert is_traceable_wrapper_subclass(subclass)
    plain_tensors = []  # 初始化空列表，用于存储普通张量
    todo = [subclass]  # 将 subclass 放入待处理列表
    while todo:
        curr = todo.pop()  # 从待处理列表中取出一个元素
        inner_keys, _ = curr.__tensor_flatten__()  # 调用 curr 对象的 __tensor_flatten__ 方法
        for key in inner_keys:
            val = getattr(curr, key)  # 获取 curr 对象的属性值
            if not is_traceable_wrapper_subclass(val):
                plain_tensors.append(val)  # 如果属性值不是可追踪包装器的子类，则将其添加到 plain_tensors 中
            else:
                todo.append(val)  # 如果属性值是可追踪包装器的子类，则将其加入待处理列表
    return plain_tensors  # 返回普通张量列表


# 定义函数 is_fake，判断给定对象 x 是否为 FakeTensor 或其包含 FakeTensor 的混合张量
def is_fake(x):
    if isinstance(x, FakeTensor):
        return True  # 如果 x 是 FakeTensor 类型，则返回 True
    if is_traceable_wrapper_subclass(x):
        attrs, _ = type(x).__tensor_flatten__(x)  # 调用 x 对象类型的 __tensor_flatten__ 方法
        flattened_tensors = [getattr(x, attr) for attr in attrs]  # 获取所有扁平化的张量
        all_fake = all(is_fake(x) for x in flattened_tensors)  # 检查所有张量是否均为 FakeTensor
        any_fake = any(is_fake(x) for x in flattened_tensors)  # 检查是否有张量为 FakeTensor
        assert all_fake == any_fake, "got mixed fake and real tensors!"  # 断言所有张量的真假一致性
        return all_fake  # 返回所有张量是否均为 FakeTensor
    elif isinstance(x, torch.Tensor) and torch._is_functional_tensor(x):
        reapply_views = torch._C._functionalization_reapply_views_tls()  # 获取功能化重应用视图
        unwrapped = torch._C._functorch._unwrap_functional_tensor(x, reapply_views)  # 解包功能化张量
        return is_fake(unwrapped)  # 递归判断解包后的张量是否为 FakeTensor
    elif isinstance(x, torch.Tensor) and is_functorch_wrapped_tensor(x):
        unwrapped = torch._C._functorch.get_unwrapped(x)  # 获取 Functorch 封装张量的解包版本
        return is_fake(unwrapped)  # 判断解包后的张量是否为 FakeTensor
    return False  # 默认返回 False，表示 x 不是 FakeTensor


# 定义函数 maybe_get_fake_mode，获取给定对象 t 的虚拟模式
def maybe_get_fake_mode(t):
    if isinstance(t, FakeTensor):
        return t.fake_mode  # 如果 t 是 FakeTensor 类型，则返回其 fake_mode
    if is_traceable_wrapper_subclass(t):
        inner_tensor_names, _ = t.__tensor_flatten__()  # 调用 t 对象的 __tensor_flatten__ 方法
        modes = [
            maybe_get_fake_mode(getattr(t, t_name)) for t_name in inner_tensor_names
        ]  # 获取所有内部张量的虚拟模式
        m = modes[0]  # 获取第一个内部张量的模式
        assert all(m is x for x in modes)  # 断言所有内部张量的模式一致
        return m  # 返回模式 m
    elif isinstance(t, torch.Tensor) and torch._is_functional_tensor(t):
        reapply_views = torch._C._functionalization_reapply_views_tls()  # 获取功能化重应用视图
        unwrapped = torch._C._functorch._unwrap_functional_tensor(t, reapply_views)  # 解包功能化张量
        return maybe_get_fake_mode(unwrapped)  # 递归获取解包后张量的虚拟模式
    elif isinstance(t, torch.Tensor) and is_functorch_wrapped_tensor(t):
        unwrapped = torch._C._functorch.get_unwrapped(t)  # 获取 Functorch 封装张量的解包版本
        return maybe_get_fake_mode(unwrapped)  # 获取解包后张量的虚拟模式
    return None  # 默认返回 None，表示无法获取虚拟模式


# 使用 functools.lru_cache(None) 装饰器定义函数 get_schema_info，获取给定函数 func 的模式信息
@functools.lru_cache(None)
def get_schema_info(func):
    return torch._C._SchemaInfo(func._schema)  # 返回 func 函数的模式信息
# torch/_decomp/decompositions.py.
# decomps are used for aot autograd tracing so we would like to unify on their
# implementation and add additional testing to them

# 使用 functools.lru_cache 装饰器缓存函数的结果，None 表示缓存大小不限
@functools.lru_cache(None)
# 定义一个函数，接受一个参数 func
def torch_decomp_decompositions(func):
    # 导入 torch._decomp 模块中的 decomposition_table
    from torch._decomp import decomposition_table

    # 获取 torch._decomp 模块中的 decompositions 对象
    decompositions = torch._decomp.decompositions

    # 检查 decomposition_table 中 func 对应的函数是否位于 "torch._decomp" 模块中
    # 并且其名称是否存在于 decompositions 对象的属性列表中
    return decomposition_table[func].__module__.startswith(
        "torch._decomp"
    ) and decomposition_table[func].__name__ in dir(decompositions)


# 接受两个参数 ty 和 tree，其中 ty 是类型 T 的子类，tree 是一个 PyTree 对象
def tree_flatten_only(ty: Type[T], tree: PyTree):
    # 使用 pytree 模块的 tree_leaves 函数获取 tree 中的所有叶子节点
    flat_vals = pytree.tree_leaves(tree)
    # 返回 flat_vals 中所有类型为 ty 的元素组成的列表
    return [elem for elem in flat_vals if isinstance(elem, ty)]


# 类 FakeTensorConverter，用于将多个张量转换为共享相同视图/存储结构的伪张量
class FakeTensorConverter:
    # 属性 tensor_memo 返回 meta_converter 的 tensor_memo 属性
    @property
    def tensor_memo(self):
        return self.meta_converter.tensor_memo

    # 类型为 MetaConverter 的属性 meta_converter
    meta_converter: MetaConverter
    # 存储映射常量张量的字典，键为 StorageWeakRef，值为 ReferenceType 列表
    constant_storage_mapping: Dict[StorageWeakRef, List[ReferenceType]]
    # 布尔型属性 export
    export: bool

    # 构造函数，接受一个命名关键字参数 copy_data 和 export，默认为 False
    def __init__(self, *, copy_data=False, export=False):
        # 初始化 meta_converter 属性为 MetaConverter 类的实例
        self.meta_converter = MetaConverter(copy_data=copy_data)
        # 初始化 export 属性
        self.export = export

        # 初始化 constant_storage_mapping 属性为空字典
        # 用于映射存储到相应常量张量的映射关系
        self.constant_storage_mapping = {}

    # 向 constant_storage_mapping 添加常量存储映射
    def add_constant_storage_mapping(self, fake_tensor):
        # 断言 fake_tensor 是 FakeTensor 类的实例，并且其 constant 属性不为 None
        assert isinstance(fake_tensor, FakeTensor) and fake_tensor.constant is not None
        # 创建 weak_st 作为 fake_tensor.constant 的 StorageWeakRef
        weak_st = StorageWeakRef(fake_tensor.constant._typed_storage())

        # 如果 weak_st 不在 constant_storage_mapping 中，则将其初始化为一个空列表
        if weak_st not in self.constant_storage_mapping:
            self.constant_storage_mapping[weak_st] = []
        
        # 将 fake_tensor 的弱引用添加到 constant_storage_mapping[weak_st] 中
        self.constant_storage_mapping[weak_st].append(weakref.ref(fake_tensor))

    # 使常量别名失效
    def invalidate_constant_aliases(self, tensor):
        # 断言 tensor 不是 FakeTensor 的实例
        assert not isinstance(tensor, FakeTensor)

        # 创建 weak_st 作为 tensor 的 StorageWeakRef
        weak_st = StorageWeakRef(tensor._typed_storage())

        # 如果 weak_st 不在 constant_storage_mapping 中，则直接返回
        if weak_st not in self.constant_storage_mapping:
            return

        # 遍历 constant_storage_mapping[weak_st] 中的所有弱引用
        for weak_tensor_ref in self.constant_storage_mapping[weak_st]:
            ten = weak_tensor_ref()
            # 如果弱引用不为 None，则调用其 _fix_weakref 方法，并将 constant 属性设置为 None
            if ten is not None:
                ten._fix_weakref()
                ten.constant = None

        # 删除 constant_storage_mapping 中的 weak_st
        del self.constant_storage_mapping[weak_st]
    # 根据给定的张量 t，查找其在描述器中的对应 ID
    tid = self.meta_converter.describer.lookup_tensor.get(t)
    # 如果未找到对应的 ID，返回 None
    if tid is None:
        return None
    # 根据找到的 ID 获取对应的张量备忘录（如果存在）
    return self.tensor_memo.get(tid)

    # 设置指定张量 t 的备忘录值为 v
    tid = self.meta_converter.describer.get_tensor_id(t)
    self.meta_converter.tensor_memo[tid] = v

    # 你可能有一个真实张量需要转换成虚假张量。
    # 如果已有一个元张量，调用 from_meta_and_device 方法。
    #
    # 你可以传递一个元张量以转换成虚假张量；
    # 虽然这种情况较为奇怪，但在进行交叉引用测试时，内部测试可能已经在操作元张量。
    def from_real_tensor(
        self,
        fake_mode,
        t,
        make_constant=False,
        shape_env=None,
        *,
        source=None,
        symbolic_context=None,
        trace=True,
    ):
        # 如果指定了设备，必须是一个元张量。
        assert (
            t.device.type == "meta"
        ), f"tensor's device must be `meta`, got {t.device.type} instead"
        # 这有点滥用（这不是“真实”的张量），但无论如何，
        # 元张量应该是新鲜的，因此不可能出错
        # 获取张量 t 的备忘录（如果存在）
        maybe_memo = self._get_memo(t)
        # 如果备忘录存在，则直接返回备忘录
        if maybe_memo is not None:
            return maybe_memo
        # 创建一个虚假张量，并将其设置为张量 t 的备忘录
        out = FakeTensor(fake_mode, t, device)
        self.set_tensor_memo(t, out)
        return out
# 使用 functools 模块的 lru_cache 装饰器，缓存 init_cuda_context 函数的结果，None 表示无限制缓存
@functools.lru_cache(None)
def init_cuda_context():
    # 如果 CUDA 可用，创建一个空的张量在 CUDA 设备上，用于触发 CUDA 的初始化
    if torch.cuda.is_available():
        # 根据 torch.version.hip 的存在性选择创建空张量或者全零张量在 CUDA 设备上
        torch.empty(1, device="cuda") if torch.version.hip is None else torch.zeros(
            1, device="cuda"
        )


# 创建上下文管理器 in_kernel_invocation_manager，用于管理内核调用过程中的 fake_mode
@contextlib.contextmanager
def in_kernel_invocation_manager(fake_mode):
    # 记录当前的 fake_mode.in_kernel_invocation 状态
    prev_in_kernel = fake_mode.in_kernel_invocation
    # 检查当前 torch._C._meta_in_tls_dispatch_include() 的值与前一个 fake_mode.in_kernel_invocation 是否相同
    meta_in_tls = torch._C._meta_in_tls_dispatch_include()
    assert meta_in_tls == prev_in_kernel, f"{meta_in_tls}, {prev_in_kernel}"

    # 使用 torch._C._DisableTorchDispatch() 禁用 Torch 的 dispatch
    with torch._C._DisableTorchDispatch():
        # 设置 fake_mode.in_kernel_invocation 为 True，表示当前处于内核调用中
        fake_mode.in_kernel_invocation = True
        # 使用 torch._C._PreserveDispatchKeyGuard() 保留 dispatch key 的设置
        with torch._C._PreserveDispatchKeyGuard():
            # 设置 torch._C._set_meta_in_tls_dispatch_include(True)，确保 meta dispatch include 在 TLS 中设置为 True
            torch._C._set_meta_in_tls_dispatch_include(True)
            try:
                yield  # 执行上下文管理器代码块
            finally:
                # 恢复 fake_mode.in_kernel_invocation 到先前的状态
                fake_mode.in_kernel_invocation = prev_in_kernel
                # 恢复 torch._C._set_meta_in_tls_dispatch_include(prev_in_kernel) 的设置


# 返回一个布尔值，指示是否应允许 Python 数字绑定到张量
def should_allow_numbers_as_tensors(func: OpOverload):
    return torch._C._should_allow_numbers_as_tensors(
        func.name().split("::")[-1].split(".")[0]
    )


# 定义一个配置类 FakeTensorConfig，用于调试环境中的假张量
class FakeTensorConfig:
    # debug 属性，从环境变量 TORCH_FAKE_TENSOR_DEBUG 中获取，如果为 "1" 则为 True，否则为 False
    debug = os.environ.get("TORCH_FAKE_TENSOR_DEBUG", "0") == "1"


# 定义一个描述符类 UnbackedMemoDescriptor，用于缓存无后端 SymInt 的表示
# 每个特定的数量（如张量中非零元素的数量）都有一个实例
class UnbackedMemoDescriptor:
    _name: str

    # 设置描述符的名称
    def __set_name__(self, owner, name):
        self._name = name

    # 返回缓存的无后端 SymInt 的键名
    def _memo(self, obj):
        return f"_{self._name}"

    # 返回缓存的无后端 SymInt 的版本计数器的键名
    def _memo_vc(self, obj):
        return f"_{self._name}_vc"

    # 返回缓存的无后端 SymInt 的 epoch 的键名
    def _memo_epoch(self, obj):
        return f"_{self._name}_epoch"
    # 在描述符类中定义 __get__ 方法，用于获取属性值
    def __get__(self, obj: "FakeTensor", objtype=None):
        # 如果对象的缓存属性值为 None，则返回 None
        if (r := getattr(obj, self._memo(obj))) is None:
            return None
        # 版本计数器追踪不是完全可靠的，但已经足够接近
        # 检查对象的版本号和伪模式的周期是否匹配
        if (
            getattr(obj, self._memo_vc(obj)) != obj._version
            or getattr(obj, self._memo_epoch(obj)) != obj.fake_mode.epoch
        ):
            # 如果不匹配，将对象的缓存属性置为 None，然后返回 None
            setattr(obj, self._memo(obj), None)
            return None
        # 返回缓存的属性值
        return r

    # 在描述符类中定义 __set__ 方法，用于设置属性值
    def __set__(self, obj, value):
        # 如果设置的值为 None，将对象的缓存属性、版本号计数器和周期计数器都置为 None
        if value is None:
            setattr(obj, self._memo(obj), None)
            setattr(obj, self._memo_vc(obj), None)
            setattr(obj, self._memo_epoch(obj), None)
        # 如果推断模式未启用
        elif not torch.is_inference_mode_enabled():
            # 设置对象的缓存属性为给定值，版本号计数器为对象的版本号，周期计数器为伪模式的周期
            setattr(obj, self._memo(obj), value)
            setattr(obj, self._memo_vc(obj), obj._version)
            setattr(obj, self._memo_epoch(obj), obj.fake_mode.epoch)
class FakeTensor(torch.Tensor):
    """
    Meta tensors give you the ability to run PyTorch code without having to
    actually do computation through tensors allocated on a `meta` device.
    Because the device is `meta`, meta tensors do not model device propagation.
    FakeTensor extends MetaTensors to also carry an additional `fake_device`
    which tracks devices that would have been used.
    """

    fake_device: torch.device
    fake_mode: "FakeTensorMode"
    constant: Optional[torch.Tensor]
    real_tensor: Optional[torch.Tensor]

    # TODO: Generalize this as needed, e.g., into a trie of memos, if
    # you do something like x[0].item()  (x[0] is fresh each time, so
    # memo mechanism here won't work)
    # 定义未支持的内存备忘录描述符，用于非受支持的操作备忘录
    nonzero_memo = UnbackedMemoDescriptor()
    item_memo = UnbackedMemoDescriptor()
    unique_memo = UnbackedMemoDescriptor()

    # Indicates to our torch_dispatch dispatching infra that
    # this is an "infra" mode with lower dispatching precedence.
    # 设置 _mode_key 作为 Torch 分发模式的键，用于指示“infra”模式
    _mode_key = torch._C._TorchDispatchModeKey.FAKE

    @property
    def device(self):
        if self.fake_mode.in_kernel_invocation:
            return torch.device("meta")
        else:
            return self.fake_device

    # Note: [Fake Tensor Dispatch Keys]
    # In order to model the behavior of device-specific autocast
    # and autograd logic, we update the dispatch keys of FakeTensors
    # to reflect their fake device. This includes the BackendComponent
    # (DispatchKey::Meta -> DispatchKey::CUDA), and also the BackendComponent
    # related Autocast and Autograd keys. __torch__dispatch__ sits below
    # Autocast and Autograd, and is only invoked when we are at the
    # kernel for the BackendComponent. Then, we add Meta to the
    # thread-local dispatch include set to hit the meta kernel
    # instead of the kernel of the BackendComponent for the fake device.
    # The `device_for_backend_keys` does that below
    # NOTE: this probably will not do the right thing for backends
    # that have dispatch keys which are higher than the "meta" key:
    # https://github.com/pytorch/pytorch/blob/main/c10/core/DispatchKey.h#L189

    # We don't support named tensors; graph break
    @property
    def names(self):
        raise UnsupportedFakeTensorException(
            "torch.compile doesn't support named tensors"
        )

    @staticmethod
    def __new__(cls, fake_mode, elem, device, constant=None, real_tensor=None):
        # 创建一个新的子类实例，基于传入的元素和设备信息
        self = torch.Tensor._make_subclass(
            cls,
            elem,
            elem.requires_grad,
            dispatch_device=True,
            device_for_backend_keys=device,
        )
        # 如果不允许不安全的数据指针访问，则设置抛出异常
        if not fake_mode._allow_unsafe_data_ptr_access:
            torch._C._set_throw_on_mutable_data_ptr(self)
        else:
            # 否则设置为在可变数据指针上发出警告（已废弃）
            torch._C._set_warn_deprecated_on_mutable_data_ptr(self)

        # 断言元素所在设备类型为"meta"，用于确认当前情景
        assert elem.device.type == "meta", elem.device.type
        # 将设备信息规范化为torch.device对象，若已是torch.device则不改变
        device = device if isinstance(device, torch.device) else torch.device(device)
        # 注意：设备类型为"meta"是允许的，但通常表明可能有混淆或错误使用的情况
        if not fake_mode.allow_meta:
            assert device.type != "meta"
        # 如果设备类型为"cuda"，初始化CUDA上下文
        if device.type == "cuda":
            init_cuda_context()

        # 如果设备类型是"cuda"、"hpu"、"xpu"或者私有后端名，且索引为None
        if (
            device.type
            in ["cuda", "hpu", "xpu", torch._C._get_privateuse1_backend_name()]
            and device.index is None
        ):
            # 如果该设备类型已初始化，则使用当前设备；否则使用默认设备
            if getattr(torch, device.type).is_initialized():
                device = torch.device(
                    f"{device.type}:{getattr(torch, device.type).current_device()}"
                )
            else:
                device = torch.device(f"{device.type}:0")
        
        # 设置虚假设备和虚假模式属性
        self.fake_device = device  # type: ignore[attr-defined]
        self.fake_mode = fake_mode  # type: ignore[attr-defined]
        self.constant = constant  # type: ignore[attr-defined]
        # 断言实际张量不是FakeTensor类型
        assert not isinstance(real_tensor, FakeTensor)
        self.real_tensor = real_tensor  # type: ignore[attr-defined]
        # 初始化用于优化的缓存变量
        self.nonzero_memo = None
        self.item_memo = None
        self.unique_memo = None

        # 如果处于调试模式，记录调试追踪信息
        if FakeTensorConfig.debug:
            self._debug_trace = CapturedTraceback.extract()  # type: ignore[attr-defined]
        return self
    # 定义类的初始化方法，接受任意数量的位置参数和关键字参数
    def __init__(self, *args, **kwargs):
        # 调用父类的初始化方法
        super().__init__()

    # 从张量创建 FakeTensor 的静态方法
    @staticmethod
    def from_tensor(t, fake_mode):
        # 调用 fake_mode 对象的 from_tensor 方法，并返回结果
        return fake_mode.from_tensor(t)

    # 用于查找通用设备的类方法，统计调用次数的装饰器
    @classmethod
    @count
    @staticmethod
    def _find_common_device(func, flat_args) -> Tuple[torch.device, bool]:
        # 返回元组 (common_device, has_scalar_only_inputs)
        
        # 初始化变量
        common_device = None
        has_scalar_only_inputs = False
        is_cpu_zero_dim = None

        # 判断是否是 CPU 上的零维张量
        def cpu_zero_dim(t):
            return t.device.type == "cpu" and t.dim() == 0

        # 合并设备信息
        def merge_devices(t):
            nonlocal common_device
            nonlocal is_cpu_zero_dim
            
            # 如果不是 FakeTensor 对象，直接返回
            if not isinstance(t, FakeTensor):
                return

            # 如果 common_device 还未设置，则使用当前张量的设备
            if common_device is None:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            # 如果当前张量的设备与 common_device 相同
            if t.device == common_device:
                # 如果当前张量是 CPU 上的零维张量，并且 common_device 也是，更新 is_cpu_zero_dim
                if is_cpu_zero_dim:
                    is_cpu_zero_dim = cpu_zero_dim(t)
                return

            # 设备不匹配！
            # 如果当前张量是 CPU 上的零维张量，则暂时使用现有的设备
            if cpu_zero_dim(t):
                return

            # 当前设备来自 CPU 上的零维张量，进行覆盖
            if is_cpu_zero_dim:
                common_device = t.device
                is_cpu_zero_dim = cpu_zero_dim(t)
                return

            # 非零维张量的设备不匹配，抛出异常
            raise RuntimeError(
                f"Unhandled FakeTensor Device Propagation for {func}, found two different devices {common_device}, {t.device}"
            )

        # 遍历扁平化的参数列表
        for arg in flat_args:
            merge_devices(arg)

        # 对于允许 Python 数字绑定到张量的函数，且未找到设备时，标记只有标量输入
        if should_allow_numbers_as_tensors(func) and common_device is None:
            has_scalar_only_inputs = True
            common_device = torch.device("cpu")  # 标量输入的运算结果一般在 CPU 上

        # 断言一定找到了通用设备
        assert common_device is not None, f"Could not find common device for {func}"

        # 返回通用设备和是否只有标量输入的标志
        return common_device, has_scalar_only_inputs

    # 对 FakeTensor 中的 tolist 方法进行特殊处理
    # 当 tolist 在张量子类的 torch 分发中被调用时，需要特殊处理
    # 普通情况下，如果程序调用 .tolist 编译仍可正常工作，因为 dynamo 中有特殊处理
    # 但对于张量子类，如果 .tolist 在 torch 分发中被调用，可能直接作用在 FakeTensor 上
    # 定义一个方法 `tolist`，用于将当前对象转换为 Python 列表
    # 确保张量维度为1，否则会抛出错误
    def tolist(self):
        assert self.dim() == 1, "NYI for higher dims"
        
        # 获取虚拟模式中的形状环境
        shape_env = self.fake_mode.shape_env
        
        # 初始化空列表 `out`，用于存储转换后的结果
        out = []
        
        # 根据列表长度循环创建对应数量的未支持的符号整数（symint）
        for _ in range(self.shape[0]):
            # 创建一个未支持的符号整数（symint）
            s = shape_env.create_unbacked_symint()
            
            # 检查符号整数 `s` 是否为有效大小
            torch._check_is_size(s)
            
            # 检查符号整数 `s` 是否大于等于2
            torch._check(s >= 2)
            
            # 将符号整数 `s` 添加到输出列表 `out` 中
            out.append(s)
        
        # 返回生成的列表 `out`
        return out
# 使用 @dataclass 装饰器创建一个名为 TensorMetadata 的数据类，表示张量的元数据
@dataclass(frozen=True)
class TensorMetadata:
    """
    The Tensor metadata relevant to hashing FakeTensors when caching.
    """

    # 张量的数据类型
    dtype: torch.dtype
    # 张量的形状
    shape: torch.Size
    # 张量的步幅
    stride: Tuple[Any, ...]
    # 张量所在的设备
    device: torch.device
    # 张量的布局
    layout: torch.layout
    # 张量的内存格式，可选
    memory_format: Optional[torch.memory_format]
    # 张量的存储偏移量
    storage_offset: int
    # 张量的存储字节数，对于稀疏张量为 None
    storage_bytes: Optional[int]
    # 张量是否需要梯度
    requires_grad: bool
    # 张量是否量化
    is_quantized: bool
    # 张量是否共轭
    is_conj: bool
    # 张量是否负数
    is_neg: bool
    # 张量是否推断中
    is_inference: bool
    # 张量是否稀疏，针对 COO 格式
    is_sparse: bool  # read: is sparse COO
    # 张量是否已聚合，对于稀疏张量可选
    is_coalesced: Optional[bool]
    # 稀疏张量的稠密维度，可选
    dense_dim: Optional[int]
    # 稀疏张量的稀疏维度，可选
    sparse_dim: Optional[int]


# 函数提取张量 t 的元数据，并返回一个 TensorMetadata 对象
def extract_tensor_metadata(t: torch.Tensor) -> "TensorMetadata":
    """
    Extract the TensorMetadata of a tensor.
    """
    # 建议张量的内存格式
    memory_format: Optional[torch.memory_format] = suggest_memory_format(t)
    # 如果张量是稀疏的或者不是连续的，则内存格式设为 None
    if is_sparse_any(t) or not t.is_contiguous(memory_format=memory_format):
        memory_format = None

    # 返回一个 TensorMetadata 对象，包含张量的各种元数据
    return TensorMetadata(
        dtype=t.dtype,
        shape=t.shape,
        stride=t.stride() if t.layout == torch.strided else (),
        device=t.device,
        layout=t.layout,
        memory_format=memory_format,
        storage_offset=t.storage_offset(),
        # 仅对有存储的张量设置存储字节数（稀疏张量为 None）
        storage_bytes=t.untyped_storage().nbytes() if not t.is_sparse else None,
        requires_grad=t.requires_grad,
        is_quantized=t.is_quantized,
        is_conj=t.is_conj(),
        is_neg=t.is_neg(),
        is_inference=t.is_inference(),
        is_sparse=t.is_sparse,
        is_coalesced=t.is_coalesced() if t.is_sparse else None,
        dense_dim=t.dense_dim() if t.is_sparse else None,
        sparse_dim=t.sparse_dim() if t.is_sparse else None,
    )


# 创建一个私有类 _DispatchCacheKey，继承自 list，用于 FakeTensor 调度缓存的键
class _DispatchCacheKey(list):
    """
    Key for the FakeTensor dispatch cache. Inspired by (copied from)
    _HashedSeq from the functools.lru_cache implementation.
    """

    # 只允许实例具有 hashvalue 属性
    __slots__ = "hashvalue"  # noqa: PLC0205

    # 初始化方法，接受一个元组 tup 和一个哈希函数 hash
    def __init__(self, tup, hash=hash):
        # 将实例设置为元组 tup 的内容
        self[:] = tup
        # 计算元组 tup 的哈希值并设置为实例的 hashvalue 属性
        self.hashvalue = hash(tup)

    # 返回实例的哈希值
    def __hash__(self):
        return self.hashvalue


# 使用 @dataclass 装饰器创建一个私有数据类 _DispatchCacheEntry，表示 FakeTensor 调度缓存的条目
@dataclass(frozen=True)
class _DispatchCacheEntry:
    """
    Entry type for the FakeTensor dispatch cache. Accounts for two possibilities:
    1) The op is inplace, and a hit means we need to alias the argument at a given
    index. 2) We need to synthesize a new FakeTensor given tensor metadata. For view
    ops, we further capture the index of the arg to alias.
    """

    # 就地操作的索引，如果命中则需要在给定索引处别名参数
    inplace_idx: Optional[int] = None
    # 张量元数据，如果需要合成新的 FakeTensor 则有效
    metadata: Optional[TensorMetadata] = None
    # 视图操作的参数索引，对于视图操作有效
    view_idx: Optional[int] = None


# 使用 @dataclass 装饰器创建一个私有数据类 _BypassDispatchCache，表示应跳过 FakeTensor 缓存的情况
@dataclass(frozen=True)
class _BypassDispatchCache(Exception):
    """
    Signals cases that should skip FakeTensor caching.
    """

    # 跳过缓存的原因描述
    reason: str


# 使用 @dataclass 装饰器创建一个名为 DispatchCacheInfo 的数据类，表示 FakeTensor 调度缓存的信息
@dataclass(frozen=True)
class DispatchCacheInfo:
    """
    Information about the state of the FakeTensor dispatch cache.
    """

    # 命中的缓存次数
    hits: int
    # 未命中的缓存次数
    misses: int
    # 被绕过的缓存计数，以跳过的原因作为键
    bypasses: Dict[str, int]
    # 缓存的大小
    size: int
# We keep one instantiation of `fake_tensor_converter` active
# for the duration of `with FakeTensorMode()`.
# This allows accurate storage aliasing across invocation of
# different operators. While this will keep all freshly allocated
# tensors alive during `FakeTensorMode`, there will no be no
# new allocations of Tensors which have non-meta storage so
# memory should not significantly increase.

# 定义一个名为 FakeTensorMode 的类，继承自 TorchDispatchMode
class FakeTensorMode(TorchDispatchMode):
    # 缓存，用于存储分发调度的缓存项
    cache: Dict[_DispatchCacheKey, _DispatchCacheEntry] = {}
    # 缓存命中次数统计
    cache_hits: int = 0
    # 缓存未命中次数统计
    cache_misses: int = 0
    # 缓存绕过次数统计
    cache_bypasses: Dict[str, int] = defaultdict(int)
    
    # 每次使用相同的 fake tensor mode 重建时，需要增加 epoch 以确保不复用未支持的内存
    epoch: int = 0
    # 是否在内核调用中
    in_kernel_invocation: bool = False

    # 初始化方法
    def __init__(
        self,
        *,
        allow_fallback_kernels=True,
        allow_non_fake_inputs=False,
        shape_env=None,
        static_shapes=None,
        # TODO: This is a temporary measure, see
        # https://github.com/pytorch/pytorch/pull/126245#discussion_r1604185748
        # We're currently solely using this to impede population of
        # item_memo for 0d scalar tensor inputs when export, because this
        # causes things that used to be deferred runtime asserts to turn into
        # guards, and then the guards are just lost.  We can potentially fix
        # this by ensuring guards also get put in the graph, but this is
        # pending a rework of how deferred runtime asserts in export.  Once
        # that's done, we can remove this.
        # 导出标志，用于阻止对 0 维标量张量输入进行 item_memo 的填充
        export=False,
        ):
        # 使用 debug 日志记录当前实例的创建模式，包括实例 ID
        log.debug("create_mode 0x%x", id(self))
        # 设置是否允许回退到 fallback 内核
        self.allow_fallback_kernels = allow_fallback_kernels

        # 导入 Dynamo 和 Functorch 的配置模块
        import torch._dynamo.config
        import torch._functorch.config

        # 根据 Functorch 配置设置是否传播真实张量
        self.propagate_real_tensors = (
            torch._functorch.config.fake_tensor_propagate_real_tensors
        )
        # 创建 FakeTensorConverter 实例，根据传播真实张量的设置复制数据并导出
        self.fake_tensor_converter = FakeTensorConverter(
            copy_data=self.propagate_real_tensors,
            export=export,
        )

        # 如果提供了静态形状，则使用提供的静态形状；否则根据形状环境判断是否为静态形状
        if static_shapes is not None:
            self.static_shapes = static_shapes
        else:
            self.static_shapes = shape_env is None

        # 在 Dynamo 中临时设置为 True，用于允许在某些情况下无条件允许标量输出，将来会移除
        self.allow_scalar_outputs = False

        # 根据 Functorch 配置设置是否允许不安全的数据指针访问
        self._allow_unsafe_data_ptr_access = (
            torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access
        )
        # 根据 Functorch 配置设置是否允许 FakeTensor 元信息
        self.allow_meta = torch._functorch.config.fake_tensor_allow_meta
        # 设置是否启用 FakeTensor 缓存，同时确保不传播真实张量
        self.cache_enabled = (
            torch._dynamo.config.fake_tensor_cache_enabled
            and not self.propagate_real_tensors
        )
        # 设置是否启用 FakeTensor 缓存交叉检查
        self.cache_crosscheck_enabled = (
            torch._dynamo.config.fake_tensor_cache_crosscheck_enabled
        )

        # 控制标志，确定是否允许在混合使用真实权重/全局变量和虚假输入时执行操作
        self.allow_non_fake_inputs = allow_non_fake_inputs

        # [in_kernel_invocation]
        # 当 FakeTensor 在用户代码中被调用时，.device 应该返回张量的虚假设备，
        # 以确保像 `if x.is_cuda` 或 torch.zeros([10, 10], device=x.device) 这样的代码继续执行，
        # 就像 FakeTensor 是真实的一样。但是，在内核执行中，我们返回 `Meta` 设备，
        # 因为内核中的所有计算都应该表现得像张量在元设备上一样。
        # 内核应该在元设备上分配新张量，并且像 `is_meta` 这样的检查应该返回 true。
        # 在 Python 引用中，我们始终通过定义 device 属性返回真实设备。
        self.in_kernel_invocation = False

        # 如果我们进入并实际启用了 FakeTensor 模式，则为 True；如果没有，则为 False。
        # 不是线程安全的，但内核调用也不是。
        # 如果在进入时已经存在其他虚假模式，则也在此处存储它。
        # 这样在退出时，我们就知道要重新启用先前的虚假模式。
        self.enter_stack: List[
            Tuple[bool, Optional[TorchDispatchMode], Optional[_bool]]
        ] = []

        # 设置形状环境，用于形状推断
        self.shape_env: ShapeEnv = shape_env

        # 提取当前调用堆栈，用于调试目的
        self._stack_trace = traceback.extract_stack()
        self._stack = None

        # 表示给我们的 torch_dispatch 调度基础设施一个信号，表明这是一个“基础设施”模式，具有较低的调度优先级。
        self._mode_key = torch._C._TorchDispatchModeKey.FAKE
    # 判断是否为我们定义的 FakeTensor 对象，并且其 fake_mode 属性与当前对象相同
    def is_our_fake(self, t):
        return isinstance(t, FakeTensor) and t.fake_mode is self

    # 判断是否需要避免设备初始化的属性
    @property
    def avoid_device_init(self):
        # 如果 CUDA 可用，则避免设备初始化，返回 True；否则返回 False
        return not torch.cuda.is_available()

    # 获取堆栈跟踪信息并将其连接成字符串
    @property
    def stack(self):
        if self._stack is None:
            self._stack = "".join(traceback.format_list(self._stack_trace))
        return self._stack

    # 自定义 torch 分发方法的装饰器，处理函数调用分发
    @count
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 确保在 FakeTensorMode 内部时不应设置 FakeTensorMode
        assert (
            torch._C._get_dispatch_mode(torch._C._TorchDispatchModeKey.FAKE) is None
        ), func
        try:
            # 调用实际的分发方法处理函数调用
            return self.dispatch(func, types, args, kwargs)
        except TypeError:
            log.exception("fake tensor raised TypeError")
            raise

    # 进入 FakeTensorMode 环境的上下文管理器方法
    def __enter__(self):
        prev_only_lift_cpu_tensors = None
        if self.avoid_device_init:
            # 如果需要避免设备初始化，设置仅提升 CPU 张量标志位
            prev_only_lift_cpu_tensors = torch._C._only_lift_cpu_tensors()
            torch._C._set_only_lift_cpu_tensors(True)
        # 取消当前 FakeTensorMode，返回之前的 FakeTensorMode
        maybe_prev_fake_mode = torch._C._unset_dispatch_mode(self._mode_key)
        # 如果当前对象不是之前的 FakeTensorMode，则记录进入栈
        if self is not maybe_prev_fake_mode:
            self.enter_stack.append(
                (True, maybe_prev_fake_mode, prev_only_lift_cpu_tensors)
            )
            return super().__enter__()  # 调用父类的 __enter__() 方法
        else:
            # 如果当前对象已经是之前的 FakeTensorMode，则重新设置 FakeTensorMode
            torch._C._set_dispatch_mode(self)
            self.enter_stack.append((False, None, prev_only_lift_cpu_tensors))
        return self
    # 退出当前上下文管理器
    def __exit__(self, a, b, c):
        # 弹出栈顶元素，包括活跃标志、可能的前一次伪模式、可能的仅提升CPU张量的标志
        (
            live,
            maybe_prev_fake_mode,
            maybe_prev_only_lift_cpu_tensors,
        ) = self.enter_stack.pop()
        # 如果有活跃的上下文
        if live:
            # 调用父类的 __exit__ 方法
            out = super().__exit__(a, b, c)
            # 如果之前存在伪模式，则重新启用
            if maybe_prev_fake_mode is not None:
                torch._C._set_dispatch_mode(maybe_prev_fake_mode)
            # 如果之前仅提升CPU张量，则重新启用
            if maybe_prev_only_lift_cpu_tensors is not None:
                torch._C._set_only_lift_cpu_tensors(maybe_prev_only_lift_cpu_tensors)

    # 类方法：返回调度缓存的信息
    @classmethod
    def cache_info(cls) -> DispatchCacheInfo:
        """
        Query the state of the dispatch cache.
        """
        return DispatchCacheInfo(
            FakeTensorMode.cache_hits,  # 缓存命中次数
            FakeTensorMode.cache_misses,  # 缓存未命中次数
            dict(FakeTensorMode.cache_bypasses),  # 缓存绕过的原因及其次数
            len(FakeTensorMode.cache),  # 缓存条目数量
        )

    # 类方法：清空调度缓存
    @classmethod
    def cache_clear(cls):
        """
        Clear the dispatch cache.
        """
        cls.cache_hits = 0  # 重置缓存命中次数
        cls.cache_misses = 0  # 重置缓存未命中次数
        cls.cache_bypasses.clear()  # 清空缓存绕过的原因及其次数的字典
        cls.cache.clear()  # 清空缓存条目的字典

    # 私有方法：缓存调度的具体实现
    def _cached_dispatch_impl(
        self,
        func: OpOverload,
        types: Tuple[Any, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Lookup a cache entry for the given arguments. If none exists, dispatch
        and cache the result (if the result is eligible for caching).
        """
        output: Union[FakeTensor, _Unassigned] = _UNASSIGNED
        try:
            key = self._cache_key(func, args, kwargs)  # 生成缓存键
            entry = FakeTensorMode.cache.get(key, None)  # 从缓存中获取条目
            # 如果缓存中有条目
            if entry is not None:
                output = self._output_from_cache_entry(entry, func, args)  # 从缓存条目中获取输出
                FakeTensorMode.cache_hits += 1  # 增加缓存命中次数
                if self.cache_crosscheck_enabled:
                    # 用于调试/测试：验证从缓存中合成的输出是否与正常调度创建的输出匹配
                    self._crosscheck_cache_output(output, func, types, args, kwargs)
            else:
                self._validate_cache_key(func, args, kwargs)  # 验证缓存键的有效性
                output = self._dispatch_impl(func, types, args, kwargs)  # 调度具体实现
                entry = self._make_cache_entry(key, func, args, kwargs, output)  # 创建缓存条目
                FakeTensorMode.cache[key] = entry  # 将条目存入缓存
                FakeTensorMode.cache_misses += 1  # 增加缓存未命中次数
        except _BypassDispatchCache as e:
            FakeTensorMode.cache_bypasses[e.reason] += 1  # 增加对应绕过缓存的次数

        # 如果输出仍为未分配状态，则重新调度具体实现
        if output is _UNASSIGNED:
            output = self._dispatch_impl(func, types, args, kwargs)

        return output
    ) -> _DispatchCacheKey:
        """
        Create a cache key given the dispatch args. Raises _BypassDispatchCache
        for any situation that precludes caching.
        """
        key_values = (
            func,
            # Translate any FakeTensor args to metadata.
            self._prep_args_for_hash(args) if args else (),
            self._prep_args_for_hash(kwargs) if kwargs else (),
            # Capture the default_dtype mode since that can affect the output tensor,
            # e.g., when operating on constant float values.
            torch.get_default_dtype(),
            # Capture the current device to support, e.g., cache tensor creation,
            # where there isn't necessarily a tensor to take the device from.
            torch._C._get_default_device(),
            # We want to create tensors from cached metadata only when the inference
            # mode is the same.
            torch.is_inference_mode_enabled(),
            # Shape env settings could affect behavior. One example seen in the wild:
            # Disallowing dynamic shapes can introduce a DynamicOutputShapeException
            # where it wasn't seen on a previous instance of the same op.
            self.shape_env.settings if self.shape_env else None,
        )
        # 返回一个 _DispatchCacheKey 对象，用于缓存分派的键
        return _DispatchCacheKey(key_values)

    def _validate_cache_key(
        self,
        func: OpOverload,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Validate that the cache key generated by _cache_key will be
        reasonable.
        """
        # 针对生成的缓存键进行验证，确保其合理性

        # 如果操作标记为数据依赖输出，则抛出异常以绕过缓存
        if torch.Tag.data_dependent_output in func.tags:
            raise _BypassDispatchCache("data dependent output")

        # 如果操作标记为动态输出形状，则抛出异常以绕过缓存
        if torch.Tag.dynamic_output_shape in func.tags:
            raise _BypassDispatchCache("dynamic output shape")

        # 如果操作标记为就地视图，则抛出异常以绕过缓存
        if torch.Tag.inplace_view in func.tags:
            raise _BypassDispatchCache("inplace view")

        # 如果操作是 unsafe_view 的默认实现，则抛出异常以绕过缓存
        if func == aten._unsafe_view.default:
            raise _BypassDispatchCache("unsafe view")

        # 如果操作在 lift 函数列表中，则抛出异常以绕过缓存
        if func in self.lift_fns:
            raise _BypassDispatchCache("lift")

        # 如果操作的名称为 "inductor::resize_storage_bytes_"，则抛出异常以绕过缓存
        if func.name() == "inductor::resize_storage_bytes_":
            raise _BypassDispatchCache("inductor::resize_storage_bytes_")

        # 如果操作不是内置函数，则抛出异常以绕过缓存
        if not torch._library.utils.is_builtin(func):
            raise _BypassDispatchCache("non-builtin")

        # 在处理存储别名时，需要在缓存命中时建立别名，但是 CompositeImplicitAutograd 操作
        # 可能会或可能不会给输入建立别名，因此在这种情况下也绕过缓存。
        if func.is_view and torch._C._dispatch_has_kernel_for_dispatch_key(
            func.name(), torch._C.DispatchKey.CompositeImplicitAutograd
        ):
            raise _BypassDispatchCache("CompositeImplicitAutograd")
    def _prep_args_for_hash(self, args: Any) -> Any:
        """
        Translate the provided args into a form suitable for caching at FakeTensor
        dispatch, i.e., convert unhashable types like lists & dicts into tuples and
        convert FakeTensors into metadata. Raises _BypassDispatchCache to signal
        unsupported cases that should bypass caching.
        """
        # 如果 args 是一个字典，则将其键和值转换为列表并合并到 args 中
        if isinstance(args, dict):
            args = list(args.keys()) + list(args.values())

        # 初始化结果列表
        result: List[Any] = []
        # 遍历 args 中的每个参数
        for arg in args:
            # 如果参数是 FakeTensor 类型
            if isinstance(arg, FakeTensor):
                # 如果不是我们自己定义的 FakeTensor，抛出异常 _BypassDispatchCache
                if not self.is_our_fake(arg):
                    raise _BypassDispatchCache("not our fake")
                # 如果 FakeTensor 拥有符号化大小和步幅，抛出异常 _BypassDispatchCache
                if arg._has_symbolic_sizes_strides:
                    raise _BypassDispatchCache("symbolic shape")
                # 如果 FakeTensor 拥有常量属性，抛出异常 _BypassDispatchCache
                if arg.constant is not None:
                    raise _BypassDispatchCache("constant attribute")
                # 如果 FakeTensor 是稀疏张量，抛出异常 _BypassDispatchCache
                if arg.is_sparse:
                    raise _BypassDispatchCache("sparse tensor")
                # 如果 FakeTensor 的布局在稀疏张量的布局列表中，抛出异常 _BypassDispatchCache
                if arg.layout in [
                    torch.sparse_csr,
                    torch.sparse_csc,
                    torch.sparse_bsr,
                    torch.sparse_bsc,
                ]:
                    raise _BypassDispatchCache("sparse tensor layout")
                # 稀疏张量不具备存储空间，因此在检查稀疏性后进行检查
                if isinstance(arg.untyped_storage().nbytes(), torch.SymInt):
                    raise _BypassDispatchCache("symbolic nbytes")
                # 如果检测到压缩的稀疏张量，抛出异常 _BypassDispatchCache
                if is_sparse_compressed(arg):
                    raise _BypassDispatchCache("sparse compressed tensor")
                # 将 FakeTensor 转换为其元数据并添加到结果列表中
                result.append(extract_tensor_metadata(arg))
            # 如果参数是 torch.Tensor 类型，抛出异常 _BypassDispatchCache
            elif isinstance(arg, torch.Tensor):
                raise _BypassDispatchCache("non-fake tensor")
            # 如果参数是 torch.SymBool、torch.SymInt 或 torch.SymFloat 类型，抛出异常 _BypassDispatchCache
            elif isinstance(arg, (torch.SymBool, torch.SymInt, torch.SymFloat)):
                raise _BypassDispatchCache("symbolic shape")
            # 如果参数是列表、元组或字典类型，则递归调用 _prep_args_for_hash 处理其元素，并将结果扩展到结果列表中
            elif isinstance(arg, (list, tuple, dict)):
                result.extend(self._prep_args_for_hash(arg))
            else:
                # 对于其他类型的参数，添加其类型和值的元组到结果列表中
                # 这里重点是捕获参数的类型，例如，1 和 1.0 尽管哈希值相同，但可能会产生不同的输出张量 dtype。
                result.append((type(arg), arg))

        # 将结果列表转换为元组并返回
        return tuple(result)
    ) -> _DispatchCacheEntry:
        """
        创建给定 'output' 张量的缓存条目对象。如果输出张量具有阻止缓存的特性，则引发 _BypassDispatchCache 异常。
        """
        # 一些操作返回张量元组，但这种情况很少见，因此避免复杂化缓存其他类型的情况。
        if not isinstance(output, FakeTensor):
            raise _BypassDispatchCache("non-FakeTensor output")

        # 避免缓存带有常量属性的 FakeTensor，因为这些可能会无效化。
        if output.constant is not None:
            raise _BypassDispatchCache("constant attribute")

        # TODO: 支持缓存稀疏输出吗？
        if output.is_sparse:
            raise _BypassDispatchCache("sparse output")

        if is_sparse_compressed(output):
            raise _BypassDispatchCache("sparse compressed output")

        # 一个就地操作真的能引用一个关键字参数吗？如果可以，那么我们需要扩展实现来处理它。
        for kval in kwargs.values():
            if id(kval) == id(output):
                raise _BypassDispatchCache("kwarg aliases output")

        # 如果这是一个就地操作，条目记录哪个输入参数是别名。
        for idx in range(len(args)):
            if id(args[idx]) == id(output):
                return _DispatchCacheEntry(
                    inplace_idx=idx, metadata=None, view_idx=None
                )

        # 否则，创建一个条目，记录输出张量的元数据。
        view_idx = None
        if func.is_view:
            idxs = [i for i, t in enumerate(args) if isinstance(t, torch.Tensor)]
            assert len(idxs) == 1
            view_idx = idxs[0]

        metadata = extract_tensor_metadata(output)
        entry = _DispatchCacheEntry(
            inplace_idx=None, metadata=metadata, view_idx=view_idx
        )

        # 注意：某些绕过缓存的检查将在从缓存的元数据合成的输出张量上执行。作为优化，
        # 我们可以在此处合成一个张量并在该实例上执行检查。这种方法使得（更频繁的）缓存命中路径尽可能轻量化。
        synth_output = self._output_from_cache_entry(entry, func, args)

        # 确保合成输出张量的 dispatch_key_set 与原始输出张量相同。
        synth_key_set = torch._C._dispatch_key_set(synth_output)
        key_set = torch._C._dispatch_key_set(output)
        if synth_key_set != key_set:
            raise _BypassDispatchCache("dispatch_key_set mismatch")

        return entry
    ) -> FakeTensor:
        """
        Create a new FakeTensor from the cache entry.
        """
        # 如果操作是原地操作，则返回别名参数
        if entry.inplace_idx is not None:
            return args[entry.inplace_idx]

        # 使用缓存的元数据合成一个新的 FakeTensor
        metadata = entry.metadata
        assert metadata and not metadata.is_sparse

        # 使用缓存的元数据创建一个空的 torch tensor
        empty = torch.empty_strided(
            metadata.shape,
            metadata.stride,
            dtype=metadata.dtype,
            layout=metadata.layout,
            device="meta",
            requires_grad=metadata.requires_grad,
        )

        # 如果元数据指示为共轭，则设置为共轭
        if metadata.is_conj:
            torch._C._set_conj(empty, True)
        # 如果元数据指示为负数，则设置为负数
        if metadata.is_neg:
            torch._C._set_neg(empty, True)

        # 准备可能的上下文管理器以抑制异常
        maybe_suppress: Callable[[], Any] = contextlib.nullcontext
        if self.shape_env is not None:
            maybe_suppress = self.shape_env.suppress_guards

        # 如果是视图操作，则存储应该与输入张量相同
        if func.is_view:
            storage = args[cast(int, entry.view_idx)].untyped_storage()
            # 在内核调用管理器和可能的异常抑制下，设置空张量的值
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(
                    storage, metadata.storage_offset, metadata.shape, metadata.stride
                )
        # 如果存储偏移量不为零
        elif metadata.storage_offset != 0:
            storage = empty.untyped_storage()
            # 在内核调用管理器和可能的异常抑制下，设置空张量的值
            with in_kernel_invocation_manager(self), maybe_suppress():
                empty.set_(
                    storage, metadata.storage_offset, metadata.shape, metadata.stride
                )
        # 如果存储字节数为零，则调整空张量的存储大小为零
        if metadata.storage_bytes == 0:
            empty.untyped_storage().resize_(0)

        # 返回用自身、空张量和元数据设备创建的 FakeTensor
        return FakeTensor(self, empty, metadata.device)

    def _crosscheck_cache_output(
        self,
        output: FakeTensor,
        func: OpOverload,
        types: Tuple[Any, ...],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ):
        """
        Helper to validate that the output synthesized from the cache matches
        the output created by normal dispatch.
        """
        try:
            # 使用正常调度创建的输出进行缓存输出的交叉验证
            true_output = self._dispatch_impl(func, types, args, kwargs)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}: Dispatch raised={e}"
            ) from e
        try:
            # 断言缓存输出与真实输出的元数据相等
            assert_metadata_eq(assert_eq, true_output, output)
        except Exception as e:
            raise RuntimeError(
                f"FakeTensor cache crosscheck failure: func={func}, "
                f"args={args}, kwargs={kwargs}"
            ) from e
    # 定义一个方法dispatch，用于分发函数调用
    def dispatch(self, func, types, args=(), kwargs=None):
        # 如果kwargs为None，则将其设置为空字典
        kwargs = kwargs or {}
        # 使用no_dispatch上下文管理器，记录调试信息到日志中
        with no_dispatch():
            log.debug("%s %s %s", func, args, kwargs)

        # 如果func在_DISPATCH_META_HANDLERS中，返回其处理器处理的结果
        if func in _DISPATCH_META_HANDLERS:
            return _DISPATCH_META_HANDLERS[func](args)

        # 如果日志的有效级别为DEBUG，则记录假张量模式下的函数调用信息到日志中
        if log.getEffectiveLevel() <= logging.DEBUG:
            log.debug(
                "%sFakeTensorMode.__torch_dispatch__: %s", " " * RECURSION_COUNT, func
            )
            # 注意：为了RAII模式而意图未使用的incr变量

        # 对一些可以直接服务的属性查询进行处理
        # 参见注释 [is_coalesced is dispatched]
        if func in _DISPATCH_HANDLE_DIRECTLY:
            # 在这里也可以使用no_dispatch，这个函数非常简单
            with in_kernel_invocation_manager(self):
                return func(*args, **kwargs)

        # 如果启用了缓存，则调用_cached_dispatch_impl方法处理函数调用
        if self.cache_enabled:
            return self._cached_dispatch_impl(func, types, args, kwargs)
        else:
            # 否则调用_dispatch_impl方法处理函数调用
            return self._dispatch_impl(func, types, args, kwargs)

    # 警告：如果这里添加任何额外的命名空间或操作符，而它们引用的是pytorch/pytorch库之外的操作符！
    # 任何已存在于此处的东西要么在pytorch/pytorch库中，要么已经被允许使用。
    # 回退操作不总是有效，并且可能导致崩溃并输出不可读的错误消息，因此默认情况下不应允许。
    _can_run_unsafe_fallback_allowed_namespaces = ordered_set(
        "debugprims",
        "prims",
        "aten",
        "xla",
        "vision",
        "torchtext",
        "torchaudio",
        "quantized",
    )

    # 定义一个方法can_run_unsafe_fallback，用于确定是否可以执行不安全的回退
    def can_run_unsafe_fallback(self, func: OpOverload):
        # 如果不允许回退内核，则返回False
        if not self.allow_fallback_kernels:
            return False
        # 如果函数命名空间在允许的命名空间列表中，或者函数名为"fbgemm::gmm"，则返回True
        return (
            func.namespace in self._can_run_unsafe_fallback_allowed_namespaces
            or func.name() == "fbgemm::gmm"
        )

    # 定义一个方法validate_and_convert_non_fake_tensors，用于验证和转换非虚假张量
    def validate_and_convert_non_fake_tensors(
        self, func, converter, flat_args, args_spec
    ):
    ):
        """
        检查张量列表是否为虚假张量。
        如果不是虚假张量，则尝试将它们转换为虚假张量。
        返回原始参数、关键字参数，以及一个扁平化列表，其中包含虚假张量的 (参数, 关键字参数)。
        """
        flat_arg_fake_tensors: List[Any] = []

        def validate(x):
            # 如果 x 不是 torch.Tensor 类型，则直接返回
            if not isinstance(x, torch.Tensor):
                return x

            nonlocal flat_arg_fake_tensors
            # 如果 x 不是我们定义的虚假张量，则执行以下逻辑
            if not self.is_our_fake(x):
                # 如果函数具有 torch.Tag.inplace_view 标签
                if torch.Tag.inplace_view in func.tags:
                    # 使用 flat_args 和 args_spec 进行反扁平化，获取参数和关键字参数
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    # 抛出断言错误，指示不能在非虚假张量输入上调用元数据变异操作
                    raise AssertionError(
                        f"Can't call metadata mutating ops on non-Fake Tensor inputs. Found in {render_call(func, args, kwargs)}"
                    )
                # 如果不允许非虚假输入，并且 x 是 FakeTensor，但 fake_mode 不是当前实例
                if not self.allow_non_fake_inputs:
                    if isinstance(x, FakeTensor) and x.fake_mode is not self:
                        raise AssertionError("Mixing fake modes NYI")
                    # 使用 flat_args 和 args_spec 进行反扁平化，获取参数和关键字参数
                    args, kwargs = pytree.tree_unflatten(flat_args, args_spec)
                    # 抛出断言错误，指示请先将所有张量转换为 FakeTensors，或者使用 'allow_non_fake_inputs' 实例化 FakeTensorMode
                    raise AssertionError(
                        f"Please convert all Tensors to FakeTensors first or instantiate FakeTensorMode "
                        f"with 'allow_non_fake_inputs'. Found in {render_call(func, args, kwargs)}"
                    )

                # 将真实张量 x 转换为虚假张量
                x = converter.from_real_tensor(self, x)

            # 将虚假张量 x 添加到 flat_arg_fake_tensors 列表中
            flat_arg_fake_tensors.append(x)
            return x

        # 对 flat_args 中的每个参数执行 validate 函数，并返回验证后的参数列表
        validated_args = [validate(a) for a in flat_args]
        return validated_args, flat_arg_fake_tensors
    def wrap_meta_outputs_with_default_device_logic(self, r, func, flat_args, device):
        converter = self.fake_tensor_converter

        # Lazily initialized, in case there are no tensor returns
        common_device = None  # 共享设备初始化为空
        has_scalar_only_inputs = False  # 是否仅包含标量输入的标志初始化为False

        def wrap(e):
            nonlocal common_device  # 使用外部函数的共享设备变量
            nonlocal has_scalar_only_inputs  # 使用外部函数的标量输入标志变量

            if not isinstance(e, torch.Tensor):  # 如果不是 torch.Tensor 对象，直接返回
                return e

            if common_device is None:  # 如果共享设备还未初始化
                (
                    common_device,
                    has_scalar_only_inputs,
                ) = FakeTensor._find_common_device(func, flat_args)  # 调用函数找到共享设备和标量输入标志

            is_our_fake = self.is_our_fake(e)  # 判断是否为我们的伪造对象
            if is_our_fake:
                torch._check(
                    e.device == common_device,
                    lambda: f"FakeTensor is wrapped to wrong device, found {e.device}, expected {common_device}",
                )
                return e  # 如果是我们的伪造对象且设备匹配，则直接返回

            elif converter is not None:  # 如果有转换器可用
                if has_scalar_only_inputs:
                    # 在 FakeTensorMode 下，操作接受仅标量输入，例如 aten.add/sub/mul/div，在 CPU 上返回一个真实的标量张量。
                    # 详细信息请参阅 _prims/__init__.py 中的 TensorMeta()。
                    # 因此，我们直接将真实张量转换为伪造张量。
                    return converter.from_real_tensor(self, e)
                else:
                    return converter.from_meta_and_device(
                        self, e, device or common_device
                    )  # 否则，使用元信息和设备信息转换张量
            else:
                return e  # 如果没有转换器可用，直接返回输入张量

        return tree_map(wrap, r)  # 对输入的数据结构 r 应用 wrap 函数

    _cpp_meta_supports_symint = ordered_set(
        aten.empty.memory_format,  # 使用 ordered_set 初始化一组特定的函数名
        aten.empty_strided.default,
        aten.as_strided_scatter.default,
        aten.as_strided.default,
        aten.as_strided_.default,
        aten.zeros.default,
        aten.detach.default,
        aten.view_as_real.default,
        aten.view_as_complex.default,
        aten.set_.source_Storage_storage_offset,
        aten._sparse_coo_tensor_with_dims_and_tensors.default,
    )

    def cpp_meta_supports_symint(self, func):
        if torch.Tag.view_copy in func.tags:  # 如果 func 标记中包含 torch.Tag.view_copy
            return True  # 返回 True，表示支持符号整数
        return func in self._cpp_meta_supports_symint  # 否则，检查 func 是否在 _cpp_meta_supports_symint 中

    lift_fns = ordered_set(aten.lift_fresh.default, aten.lift_fresh_copy.default)

    def may_turn_const(self, t):
        return (
            t.numel() <= CONSTANT_NUMEL_LIMIT  # 张量元素数量小于等于常量数量限制
            and not t.is_sparse  # 并且不是稀疏张量
            and not self.is_our_fake(t)  # 并且不是我们的伪造对象
            and not t.device.type == "meta"  # 并且设备类型不是 "meta"
        )

    def invalidate_written_to_constants(
        self, func, flat_arg_fake_tensors, args, kwargs
        # 使写入到常量的值无效
    ):
        # 检查平坦化后的参数伪张量中是否有任何常量存在
        any_constant = any(e.constant is not None for e in flat_arg_fake_tensors)
        # 获取函数的模式信息
        schema_info = get_schema_info(func)
        # 如果存在任何常量且模式信息表明函数可变
        if any_constant and schema_info.is_mutable():
            # 标准化函数参数，仅使用关键字参数进行标准化
            _, new_kwargs = normalize_function(
                func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
            )
            # 遍历新的关键字参数
            for k, v in new_kwargs.items():
                # 如果关键字是 "input" 但模式信息不包含该参数，则将其改为 "self"
                k = k if (k != "input" or schema_info.has_argument(k)) else "self"
                # 如果参数是我们的伪张量且模式信息表明该参数可变且有常量值
                if (
                    self.is_our_fake(v)
                    and schema_info.is_mutable(k)
                    and v.constant is not None
                ):
                    # 使常量别名无效化
                    self.fake_tensor_converter.invalidate_constant_aliases(v.constant)

    # 将真实张量转换为伪张量
    def from_tensor(
        self,
        tensor,
        *,
        static_shapes=None,
        source: Optional[Source] = None,
        symbolic_context=None,
        trace=True,
    ):
        # 获取形状环境，默认为实例的形状环境
        shape_env: Optional[ShapeEnv] = self.shape_env
        # 如果未指定静态形状，则使用实例的静态形状
        if static_shapes is None:
            static_shapes = self.static_shapes
        # 如果指定了静态形状，则不能同时设置符号上下文
        if static_shapes:
            assert (
                symbolic_context is None
            ), "cannot set both static_shapes and symbolic_context"
            shape_env = None
        # 使用伪张量转换器将真实张量转换为伪张量
        return self.fake_tensor_converter.from_real_tensor(
            self,
            tensor,
            shape_env=shape_env,
            source=source,
            symbolic_context=symbolic_context,
            trace=trace,
        )
# NB: 返回虚假张量
def run_fallback_kernel(
    fake_mode, func, flat_args, args_spec, orig_not_implemented_exception
):
    # 这些都应该是支持的，为了安全起见
    # 避免对原地修改元数据的操作符进行回退
    # 因为输入的虚假张量将不会被修改
    if torch.Tag.inplace_view in func.tags:
        raise orig_not_implemented_exception

    # 用于保存输入实现的字典
    inp_impls = {}

    # 不要使用 in_kernel_invocation_manager(fake_mode)，因为我们希望执行真实的计算（不使用元设备）
    with no_dispatch():

        def to_real_tensor(e):
            if fake_mode.is_our_fake(e):
                # 创建与输入张量相同形状的零张量，使用输入张量的虚假设备
                out = torch.zeros_like(e, device=e.fake_device)
                if e.is_sparse:
                    out._coalesced_(e.is_coalesced())
                # 记录虚假张量和其对应的输入实现
                inp_impls[id(out)] = e
                return out
            return e

        # 将所有输入参数转换为真实张量
        flat_args = [to_real_tensor(a) for a in flat_args]
        # 使用给定的参数规范将扁平参数列表展开成参数和关键字参数
        args, kwargs = pytree.tree_unflatten(flat_args, args_spec)

        # 调用指定的函数
        r = func(*args, **kwargs)

    # 用于保存张量实现的集合
    tensor_impls = set()
    # 用于保存存储器的集合
    storages = set()

    # 遍历扁平参数列表
    for e in flat_args:
        if isinstance(e, torch.Tensor):
            if not e.is_sparse:
                # 将非稀疏张量的存储器添加到存储器集合中
                storages.add(e._typed_storage()._cdata)

    # TODO: 还需要检查输入的元数据变化
    # 由于转换为设备，输出和输入之间的正确别名/元数据关系
    # 将不会被设置好，除非我们可以重用一个输入实现

    def map_out(e):
        if id(e) not in inp_impls and (
            isinstance(e, torch.Tensor)
            and not e.is_sparse
            and e._typed_storage()._cdata in storages
        ):
            raise orig_not_implemented_exception

        if isinstance(e, torch.Tensor):
            if id(e) in inp_impls:
                return inp_impls[id(e)]
            else:
                # 将真实张量转换为虚假张量
                return fake_mode.fake_tensor_converter.from_real_tensor(fake_mode, e)
        else:
            return e

    # 将映射函数应用到结果 r 中的每个元素上
    return pytree.tree_map(map_out, r)


# 仅用于允许将模块复制到虚假张量中使用，不适用于其他地方
class FakeCopyMode(TorchFunctionMode):
    def __init__(self, fake_mode):
        self.fake_mode = fake_mode

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}

        # 在参数深拷贝中会调用 clone 方法
        if func == torch._C.TensorBase.clone:
            return func(
                self.fake_mode.from_tensor(args[0], static_shapes=True), **kwargs
            )
        elif func == torch.Tensor.__deepcopy__:
            assert len(args) == 2 and len(kwargs) == 0
            tensor, memo = args

            if id(tensor) in memo:
                return memo[id(tensor)]

            # 从真实张量创建虚假张量，并将其保存到 memo 中
            out = self.fake_mode.from_tensor(tensor, static_shapes=True)
            memo[id(tensor)] = out
            return out
        else:
            with torch._C.DisableTorchFunctionSubclass():
                return func(*args, **kwargs)


def _device_handler(args):
    # 确保参数只有一个且为 FakeTensor 类型
    assert len(args) == 1 and isinstance(args[0], FakeTensor)
    # 如果参数指定的 fake_mode 表示正在内核调用中，则返回一个元设备（meta device）
    if args[0].fake_mode.in_kernel_invocation:
        return torch.device("meta")
    else:
        # 否则返回参数指定的 fake_device
        return args[0].fake_device
# 定义一个函数 _check_for_subclass，用于检查给定参数列表中是否存在非 FakeTensor 类型但是是 torch.Tensor 的子类的情况。
# 返回 True 如果存在这样的参数，否则返回 False。
def _check_for_subclass(flat_args):
    return any(_check_for_subclass_arg(x) for x in flat_args)

# 定义一个函数 _check_for_subclass_arg，用于检查单个参数 x 是否满足以下条件：
# - 不是 FakeTensor 类型
# - 是 torch.Tensor 类型
# - 其类型不是 torch.Tensor
# - 其类型不是 torch.nn.Parameter
# 如果 x 满足上述条件，则返回 True，否则返回 False。
def _check_for_subclass_arg(x):
    return (
        not isinstance(x, FakeTensor)
        and isinstance(x, torch.Tensor)
        and type(x) is not torch.Tensor
        and type(x) is not torch.nn.Parameter
    )

# 定义一个字典 _DISPATCH_META_HANDLERS，将 torch.ops.prim.device.default 映射到 _device_handler 函数，
# 将 torch.ops.aten.size.default 映射到一个 lambda 表达式，该 lambda 表达式将 args[0].size() 转换为整数元组，
# 将 torch.ops.aten.stride.default 映射到一个 lambda 表达式，该 lambda 表达式将 args[0].stride() 转换为整数元组，
# 将 torch.ops.aten.storage_offset.default 映射到一个 lambda 表达式，该 lambda 表达式将 args[0].storage_offset() 转换为整数。
_DISPATCH_META_HANDLERS = {
    torch.ops.prim.device.default: _device_handler,
    torch.ops.aten.size.default: lambda args: tuple(int(s) for s in args[0].size()),
    torch.ops.aten.stride.default: lambda args: tuple(int(s) for s in args[0].stride()),
    torch.ops.aten.storage_offset.default: lambda args: int(args[0].storage_offset()),
}

# 定义一个有序集合 _DISPATCH_HANDLE_DIRECTLY，包含以下操作：
# - torch.ops.aten.is_coalesced.default
# - torch.ops.aten.dense_dim.default
# - torch.ops.aten.sparse_dim.default
# 这些操作需要直接处理。
_DISPATCH_HANDLE_DIRECTLY = ordered_set(
    torch.ops.aten.is_coalesced.default,
    torch.ops.aten.dense_dim.default,
    torch.ops.aten.sparse_dim.default,
)

# 导入 torch._subclasses.fake_impls 模块中的一些符号，并忽略 F401 警告。
from torch._subclasses.fake_impls import (
    _device_not_kwarg_ops,  # 导入 _device_not_kwarg_ops 符号
    _is_tensor_constructor,  # 导入 _is_tensor_constructor 符号
    _like_tensor_constructors,  # 导入 _like_tensor_constructors 符号
    contains_tensor_types,  # 导入 contains_tensor_types 符号
    get_fast_op_impls,  # 导入 get_fast_op_impls 符号
    has_meta,  # 导入 has_meta 符号
    op_implementations_checks,  # 导入 op_implementations_checks 符号
    stride_incorrect_op,  # 导入 stride_incorrect_op 符号
)
```