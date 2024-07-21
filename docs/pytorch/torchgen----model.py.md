# `.\pytorch\torchgen\model.py`

```
# 导入必要的模块和类
from __future__ import annotations

import dataclasses  # 用于定义数据类
import itertools  # 提供用于迭代工具的函数
import re  # 提供正则表达式的支持
from dataclasses import dataclass  # 导入数据类装饰器
from enum import auto, Enum  # 导入枚举相关的类和函数
from typing import Callable, Iterator, Sequence  # 提供类型提示的支持

from torchgen.utils import assert_never, NamespaceHelper, OrderedSet  # 导入自定义的工具函数和类


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           DATA MODEL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
# 一些关于数据模型的通用原则。
#
# - 停止使用 C++ 数据类型作为内部数据表示格式。相反，内部数据结构围绕 JIT 模式表示展开。
#   这避免了旧代码生成器中的一个大问题，即我们从 native_functions.yaml 中读取所有类型，然后立即必须重新将它们转换为 C++ 类型。
#
# - 更语义化的数据表示。不再将所有内容表示为字典和字符串，而是为代码生成必须处理的每个有趣实体定义数据类。
#   这些数据类具有强烈的语义不变性：例如，通常要求它们在解析时能够无损往返。
#   这些结构是不可变的，并且预期在构建期间仅填充信息一次。


# 表示源位置；用于更好的错误报告
@dataclass(frozen=True)
class Location:
    file: str  # 文件名
    line: int  # 行号

    def __str__(self) -> str:
        return f"{self.file}:{self.line}"  # 格式化输出位置信息


# native_functions.yaml 中 'variants' 字段的有效值
class Variant(Enum):
    function = auto()  # 函数类型
    method = auto()    # 方法类型


# 默认的核心命名空间
DEFAULT_KERNEL_NAMESPACE = "at::native"

# 注意：保持列表与 `c10/core/DispatchKey.h` 中的 `DispatchKey` 同步
BACKEND_COMPONENTS = "CPU CUDA HIP XLA MTIA MPS IPU XPU HPU VE Lazy Meta PrivateUse1 PrivateUse2 PrivateUse3".split()
FUNCTIONALITY_KEYS = [
    "",  # 空字符串
    "Quantized",  # 量化功能
    "Sparse",     # 稀疏功能
    "SparseCsr",  # CSR 稀疏功能
    "NestedTensor",  # 嵌套张量功能
    "Autograd",       # 自动微分功能
]

# derivatives.yaml 中可以用于派生的分发列表
# 现在我们省略了 AutogradFunctionality 和 AutogradOther
AUTOGRAD_KEYS = ["AutogradNestedTensor"] + [
    "Autograd" + component for component in BACKEND_COMPONENTS
]

FRAGMENT_NAMESPACES = {"quantized", "quantized_decomposed"}


# 这不必与头文件同步，它只需要包含我们在代码生成器中实际使用或希望有 pyi 条目的条目
class DispatchKey(Enum):
    Undefined = 0
    CatchAll = Undefined

    FPGA = auto()
    MAIA = auto()
    Vulkan = auto()
    Metal = auto()
    MKLDNN = auto()
    OpenGL = auto()
    OpenCL = auto()
    IDEEP = auto()
    CustomRNGKeyId = auto()
    MkldnnCPU = auto()
    Sparse = auto()
    SparseCsr = auto()
    NestedTensor = auto()
    Dense = auto()

    PythonTLSSnapshot = auto()
    PreDispatch = auto()
    PythonDispatcher = auto()
    Python = auto()
    FuncTorchDynamicLayerBackMode = auto()
    # 创建枚举成员 ZeroTensor
    ZeroTensor = auto()
    # 创建枚举成员 Conjugate
    Conjugate = auto()
    # 创建枚举成员 Negative
    Negative = auto()
    # 创建枚举成员 BackendSelect
    BackendSelect = auto()
    # 创建枚举成员 Named
    Named = auto()
    # 创建枚举成员 AutogradOther
    AutogradOther = auto()
    # 创建枚举成员 AutogradFunctionality
    AutogradFunctionality = auto()
    # 创建枚举成员 AutogradNestedTensor
    AutogradNestedTensor = auto()
    # 创建枚举成员 Tracer
    Tracer = auto()
    # 创建枚举成员 Autocast
    Autocast = auto()
    # 创建枚举成员 AutocastCPU
    AutocastCPU = auto()
    # 创建枚举成员 AutocastCUDA
    AutocastCUDA = auto()
    # 创建枚举成员 Batched
    Batched = auto()
    # 创建枚举成员 VmapMode
    VmapMode = auto()
    # 创建枚举成员 FuncTorchGradWrapper
    FuncTorchGradWrapper = auto()
    # 创建枚举成员 FuncTorchBatched
    FuncTorchBatched = auto()
    # 创建枚举成员 BatchedNestedTensor
    BatchedNestedTensor = auto()
    # 创建枚举成员 FuncTorchVmapMode
    FuncTorchVmapMode = auto()
    # 创建枚举成员 FuncTorchDynamicLayerFrontMode
    FuncTorchDynamicLayerFrontMode = auto()
    # 创建枚举成员 Functionalize
    Functionalize = auto()
    # 创建枚举成员 TESTING_ONLY_GenericWrapper
    TESTING_ONLY_GenericWrapper = auto()
    # 创建枚举成员 TESTING_ONLY_GenericMode
    TESTING_ONLY_GenericMode = auto()

    # 创建枚举成员 ADInplaceOrView
    ADInplaceOrView = auto()
    # 创建枚举成员 Autograd
    Autograd = auto()
    # 创建枚举成员 CompositeImplicitAutograd
    CompositeImplicitAutograd = auto()
    # 创建枚举成员 CompositeImplicitAutogradNestedTensor
    CompositeImplicitAutogradNestedTensor = auto()
    # 创建枚举成员 CompositeExplicitAutograd
    CompositeExplicitAutograd = auto()
    # 创建枚举成员 CompositeExplicitAutogradNonFunctional
    CompositeExplicitAutogradNonFunctional = auto()
    # 创建枚举成员 FuncTorchBatchedDecomposition
    FuncTorchBatchedDecomposition = auto()

    # BEGIN 自动生成的枚举成员
    # 创建枚举成员 CPU
    CPU = auto()
    # 创建枚举成员 CUDA
    CUDA = auto()
    # 创建枚举成员 HIP
    HIP = auto()
    # 创建枚举成员 XLA
    XLA = auto()
    # 创建枚举成员 MTIA
    MTIA = auto()
    # 创建枚举成员 MPS
    MPS = auto()
    # 创建枚举成员 IPU
    IPU = auto()
    # 创建枚举成员 XPU
    XPU = auto()
    # 创建枚举成员 HPU
    HPU = auto()
    # 创建枚举成员 VE
    VE = auto()
    # 创建枚举成员 Lazy
    Lazy = auto()
    # 创建枚举成员 Meta
    Meta = auto()
    # 创建枚举成员 PrivateUse1
    PrivateUse1 = auto()
    # 创建枚举成员 PrivateUse2
    PrivateUse2 = auto()
    # 创建枚举成员 PrivateUse3
    PrivateUse3 = auto()
    # 创建枚举成员 QuantizedCPU
    QuantizedCPU = auto()
    # 创建枚举成员 QuantizedCUDA
    QuantizedCUDA = auto()
    # 创建枚举成员 QuantizedHIP
    QuantizedHIP = auto()
    # 创建枚举成员 QuantizedXLA
    QuantizedXLA = auto()
    # 创建枚举成员 QuantizedMTIA
    QuantizedMTIA = auto()
    # 创建枚举成员 QuantizedMPS
    QuantizedMPS = auto()
    # 创建枚举成员 QuantizedIPU
    QuantizedIPU = auto()
    # 创建枚举成员 QuantizedXPU
    QuantizedXPU = auto()
    # 创建枚举成员 QuantizedHPU
    QuantizedHPU = auto()
    # 创建枚举成员 QuantizedVE
    QuantizedVE = auto()
    # 创建枚举成员 QuantizedLazy
    QuantizedLazy = auto()
    # 创建枚举成员 QuantizedMeta
    QuantizedMeta = auto()
    # 创建枚举成员 QuantizedPrivateUse1
    QuantizedPrivateUse1 = auto()
    # 创建枚举成员 QuantizedPrivateUse2
    QuantizedPrivateUse2 = auto()
    # 创建枚举成员 QuantizedPrivateUse3
    QuantizedPrivateUse3 = auto()
    # 创建枚举成员 SparseCPU
    SparseCPU = auto()
    # 创建枚举成员 SparseCUDA
    SparseCUDA = auto()
    # 创建枚举成员 SparseHIP
    SparseHIP = auto()
    # 创建枚举成员 SparseXLA
    SparseXLA = auto()
    # 创建枚举成员 SparseMTIA
    SparseMTIA = auto()
    # 创建枚举成员 SparseMPS
    SparseMPS = auto()
    # 创建枚举成员 SparseIPU
    SparseIPU = auto()
    # 创建枚举成员 SparseXPU
    SparseXPU = auto()
    # 创建枚举成员 SparseHPU
    SparseHPU = auto()
    # 创建枚举成员 SparseVE
    SparseVE = auto()
    # 创建枚举成员 SparseLazy
    SparseLazy = auto()
    # 创建枚举成员 SparseMeta
    SparseMeta = auto()
    # 创建枚举成员 SparsePrivateUse1
    SparsePrivateUse1 = auto()
    # 创建枚举成员 SparsePrivateUse2
    SparsePrivateUse2 = auto()
    # 创建枚举成员 SparsePrivateUse3
    SparsePrivateUse3 = auto()
    # 创建枚举成员 SparseCsrCPU
    SparseCsrCPU = auto()
    # 创建枚举成员 SparseCsrCUDA
    SparseCsrCUDA = auto()
    # 创建枚举成员 SparseCsrHIP
    SparseCsrHIP = auto()
    # 创建枚举成员 SparseCsrXLA
    SparseCsrXLA = auto()
    # 创建枚举成员 SparseCsrMTIA
    SparseCsrMTIA = auto()
    # 创建枚举成员 SparseCsrMPS
    SparseCsrMPS = auto()
    # 创建枚举成员 SparseCsrIPU
    SparseCsrIPU = auto()
    # 创建枚举成员 SparseCsrXPU
    SparseCsrXPU = auto()
    # 创建枚举成员 SparseCsrHPU
    SparseCsrHPU = auto()
    # 创建枚举成员 SparseCsrVE
    SparseCsrVE = auto()
    # 创建枚举成员 SparseCsrLazy
    SparseCsrLazy = auto()
    # 创建枚举成员 SparseCsrMeta
    SparseCsrMeta = auto()
    # 创建枚举成员 SparseCsrPrivateUse1
    SparseCsrPrivateUse1 = auto()
    # 创建枚举成员 SparseCsrPrivateUse2
    SparseCsrPrivateUse2 = auto()
    # 创建枚举成员 SparseCsrPrivateUse3
    SparseCsrPrivateUse3 = auto()
    # 创建枚举成员 NestedTensorCPU
    NestedTensorCPU = auto()
    # 创建枚举成员 NestedTensorCUDA
    NestedTensorCUDA = auto()
    # 创建枚举成员 NestedTensorHIP
    NestedTensorHIP = auto()
    # 创建枚举成员 NestedTensorXLA
    NestedTensorXLA = auto()
    # 创建枚举成员 NestedTensorMTIA
    NestedTensorMTIA = auto()
    # 创建枚举成员 NestedTensorMPS
    NestedTensorMPS = auto()
    # 创建枚举成员 NestedTensorIPU
    NestedTensorIPU = auto()
    # 创建枚举成员 NestedTensorXPU
    NestedTensorXPU = auto()
    #
    AutogradMTIA = auto()
    AutogradMPS = auto()
    AutogradIPU = auto()
    AutogradXPU = auto()
    AutogradHPU = auto()
    AutogradVE = auto()
    AutogradLazy = auto()
    AutogradMeta = auto()
    AutogradPrivateUse1 = auto()
    AutogradPrivateUse2 = auto()
    AutogradPrivateUse3 = auto()
    # END autogenerated

    # 返回枚举成员的自动递增值，用于自动生成唯一的标识符
    def __str__(self) -> str:
        # 返回枚举成员的名称作为字符串表示
        return self.name

    # 返回枚举成员的小写名称字符串表示
    def lower(self) -> str:
        return str(self).lower()

    # 根据给定的字符串值解析并返回对应的 DispatchKey 枚举成员
    @staticmethod
    def parse(value: str) -> DispatchKey:
        for k, v in DispatchKey.__members__.items():
            if k == value:
                return v
        # 如果未找到匹配的枚举成员，则引发断言错误
        raise AssertionError(f"unknown dispatch key {value}")
# 枚举类型 `_TorchDispatchModeKey` 定义了几个常量，用于表示不同的调度模式
class _TorchDispatchModeKey(Enum):
    FAKE = auto()       # 自动分配下一个可用值给 FAKE
    PROXY = auto()      # 自动分配下一个可用值给 PROXY
    FUNCTIONAL = auto() # 自动分配下一个可用值给 FUNCTIONAL


# 生成每个后端条目的代码，返回字符串形式的结果
def codegen_per_backend_entries() -> str:
    r = []  # 创建一个空列表用于存储生成的条目
    for fk in FUNCTIONALITY_KEYS:
        for bc in BACKEND_COMPONENTS:
            r.append(f"    {fk}{bc} = auto()")  # 将每个生成的条目添加到列表中
    return "\n".join(r)  # 返回所有条目的字符串形式，每行一个


# 检查是否所有的功能键和后端组件在 DispatchKey 枚举中都存在，如果不是则抛出运行时错误
for fk in FUNCTIONALITY_KEYS:
    for bc in BACKEND_COMPONENTS:
        if not hasattr(DispatchKey, fk + bc):
            r = codegen_per_backend_entries()  # 生成条目
            print(r)  # 打印生成的条目列表
            raise RuntimeError(
                f"Missing {fk}{bc} from DispatchKey enum.  Here is the autogenerated list we expect to have:\n\n{r}"
            )  # 抛出运行时错误，说明缺失的条目


# 定义了几个 DispatchKey 的集合，分别表示结构化调度和 ufunc 调度所支持的调度键
STRUCTURED_DISPATCH_KEYS = {DispatchKey.MPS, DispatchKey.CUDA, DispatchKey.CPU}
UFUNC_DISPATCH_KEYS = {DispatchKey.CUDA, DispatchKey.CPU}


# 定义了一个包含所有支持的调度键的列表
dispatch_keys = [
    DispatchKey.CPU,
    DispatchKey.SparseCPU,
    DispatchKey.SparseCsrCPU,
    DispatchKey.MkldnnCPU,
    DispatchKey.CUDA,
    DispatchKey.MPS,
    DispatchKey.SparseCUDA,
    DispatchKey.SparseCsrCUDA,
    DispatchKey.QuantizedCPU,
    DispatchKey.QuantizedCUDA,
    DispatchKey.CompositeImplicitAutograd,
    DispatchKey.CompositeImplicitAutogradNestedTensor,
    DispatchKey.CompositeExplicitAutograd,
    DispatchKey.CompositeExplicitAutogradNonFunctional,
    DispatchKey.NestedTensorCPU,
    DispatchKey.NestedTensorCUDA,
    DispatchKey.Meta,
    DispatchKey.SparseMeta,
    DispatchKey.SparseCsrMeta,
    DispatchKey.QuantizedMeta,
    DispatchKey.NestedTensorMeta,
    DispatchKey.ZeroTensor,
]


# 判断给定的调度键是否属于 "支持所有后端" 的调度键
def is_generic_dispatch_key(dk: DispatchKey) -> bool:
    return dk in {
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeImplicitAutogradNestedTensor,
    }


# 判断给定的调度键是否属于 CUDA 特定的调度键
def is_cuda_dispatch_key(dk: DispatchKey) -> bool:
    return dk in {
        DispatchKey.CUDA,
        DispatchKey.QuantizedCUDA,
        DispatchKey.SparseCUDA,
        DispatchKey.SparseCsrCUDA,
        DispatchKey.NestedTensorCUDA,
        DispatchKey.AutogradCUDA,
    }


# 判断给定的调度键是否属于结构化调度键
def is_structured_dispatch_key(dk: DispatchKey) -> bool:
    return dk in STRUCTURED_DISPATCH_KEYS


# 判断给定的调度键是否属于 ufunc 调度键
def is_ufunc_dispatch_key(dk: DispatchKey) -> bool:
    return dk in UFUNC_DISPATCH_KEYS


# 定义了枚举类型 ScalarType，表示不同的标量类型
class ScalarType(Enum):
    Byte = auto()
    Char = auto()
    Short = auto()
    Int = auto()
    Long = auto()
    Half = auto()
    Float = auto()
    Double = auto()
    ComplexHalf = auto()
    ComplexFloat = auto()
    # 自动分配枚举值 ComplexDouble
    ComplexDouble = auto()
    # 自动分配枚举值 Bool
    Bool = auto()
    # 自动分配枚举值 BFloat16
    BFloat16 = auto()
    # 自动分配枚举值 Float8_e5m2
    Float8_e5m2 = auto()
    # 自动分配枚举值 Float8_e5m2fnuz
    Float8_e5m2fnuz = auto()
    # 自动分配枚举值 Float8_e4m3fn
    Float8_e4m3fn = auto()
    # 自动分配枚举值 Float8_e4m3fnuz
    Float8_e4m3fnuz = auto()

    # 返回枚举成员的字符串表示
    def __str__(self) -> str:
        return self.name

    # 尝试解析字符串值为对应的 ScalarType 枚举成员，如果找不到则返回 None
    @staticmethod
    def maybe_parse(value: str) -> ScalarType | None:
        for k, v in ScalarType.__members__.items():
            if k == value:
                return v
        return None

    # 解析字符串值为对应的 ScalarType 枚举成员，如果找不到则引发异常
    @staticmethod
    def parse(value: str) -> ScalarType:
        mb_r = ScalarType.maybe_parse(value)
        assert mb_r is not None, f"unknown dtype {value}"
        return mb_r

    # 解析由逗号和空格分隔的多个字符串值集合，返回有序集合 OrderedSet[ScalarType]
    @staticmethod
    def parse_set(values: str) -> OrderedSet[ScalarType]:
        dtypes: OrderedSet[ScalarType] = OrderedSet()
        for value in values.split(", "):
            if value in DTYPE_CLASSES:
                dtypes.update(DTYPE_CLASSES[value])
            else:
                dtypes.add(ScalarType.parse(value))
        return dtypes
# 定义一个空的字典，键是字符串，值是有序集合OrderedSet，存储标量类型
DTYPE_CLASSES: dict[str, OrderedSet[ScalarType]] = {}

# NB: Integral 不包括布尔类型
# 将标量类型ScalarType.Byte, ScalarType.Char, ScalarType.Int,
# ScalarType.Long, ScalarType.Short按顺序添加到"DTYPE_CLASSES"字典中的"Integral"键下的有序集合OrderedSet中
DTYPE_CLASSES["Integral"] = OrderedSet(
    [
        ScalarType.Byte,
        ScalarType.Char,
        ScalarType.Int,
        ScalarType.Long,
        ScalarType.Short,
    ]
)

# NB: Floating 不包括低精度类型
# 将标量类型ScalarType.Float, ScalarType.Double按顺序添加到"DTYPE_CLASSES"字典中的
# "Floating"键下的有序集合OrderedSet中
DTYPE_CLASSES["Floating"] = OrderedSet([ScalarType.Float, ScalarType.Double])

# 将标量类型ScalarType.ComplexFloat, ScalarType.ComplexDouble按顺序添加到
# "DTYPE_CLASSES"字典中的"Complex"键下的有序集合OrderedSet中
DTYPE_CLASSES["Complex"] = OrderedSet(
    [ScalarType.ComplexFloat, ScalarType.ComplexDouble]
)

# 将"DTYPE_CLASSES"字典中"Integral"和"Floating"键的有序集合合并，并赋值给"All"键
DTYPE_CLASSES["All"] = DTYPE_CLASSES["Integral"] | DTYPE_CLASSES["Floating"]

# 将"DTYPE_CLASSES"字典中"All"和"Complex"键的有序集合合并，并赋值给"AllAndComplex"键
DTYPE_CLASSES["AllAndComplex"] = DTYPE_CLASSES["All"] | DTYPE_CLASSES["Complex"]

# 将"DTYPE_CLASSES"字典中"Floating"和"Complex"键的有序集合合并，并赋值给"FloatingAndComplex"键
DTYPE_CLASSES["FloatingAndComplex"] = (
    DTYPE_CLASSES["Floating"] | DTYPE_CLASSES["Complex"]
)


# 表示native_functions.yaml中ufunc_inner_loop的有效条目
# NB: 如果添加一个新的UfuncKey，需要教导torchgen.dest.ufunc如何处理它。
# 大多数逻辑将忽略它们不理解的键，因此新的键将被静默忽略，直到您连接处理它的逻辑。
class UfuncKey(Enum):
    # 这些是低级键，表示代码生成产生的一个特定内核的实例
    CUDAFunctor = auto()
    CUDAFunctorOnOther = auto()
    CUDAFunctorOnSelf = auto()

    CPUScalar = auto()
    CPUVector = auto()

    # 这些是用户通常会指定的键，并隐式“填充”低级键
    ScalarOnly = auto()  # CUDA*, CPUScalar
    Generic = auto()  # CUDA*, CPU*

    def __str__(self) -> str:
        return self.name

    @staticmethod
    def parse(value: str) -> UfuncKey:
        # 解析字符串值并返回对应的UfuncKey枚举值
        for k, v in UfuncKey.__members__.items():
            if k == value:
                return v
        # 如果未找到对应的枚举值，则引发断言错误
        raise AssertionError(f"unknown ufunc key {value}")


class DeviceCheckType(Enum):
    # 设备检查类型枚举，表示不进行检查和进行精确相同检查
    NoCheck = 0
    ExactSame = 1


class ViewSchemaKind(Enum):
    # 视图模式枚举，表示别名、原地别名和非别名
    aliasing = auto()
    aliasing_inplace = auto()
    non_aliasing = auto()


# 代码生成的基本输入是native_functions.yaml。
# 名称“native”来自于原生函数与遗留TH函数的区别。虽然遗留TH函数已经不存在，
# 但“native”描述符仍然保留。
#
# NativeFunction表示native_functions.yaml中的单个条目。它的字段大致对应于您在YAML中看到的内容，
# 但经过规范化和解析后的版本。
#
# 您可以在这个类中看到我们如何设置数据类的一些整体设计模式，
# 但我们将在FunctionSchema中推迟对其的完整讨论。
@dataclass(frozen=True)
class NativeFunction:
    # 这个操作符的命名空间。例如，如果我们有"at::add"，那么命名空间将是"at"。
    # 这使得可以使用自定义命名空间通过相同的DSL注册操作符。
    # 如果未指定，默认命名空间将是"at"。
    namespace: str

    # 操作符的函数模式。此模式
    # 已解析；有关其结构的更多信息，请参阅FunctionSchema。
    # （此类型被引用，因为我们正在向前引用文件中稍后定义的类型。
    # 我选择了这种类的顺序以提高表达清晰度。）
    func: FunctionSchema

    # 是否生成可变张量参数的可变引用像普通张量一样
    use_const_ref_for_mutable_tensors: bool

    # 是否省略自动生成DeviceGuard
    device_guard: bool

    # 如何生成设备检查的自动生成
    device_check: DeviceCheckType

    # 将函数放入的Python模块
    python_module: str | None

    # TODO: 弄清楚这个是做什么的
    category_override: str | None

    # 如果native_functions.yaml中未指定任何变体，则假定为{'function'}。
    variants: set[Variant]

    # 是否应跳过为此内核生成注册
    # 这有点两难，因为手动注册不参与基于代码生成的选择性构建！
    manual_kernel_registration: bool

    # 是否跳过为此内核生成TensorMethod/Functions绑定
    # 技术上，这实际上并不跳过生成绑定；相反，绑定会生成到__dispatch_{funcname}
    # 因此，如果需要，您可以使用正常绑定。
    manual_cpp_binding: bool

    # 定义本地函数条目所在的YAML文件中的位置。
    # 这是为了方便报告错误消息！
    loc: Location

    # 预期为此NativeFunction自动生成的运算符列表。
    # 注意：此列表实际上不会直接由代码生成用于生成任何内容。
    # 相反，代码生成器仅基于函数模式确定要生成的运算符，并使用autogen声明进行错误检查。
    # 我们期望为每个自动生成的NativeFunction在native_functions.yaml中显式调用。
    autogen: list[OperatorName]

    # 如果非空，则此内核将进行ufunc代码生成。
    # 按ufunc_key排序
    ufunc_inner_loop: dict[UfuncKey, UfuncInnerLoop]

    # 是否此输出函数为“结构化内核”。
    # 结构化内核与普通内核定义有所不同；特别是，它们的形状检查逻辑与内核分开定义。
    # 只有输出函数可以是结构化的；其他函数使用structured_delegate关键字委派到输出函数。
    # 每个结构化内核必须至少有一个输出和一个功能变体。
    structured: bool

    # 是否此非输出函数为结构化内核，以此处引用的输出内核定义。
    structured_delegate: OperatorName | None

    # 仅适用于结构化内核。指定替代方案为何
    # to inherit from when defining the meta class for the structured
    # operator.  This will usually be TensorIteratorBase.  This also
    # changes the semantics of set_output to call the parent class.
    structured_inherits: str | None

    # Structured kernels can declare elements as "precomputed". These elements
    # are returned by the meta function in one struct and passed to the impl
    # function in lieu of certain kernel arguments that these precomputed
    # elements supersede. Information about the names and types of these
    # precomputed elements and how they correspond to kernel arguments is stored
    # in this member, if applicable.
    precomputed: Precompute | None

    # Argument names whose default  should be excluded from the C++ interface.
    # Intended for resolving overload ambiguities between signatures.
    cpp_no_default_args: set[str]

    # Note [Abstract ATen methods]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # An abstract ATen method is one whose dispatch differs between
    # types.  These are implemented in derived types (with a
    # standard (throwing) definition in Type).  A concrete ATen
    # method is one which has the same dispatch for all types;
    # we just implement it in the base Type.  This is exposed
    # in Declarations.yaml via a field named 'abstract'.
    is_abstract: bool

    # Whether or not the NativeFunction contains a backend-agnostic kernel
    has_composite_implicit_autograd_kernel: bool
    has_composite_implicit_autograd_nested_tensor_kernel: bool
    has_composite_explicit_autograd_kernel: bool
    has_composite_explicit_autograd_non_functional_kernel: bool

    # Tags are used to describe semantic information about (groups of) operators,
    # That aren't easily inferrable directly from the operator's schema.
    tags: set[str]

    # NB: The benefit of defining a dataclass is that we automatically get
    # a constructor defined for all the fields we specify.  No need
    # to explicitly write it out.

    # We parse both the NativeFunction + backend-specific information about it, which it stored in a corresponding BackendIndex.
    @staticmethod
    def from_yaml(
        ei: dict[str, object],
        loc: Location,
        valid_tags: set[str],
        ignore_keys: set[DispatchKey] | None = None,
    # Define a method for creating an instance of this class from YAML data, handling backend-specific details and location information.

    def validate_unstructured(self) -> None:
        # TODO: probably better to accumulate these errors and report them all
        # at once
        assert not self.structured, (
            "This function is structured, but there was "
            "no valid functional variant of it."
        )
        assert self.structured_delegate, (
            "This function delegates to another structured out function, "
            "but no valid function was found (the delegate may not exist, or it has the wrong type)"
        )
    # Define a validation method to ensure that the function is correctly structured and delegates properly to another function.

    # __post_init__ functions in dataclasses can be used to do extra
    # validation after construction.
    #
    # 注意这里我们没有进行任何类型验证。事实上，我们完全依赖于 mypy 来检查你是否正确地使用了类型！
    # 验证是针对那些不能（方便地）在类型系统中编码的非平凡不变量的。
    # 在对象初始化后执行的方法，用于进行额外的初始化操作
    def __post_init__(self) -> None:
        # 如果函数具有输出参数
        if self.func.arguments.out:
            # 断言只有一个函数变体是 function，用于避免 Python 参数绑定错误
            assert self.variants == {Variant.function}, (
                "Native functions with out arguments MUST "
                "be declared with only function variant; e.g., variants: function; "
                "otherwise you will tickle a Python argument binding bug "
                "(which usually manifests itself as the result variable being undefined.)"
            )
        # 如果是结构化数据
        if self.structured:
            # 断言函数属于 out= 变体
            assert self.func.kind() == SchemaKind.out, (
                "Put structured field on the out= "
                "variant of a function; did you mean structured_delegate?"
            )
            # 断言设备保护标志为真
            assert (
                self.device_guard
            ), "device_guard: False is not respected by structured kernels"
        # 如果是结构化数据委托
        if self.structured_delegate:
            # 断言函数不属于 out= 变体
            assert self.func.kind() != SchemaKind.out, (
                "structured_delegate field not allowed "
                "on out= functions; did you mean structured?"
            )
            # 断言设备保护标志为真
            assert (
                self.device_guard
            ), "device_guard: False is not respected by structured kernels"
        # 断言结构化数据和结构化数据委托不能同时存在
        assert not (
            self.structured and self.structured_delegate
        ), "Cannot have both structured and structured_delegate on function"
        # 获取默认参数的集合
        defaulted_arguments = {
            a.name for a in self.func.schema_order_arguments() if a.default is not None
        }
        # 找出不合法的 cpp_no_default_args
        invalid_args = set.difference(self.cpp_no_default_args, defaulted_arguments)
        # 断言没有不合法的 cpp_no_default_args
        assert len(invalid_args) == 0, f"Invalid cpp_no_default_args: {invalid_args}"
        # 如果结构化数据继承属性不为空
        if self.structured_inherits is not None:
            # 断言结构化数据必须为 True
            assert (
                self.structured
            ), "structured_inherits must also imply structured: True"
        # 如果函数名以 "_foreach" 开头
        if str(self.func.name).startswith("_foreach"):
            # 断言设备检查类型为 NoCheck，因为 foreach 核心在张量位于不同设备时会回退到慢路径
            assert self.device_check == DeviceCheckType.NoCheck, (
                "foreach kernels fall back to slow path when tensor are on different devices, "
                "device_check not allowed to be enabled"
            )

        # 注意：如果函数名中意外包含了 "rand"、"dropout" 等标记为随机性的词汇，
        # 但实际上函数不是随机的，可以根据需要修改此处逻辑
        if (
            "rand" in str(self.func.name)
            or (
                (
                    "dropout" in str(self.func.name)
                    or any(
                        "dropout" in arg.name for arg in self.func.arguments.flat_all
                    )
                )
                # dropout 的反向传播通常是确定性的
                and "backward" not in str(self.func.name)
                and str(self.func.name.name) not in ["_cudnn_init_dropout_state"]
            )
            # 如果函数的参数包含生成器参数
            or self.func.arguments.has_generator_arg()
        ):
            # 断言 "nondeterministic_seeded" 在函数的标签中
            assert "nondeterministic_seeded" in self.tags, str(self.func.name)
    # 检查是否存在复合内核
    @property
    def has_composite_kernel(self) -> bool:
        return (
            self.has_composite_implicit_autograd_kernel
            or self.has_composite_explicit_autograd_kernel
            or self.has_composite_explicit_autograd_non_functional_kernel
        ) or (
            self.has_composite_implicit_autograd_kernel
            and self.has_composite_implicit_autograd_nested_tensor_kernel
        )

    # 检查是否为视图操作
    @property
    def is_view_op(self) -> bool:
        rets = self.func.returns
        # 检查是否为非变异视图
        is_non_mutating_view = len(rets) > 0 and any(
            r.annotation is not None and not r.annotation.is_write for r in rets
        )
        # 检查是否为内存中的视图
        # 参见“Functionalization”中的“resize_”注释以获取更多详细信息
        is_inplace_view = (
            "inplace_view" in self.tags
            and str(self.func.name) != "resize_"
            and str(self.func.name) != "resize_as_"
        )
        # 检查是否为通配符视图
        is_wildcard_view = any(
            inp.annotation is not None and "*" in inp.annotation.alias_set_after
            for inp in self.func.schema_order_arguments()
        )
        return is_non_mutating_view or is_inplace_view or is_wildcard_view

    # 返回视图模式类型
    @property
    def view_schema_kind(self) -> ViewSchemaKind:
        if self.is_view_op and self.func.name.name.inplace:
            assert "inplace_view" in self.tags
            return ViewSchemaKind.aliasing_inplace
        if self.is_view_op:
            return ViewSchemaKind.aliasing
        else:
            return ViewSchemaKind.non_aliasing

    # 返回根操作名
    @property
    def root_name(self) -> str:
        return self.func.name.name.base

    # 检查是否为结构化组的一部分
    @property
    def part_of_structured_group(self) -> bool:
        return self.structured or self.structured_delegate is not None
class SchemaKind(Enum):
    # 定义枚举类型，包含不同的模式：functional、inplace、out、mutable、scratch
    functional = auto()
    inplace = auto()
    out = auto()
    mutable = auto()
    scratch = auto()


# 一个结构化的内核组，保证有 functional 和 out 变体，并且可选地有 inplace 变体。
#
# 注意：即使函数没有被标记为结构化，我们也会创建 NativeFunctionsGroup。通过测试 structured 变量可以确定它是否实际上是结构化的。
@dataclass(frozen=True)
class NativeFunctionsGroup:
    functional: NativeFunction  # functional 类型的 NativeFunction
    inplace: NativeFunction | None  # 可能为 None 的 inplace 类型的 NativeFunction
    mutable: NativeFunction | None  # 可能为 None 的 mutable 类型的 NativeFunction
    out: NativeFunction  # out 类型的 NativeFunction

    @property
    def structured(self) -> bool:
        # 返回操作符是否具有 meta() 函数的信息，这个信息与后端无关。
        return self.out.structured

    def signature(self) -> FunctionSchema:
        # 返回 out 变体的函数签名
        return self.out.func.signature()

    def functions(self) -> Iterator[NativeFunction]:
        # 生成器函数，依次返回 functional、out 变体，如果 inplace 和 mutable 不为 None，也返回它们
        yield self.functional
        yield self.out
        if self.inplace is not None:
            yield self.inplace
        if self.mutable is not None:
            yield self.mutable

    @property
    def root_name(self) -> str:
        # 返回 functional 变体的 root_name
        return self.functional.root_name

    @staticmethod
    def from_dict(d: dict[SchemaKind, NativeFunction]) -> NativeFunctionsGroup | None:
        # 断言字典 d 不为空
        assert d
        # 如果字典长度为 1，则返回 None
        if len(d) == 1:
            return None
        d = dict(d)  # 非破坏性地更新字典
        functional = d.pop(SchemaKind.functional, None)
        inplace = d.pop(SchemaKind.inplace, None)
        mutable = d.pop(SchemaKind.mutable, None)
        out = d.pop(SchemaKind.out, None)
        # 断言字典为空
        assert not d
        # 断言 functional 不为 None
        assert functional is not None
        # 如果 out 是 None，则返回 None
        if out is None:
            return None
        # 假设所有变体都有相同的命名空间，返回 NativeFunctionsGroup 对象
        return NativeFunctionsGroup(
            functional=functional,
            inplace=inplace,
            mutable=mutable,
            out=out,
        )


@dataclass(frozen=True)
class BackendMetadata:
    # 后端内核的名称，针对给定的操作符
    # 对于内核树后端，这些名称直接来自 native_functions.yaml 中的 'dispatch' 字段。
    # dispatch 条目是可选的；在这种情况下，相当于写了：
    #
    #   dispatch:
    #       CompositeImplicitAutograd: $operator_name
    kernel: str
    # 操作符是否在该特定后端上实现了结构化内核
    # 对于内核树后端，它们在 structured 上有相同的值- 这在 native_functions.yaml 中列出。
    # 然而，像 XLA 这样的外部后端可以独立地切换哪些操作是结构化的。
    structured: bool

    # 内核的命名空间，默认值为 DEFAULT_KERNEL_NAMESPACE
    cpp_namespace: str
    # 检查对象的内核属性是否支持符号整数计算
    def supports_symint(self) -> bool:
        # 返回一个布尔值，指示对象的内核属性中是否包含字符串 "_symint"
        return "_symint" in self.kernel
# 使用 `dataclass` 装饰器定义了一个名为 `UfuncInnerLoop` 的数据类，表示一种通用函数的内部循环。
@dataclass(frozen=True)
class UfuncInnerLoop:
    # 内部循环的名称，为字符串类型
    name: str
    # 支持的数据类型集合，使用有序集合 `OrderedSet` 存储 `ScalarType` 类型
    supported_dtypes: OrderedSet[ScalarType]
    # `ufunc_key` 影响 `name` 语义的关键信息，与 `name` 一起存储以便进一步处理

    # 解析静态方法，用于从字符串 `value` 解析并创建 `UfuncInnerLoop` 实例
    @staticmethod
    def parse(value: str, ufunc_key: UfuncKey) -> UfuncInnerLoop:
        # 根据空格拆分 `value` 获取 `name` 和 `supported_dtypes_str`
        name, supported_dtypes_str = value.split(" ", 1)
        # 断言确保 `supported_dtypes_str` 符合预期格式
        assert supported_dtypes_str[0] == "("
        assert supported_dtypes_str[-1] == ")"
        # 使用有序集合 `OrderedSet` 存储支持的数据类型
        supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
        # 遍历拆分 `supported_dtypes_str` 中的每个元素并解析为 `ScalarType`，加入到集合中
        for k in supported_dtypes_str[1:-1].split(", "):
            supported_dtypes |= ScalarType.parse_set(k)
        # 返回一个新的 `UfuncInnerLoop` 实例
        return UfuncInnerLoop(
            name=name, supported_dtypes=supported_dtypes, ufunc_key=ufunc_key
        )


# `BackendIndex` 类表示一个后端索引，编码了每个操作符在不同后端上的信息。
# 这些信息可能因后端不同而异，例如内核名称（在 `native_functions.yaml` 中的 'dispatch' 项）。
@dataclass(frozen=True)
class BackendIndex:
    # 用于区分不同操作符信息的调度键
    dispatch_key: DispatchKey
    # 对于结构化内核非常重要，决定了使用哪个变体作为主实现。
    use_out_as_primary: bool
    # 后端是否需要设备保护和设备检查。
    device_guard: bool
    # 后端是否为外部（非官方支持），如 XLA
    external: bool
    # 每个操作符名称与后端元数据之间的映射，使用字典存储
    index: dict[OperatorName, BackendMetadata]

    # 将子索引 `child_index` 合并到父索引 `parent_index` 中的静态方法
    @staticmethod
    def grow_index(
        parent_index: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
        child_index: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
    ) -> None:
        for k, v in child_index.items():
            for op_name, metadata in v.items():
                # 确保在同一个调度键下不会出现重复操作符名称
                assert (
                    op_name not in parent_index[k]
                ), f"duplicate operator {op_name} for dispatch key {k}"
                parent_index[k][op_name] = metadata

    # 获取主要实现函数 `primary`，根据 `use_out_as_primary` 决定返回 `g.out` 或 `g.functional`
    def primary(self, g: NativeFunctionsGroup) -> NativeFunction:
        if self.use_out_as_primary:
            return g.out
        else:
            return g.functional

    # 检查是否存在内核函数 `has_kernel`，通过调用 `get_kernel` 实现
    def has_kernel(self, g: NativeFunction | NativeFunctionsGroup) -> bool:
        m = self.get_kernel(g)
        return m is not None

    # 获取内核函数 `get_kernel`，返回 `g` 对应的内核或内核组
    def get_kernel(
        self, g: NativeFunction | NativeFunctionsGroup
    ) -> BackendMetadata | None:
        # 如果 g 是 NativeFunction 类型，则直接将其赋给 f
        if isinstance(g, NativeFunction):
            f = g
        # 如果 g 是 NativeFunctionsGroup 类型，则调用 self.primary 方法获取主要函数
        elif isinstance(g, NativeFunctionsGroup):
            f = self.primary(g)
        else:
            # 如果 g 不是上述两种类型，则断言出现错误
            assert_never(g)
        # 如果 f 函数名不在 self.index 中，则返回 None
        if f.func.name not in self.index:
            return None
        # 返回 self.index 中 f 函数名对应的 BackendMetadata 对象
        return self.index[f.func.name]

    def native_function_class_name(self) -> str | None:
        # 如果 self.external 为真，则返回格式化的字符串，包含 dispatch_key 的字符串形式和 "NativeFunctions"
        if self.external:
            return f"{str(self.dispatch_key)}NativeFunctions"
        else:
            # 否则返回 None，表示没有特定的类名与内部的内核函数相关联
            # TODO: 这个不一致性并不是必需的；我们也可以为内部内核生成一个类。
            # 这只需要仔细更新每个内核定义和每个内部 aten 内核的调用点。
            return None
# 定义一个数据类 FunctionSchema，用于表示代码生成中的函数模式。
# 这个数据类在整个代码生成过程中非常重要，因为它定义了操作符的类型签名，
# 大部分的代码生成都是基于类型的指导（例如，查看类型，决定如何操作，考虑如何生成 C++ 函数存根！）

# 在这个类中，我们还将看到如何在代码生成中建模数据的一般结构。在开始之前，有几个值得注意的属性：
#
#   - 这些数据类是一个“无损”的表示，与它们从中解析出来的字符串一样。事实上，我们断言，
#     根据存储在数据类中的信息，我们可以精确地重建我们从中解析出的字符串
#     （并且在解析定义内部进行了断言）。这样做有几个原因：
#
#       - 如果发现根据数据类重新构建字符串很困难，这表明你的数据表示可能是错误的。
#
#       - 它有助于确保数据类中存在所有相关的信息，这样下游用户就不会被诱使重新解析原始字符串，
#         以获取一些被省略的信息。
#
#       - 它强制你以内存中相同的方式来表示数据，这样使得对于熟悉文本格式的人来说更容易理解数据类。
#         （作为一种权衡，这意味着即使在不方便的情况下也必须对语法进行建模。但也许这意味着语法有问题！）
#         如果你不理解内部表示，请查看打印代码，看看它是如何映射到表面语法的！
#
#       - 它使得测试解析代码变得更容易，因为与字符串代码不一致的解析代码将会早早地并且大声地失败。
#         （作为一种权衡，这使得解析代码有点脆弱（特别是对于微小的空格更改，你很可能触发一个断言错误）。
#
#     总的来说，尽量使 __str__ 代码尽可能简单（即使在更复杂的解析逻辑的代价下）。此外，
#     尽量减少数据表示中的冗余。（预计算的字段是可以的：它们被定义为在相关的规范表示上的简单函数。）

#   - 这些数据类都是冻结的；一旦构造，它们的值就不会改变。这使得很容易确定给定数据来自哪里：只需查看构造函数。
#     作为一种权衡，你不能轻松地从事后分析中添加额外信息到模式中。我们施加这种限制是为了使这些结构更易于理解。

@dataclass(frozen=True)
class FunctionSchema:
    # 描述此函数模式的操作符的名称。
    name: OperatorName

    # 函数的参数
    arguments: Arguments

    # TODO: 需要在某个时候处理参数名称的冲突
    returns: tuple[Return, ...]


# 定义函数的返回值类型为元组，其中元素是 Return 对象

@property
def is_mutable(self) -> bool:
    def is_write(arg: Argument) -> bool:
        # 检查参数的注解是否存在且具有写操作
        if arg.annotation is None:
            return False
        return arg.annotation.is_write

    # 判断是否存在至少一个可写的参数
    return any(is_write(a) for a in self.arguments.flat_all)


def schema_order_arguments(self) -> Iterator[Argument]:
    # 返回按照函数声明顺序排列的参数迭代器
    return itertools.chain(
        self.arguments.flat_positional,
        self.arguments.flat_kwarg_only,
        self.arguments.out,
    )


decl_re = re.compile(r"(?P<name>[^\(]+)\((?P<args>.*)\) -> (?P<returns>.*)")


@staticmethod
def parse(func: str) -> FunctionSchema:
    # 解析函数声明字符串，返回 FunctionSchema 对象
    decls = FunctionSchema.decl_re.findall(func)
    assert len(decls) == 1, f"Invalid function schema: {func}"
    ops, args, return_decl = decls[0]
    name = OperatorName.parse(ops)
    arguments = Arguments.parse(args)
    returns = parse_returns(return_decl)
    r = FunctionSchema(name=name, arguments=arguments, returns=returns)
    assert str(r) == func, f"{str(r)} != {func}"
    return r


def returns_are_aliased(self) -> bool:
    # 判断函数的返回值中是否存在至少一个具有写操作的返回值注解
    # 先前已经断言过函数声明中不能混合存在别名和非别名的返回值
    return any(
        r
        for r in self.returns
        if r.annotation is not None and r.annotation.is_write
    )


def is_functional_fn(self) -> bool:
    # 判断函数名是否包含 "functional"，用于判断函数是否为函数式函数
    return "functional" in self.name.overload_name
    def is_out_fn(self) -> bool:
        # Note [is_out_fn]
        #
        # out functions are the variants which take an explicit out= argument
        # to populate into.  We need to know if a schema corresponds to an
        # out function for several reasons:
        #
        #   - They codegen differently in C++ API
        #       - codegen to at::add_out rather than at::add
        #       - out argument is moved to front of C++ argument list
        #
        # out functions are DEFINED to be any function with a keyword-only
        # argument that is mutable.  In principle, this could lead to a
        # false positive if you define a function that mutates a
        # kwarg only argument, but this isn't the "true" output of this
        # function.  A more robust definition that would work in this
        # case would also look at:
        #
        #   - The output types.  Out functions take in the arguments
        #     they mutate and then return them again; this is sort
        #     of "definitionally" what makes something an out function.
        #     Historically, we DO check this for consistency.
        #   - Correspondence with pure variant.  An out function
        #     should have a signature equivalent to its pure variant,
        #     but just with extra kwargs for the output elements.  This
        #     is difficult to actually check for and historically
        #     we only do this check in tools/

        # 返回是否存在一个可变的关键字参数 'out'，用于判断函数是否为 out 函数
        return bool(self.arguments.out)
    # 返回此模式的架构类型，可能是 inplace、scratch、out、mutable 或 functional 之一
    def kind(self) -> SchemaKind:
        """
        What kind of schema is this?  A functional schema is one
        that returns a newly allocated output; an inplace schema
        modifies the self argument inplace; an out schema writes
        the result into an explicitly provided out argument.
        """
        # 检查是否存在 out= 参数
        is_out = bool(self.arguments.out)
        # 检查是否存在以 "_scratch_" 开头的参数
        is_scratch = bool(
            [arg for arg in self.arguments.out if arg.name.startswith("_scratch_")]
        )
        # 检查是否为 inplace 操作
        is_inplace = self.name.name.inplace
        # 检查是否存在可变的 post_self_positional 参数
        is_mutable = any(
            a.annotation is not None and a.annotation.is_write
            for a in self.arguments.post_self_positional
        )
        # 确保 out= 和 inplace 类型不会同时存在
        assert not (is_out and is_inplace)
        
        # 根据条件返回相应的架构类型
        if is_inplace:
            return SchemaKind.inplace
        elif is_scratch:
            assert (
                is_out
            ), "invariant: all scratch operators are expected to be out= operators too"
            return SchemaKind.scratch
        elif is_out:
            assert (
                not is_scratch
            ), "We should not categorize a scratch op as an out variant. Check if the order of if statements are expected!"
            return SchemaKind.out
        elif is_mutable:
            return SchemaKind.mutable
        else:
            return SchemaKind.functional

    # 返回每个返回值的别名（如果有），否则返回 None
    # 如果返回值名称被强制与别名信息保持一致，那么我们将不需要这个方法
    def aliased_return_names(self) -> list[str | None]:
        # 初始化一个空列表用于存储返回值的别名
        outs: list[str | None] = []
        # 遍历每个返回值
        for r in self.returns:
            # 找到与当前返回值 r 具有相同注解的参数列表
            aliased_args = [
                a
                for a in self.arguments.flat_all
                if a.annotation is not None and a.annotation == r.annotation
            ]
            # 根据找到的别名参数数量决定是否有别名
            if len(aliased_args) == 0:
                outs.append(None)  # 没有找到别名参数，添加 None
            elif len(aliased_args) == 1:
                outs.append(aliased_args[0].name)  # 找到一个别名参数，将其名称添加到列表中
            else:
                # 找到多个别名参数，引发断言错误
                aliased_names = ", ".join(a.name for a in aliased_args)
                raise AssertionError(
                    f"Found a return ({r.name})that aliases multiple inputs ({aliased_names})"
                )
        return outs  # 返回所有返回值的别名列表
    # 定义一个方法 `signature`，接受多个命名参数，并返回一个函数签名对象
    def signature(
        self,
        *,
        strip_default: bool = False,
        strip_view_copy_name: bool = False,
        keep_return_names: bool = False,
    ):
        # 返回调用 `view_signature` 方法的结果，传递参数 `strip_view_copy_name=True`
        return self.view_signature(strip_view_copy_name=True)

    # 定义一个方法 `view_signature`，返回一个函数模式对象 `FunctionSchema`
    def view_signature(self) -> FunctionSchema:
        return self.signature(strip_view_copy_name=True)

    # 定义一个方法 `with_name`，接受一个操作符名称参数 `name`，返回一个新的函数模式对象 `FunctionSchema`
    def with_name(self, name: OperatorName) -> FunctionSchema:
        return FunctionSchema(
            name=name,
            arguments=self.arguments,
            returns=self.returns,
        )

    # 定义一个属性方法 `modifies_arguments`，返回一个布尔值，指示函数是否修改其参数
    def modifies_arguments(self) -> bool:
        return self.kind() in [SchemaKind.inplace, SchemaKind.out, SchemaKind.mutable]

    # 定义一个方法 `has_symint`，返回一个布尔值，指示函数参数中是否存在符号整数参数
    def has_symint(self) -> bool:
        return self.arguments.has_symint_arg()

    # 定义一个魔术方法 `__str__`，返回一个字符串，描述函数对象的字符串表示形式
    def __str__(self) -> str:
        all_arguments_str = str(self.arguments)
        if len(self.returns) == 1:
            returns = str(self.returns[0])  # 省略括号
        else:
            returns = "(" + ", ".join(map(str, self.returns)) + ")"
        return f"{self.name}({all_arguments_str}) -> {returns}"
# Here is the rest of the data model, described more briefly.
# 这里是数据模型的其余部分，简要描述。

# Simplified version for what actually shows up in built-ins.
# Look at alias_info.h for expanded syntax.  If you need the structure,
# you also need to make this structure recursive so it can be lined
# up with the type components too.  For primitives this isn't really
# necessary
# 简化版本展示内置内容的实际情况。
# 查看 alias_info.h 获取扩展语法。如果需要该结构，
# 还需要使该结构递归，以便与类型组件对齐。对于基元类型，这并不是必需的。

@dataclass(frozen=True)
# 使用 dataclass 装饰器声明 Annotation 类，使其成为不可变数据类。

class Annotation:
    # Typically only has one element.  Not actually a set so
    # we can conveniently assume it is canonically ordered
    # 通常只有一个元素。实际上不是一个集合，因此
    # 我们可以方便地假设它是按照规范顺序排列的。
    
    alias_set: tuple[str, ...]
    # 别名集合，通常只有一个元素的元组。存储别名的字符串。

    is_write: bool
    # 表示是否写操作的布尔值。

    alias_set_after: tuple[str, ...]
    # 操作后的别名集合，通常只有一个元素的元组。存储别名的字符串。

    @staticmethod
    def parse(ann: str) -> Annotation:
        # TODO: implement a proper parser if this gets more ugly
        # Regex Explanation:
        # Example: "a! -> a|b"
        # Group #1: alias before optional '|', required. Matches the first
        #   character 'a' in the example
        # Group #2: optional alias set after optional '|', matches empty string
        #   in the example
        # Group #3: optional "is write" flag, matches '!' in the example.
        # Group #4: optional section containing arrow, matches " -> a|b" in the
        #   example.
        # Group #5: optional alias after set, supports wildcard, matches "a|b"
        #   in the example.
        # Group #6: optional sub-section of alias after set, matches "|b" in the
        #   example.
        
        # 使用正则表达式解析给定的注解字符串 ann
        m = re.match(r"^([a-z])(\|[a-z])*(!?)( -> (\*|[a-z](\|[a-z])*))?$", ann)
        
        # 确保解析结果不为 None，否则抛出异常
        assert m is not None, f"unrecognized alias annotation {ann}"
        
        # 提取解析结果中的各个部分，并进行进一步处理
        before_alias = m.group(1) + (m.group(2) if m.group(2) else "")
        alias_set = tuple(before_alias.split("|"))
        is_write = m.group(3) == "!"
        
        # 如果 is_write 为 True，则别名集合长度不应大于 1
        assert not (
            is_write and len(alias_set) > 1
        ), f"alias set larger than 1 is not mutable, got {ann} instead."
        
        # 提取操作后的别名集合
        after_set = tuple(m.group(5).split("|")) if m.group(5) else tuple()
        
        # 如果前置别名集合长度大于 1 且操作后的别名集合长度大于 1，则抛出异常
        assert not (
            len(before_alias) > 1 and len(after_set) > 1
        ), f"before alias set and after alias set cannot be larger than 1 at the same time, got {ann} instead."
        
        # 创建 Annotation 类的实例对象 r，存储解析后的结果
        r = Annotation(
            alias_set=alias_set, is_write=is_write, alias_set_after=after_set
        )
        
        # 确保实例对象的字符串表示与原始注解字符串相同，否则抛出异常
        assert str(r) == ann, f"{r} != {ann}"
        
        # 返回解析后的 Annotation 对象
        return r

    def __str__(self) -> str:
        # 返回 Annotation 对象的字符串表示形式
        alias_set = "|".join(self.alias_set)
        if self.is_write:
            alias_set = f"{alias_set}!"
        alias_set_after = "|".join(self.alias_set_after)
        if alias_set_after:
            alias_set = f'{alias_set}{" -> "}{alias_set_after}'
        return alias_set
        #
# 使用 dataclass 装饰器定义一个不可变的类 Type，用于表示类型信息
@dataclass(frozen=True)
class Type:
    
    # 静态方法：解析字符串类型描述，返回相应的 Type 对象
    @staticmethod
    def parse(t: str) -> Type:
        # 调用内部方法 _parse 解析类型字符串
        r = Type._parse(t)
        # 断言解析结果的字符串形式与原始输入字符串相同，用于验证解析正确性
        assert str(r) == t, f"{r} != {t}"
        return r

    # 静态方法：解析字符串类型描述，返回相应的 Type 对象
    @staticmethod
    def _parse(t: str) -> Type:
        # 匹配可选类型（例如 int?）
        m = re.match(r"^(.+)\?$", t)
        if m is not None:
            # 如果匹配成功，返回 OptionalType 对象
            return OptionalType(Type.parse(m.group(1)))
        
        # 匹配列表类型（例如 int[10]）
        m = re.match(r"^(.+)\[([0-9]+)?\]$", t)
        if m is not None:
            # 获取列表大小，如果未指定大小则为 None
            size = int(m.group(2)) if m.group(2) is not None else None
            # 返回 ListType 对象
            return ListType(elem=Type.parse(m.group(1)), size=size)

        # 匹配自定义类类型（以 '__torch__.torch.classes.' 开头）
        m = re.match(r"^__torch__\.torch\.classes\.([a-zA-Z0-9_.]+)$", t)
        if m is not None:
            # 返回 CustomClassType 对象
            return CustomClassType(m.group(1))
        
        # 尝试根据字符串在 BaseTy 枚举中查找对应的基本类型，如果找不到则抛出异常
        try:
            return BaseType(BaseTy[t])
        except KeyError as e:
            raise RuntimeError(f"unrecognized type {t}") from e

    # 抽象方法：返回类型对象的字符串表示，子类需要实现该方法
    def __str__(self) -> str:
        raise NotImplementedError

    # 抽象方法：检查当前类型是否与给定的基本类型相似，子类需要实现该方法
    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        raise NotImplementedError

    # 返回当前类型是否类似于 Tensor 类型
    def is_tensor_like(self) -> bool:
        return self.is_base_ty_like(BaseTy.Tensor)

    # 返回当前类型是否类似于 Generator 类型
    def is_generator_like(self) -> bool:
        return self.is_base_ty_like(BaseTy.Generator)

    # 返回当前类型是否类似于 SymInt 类型
    def is_symint_like(self) -> bool:
        return self.is_base_ty_like(BaseTy.SymInt)

    # 抽象方法：返回当前类型是否可为空，子类需要实现该方法
    def is_nullable(self) -> bool:
        raise NotImplementedError

    # 抽象方法：返回当前类型是否类似于列表类型，如果是则返回 ListType 对象，否则返回 None，子类需要实现该方法
    def is_list_like(self) -> ListType | None:
        raise NotImplementedError
    # 检查当前对象是否类似于指定的基本类型
    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        # 调用元素对象的方法，检查其是否类似于指定的基本类型
        return self.elem.is_base_ty_like(base_ty)

    # 检查当前对象是否类似于符号整数类型
    def is_symint_like(self) -> bool:
        # 调用元素对象的方法，检查其是否类似于符号整数类型
        return self.elem.is_symint_like()

    # 始终返回 True，表示当前对象是可空的
    def is_nullable(self) -> bool:
        return True

    # 检查当前对象是否类似于列表类型，如果是则返回列表类型，否则返回 None
    def is_list_like(self) -> ListType | None:
        # 调用元素对象的方法，检查其是否类似于列表类型
        return self.elem.is_list_like()
# A type representing a PyTorch custom class
@dataclass(frozen=True)
class CustomClassType(Type):
    class_name: str

    def __str__(self) -> str:
        """
        Return the class name with the prefix __torch__.torch.classes.
        """
        return f"__torch__.torch.classes.{self.class_name}"

    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        # Custom classes are not considered base types.
        return False

    def is_symint_like(self) -> bool:
        # Custom classes are not considered symint-like.
        return False

    def is_nullable(self) -> bool:
        """
        Assume a custom class is not nullable.
        """
        return False

    def is_list_like(self) -> ListType | None:
        # Custom classes are not list-like.
        return None


# List types specify that we may have multiples of an element.  We
# also support explicit sizes on list types, but these have
# some nontrivial semantics!  (However, for C++ API purposes, explicit
# sizes are mostly erased from the type system.)
#
# DANGER WILL ROBINSON: C++ elaboration depends on elem type; e.g.,
# int[] elaborates differently than bool[3]!
@dataclass(frozen=True)
class ListType(Type):
    elem: Type
    size: int | None

    def __str__(self) -> str:
        """
        Return a string representation of the list type, including size if provided.
        """
        size = f"{self.size}" if self.size else ""
        return f"{self.elem}[{size}]"

    def is_base_ty_like(self, base_ty: BaseTy) -> bool:
        # Check if the list element is base type-like.
        return self.elem.is_base_ty_like(base_ty)

    def is_symint_like(self) -> bool:
        # Check if the list element is symint-like.
        return self.elem.is_symint_like()

    def is_nullable(self) -> bool:
        # Check if the list element is nullable.
        return self.elem.is_nullable()

    def is_list_like(self) -> ListType | None:
        # List types are themselves list-like.
        return self


@dataclass(frozen=True)
class Argument:
    # NB: I didn't put kwarg_only as a boolean field here, unlike
    # c10::Argument, so that printing works correctly

    name: str
    type: Type
    default: str | None

    # The semantics of the annotation field are a little strange.
    #
    # Alias annotations parametrize Tensors (since Tensors are the only things
    # that can alias.)  This motivates why I write Tensor(a!)?  (and not, for
    # example, Tensor?(a!)), because the (a!) describes aliasing on the tensor,
    # which may be optional (i.e., the alias annotation should bind first to
    # Tensor, before the optional postfix annotation).
    #
    # However, despite being a property of Tensor, we (and c10::Argument)
    # store the annotation at the top level of the Argument, rather than
    # inside the embedded Tensor type.  In the C++ version of this
    # class, we then go through great lengths to mimic the type
    # structure in the annotation structure so we can correlate
    # annotations with types.
    #
    # Now, it turns out, in all applications in code generation, the
    # structure of annotated types is very simple.  So we just hard
    # code it here.  But if we ever do get anything more complex, this
    # model will have to change!
    annotation: Annotation | None

    @property
    def alias_info(self) -> Annotation | None:
        """
        Return the alias information of the argument, which is stored in the annotation field.
        """
        return self.annotation

    @staticmethod
    # 解析参数字符串并返回 Argument 对象
    def parse(arg: str) -> Argument:
        # 声明变量 name 和 default
        name: str
        default: str | None
        # 检查参数中是否包含空格，如果没有则抛出异常
        assert " " in arg, f"illegal argument '{arg}'"
        # 将参数按最后一个空格分割为类型及注解和名称及默认值两部分
        type_and_annot, name_and_default = arg.rsplit(" ", 1)
        # 如果默认值部分包含等号，则解析出名称和默认值
        if "=" in name_and_default:
            assert (
                name_and_default.count("=") == 1
            ), f"illegal argument with default value: '{name_and_default}'"
            name, default = name_and_default.split("=")
        else:
            # 否则，默认值为 None
            name = name_and_default
            default = None
        # TODO: deduplicate annotation matching with Return
        # 使用正则表达式匹配类型字符串中是否包含 Tensor 格式
        match = re.match(r"Tensor\((.+)\)(.*)", type_and_annot)
        # 声明 annotation 变量，用于存储注解信息
        annotation: Annotation | None
        # 如果匹配成功
        if match:
            # 如果更新这部分，请确保 __str__ 方法仍然有效
            assert match.group(2) in [
                "",
                "?",
                "[]",
            ], "unrecognized alias analysis form with Tensor"
            # 构建类型字符串，添加注解信息
            type_s = "Tensor" + match.group(2)
            # 解析注解信息并赋值给 annotation
            annotation = Annotation.parse(match.group(1))
        else:
            # 否则，类型字符串即为 type_and_annot
            type_s = type_and_annot
            annotation = None
        # 解析类型信息并赋值给 type
        type = Type.parse(type_s)
        # 创建 Argument 对象并返回
        r = Argument(
            name=name,
            type=type,
            default=default,
            annotation=annotation,
        )
        # 检查创建的 Argument 对象是否与原始参数字符串相匹配，否则抛出异常
        assert str(r) == arg, f"{str(r)} != {arg}"
        return r

    # 判断属性 is_write 是否为 True
    @property
    def is_write(self) -> bool:
        return self.annotation is not None and self.annotation.is_write

    # 返回对象的字符串表示形式
    def __str__(self) -> str:
        # 获取类型字符串
        type = f"{self.type}"
        # 如果有注解信息
        if self.annotation:
            # 确保类型在 ["Tensor", "Tensor?", "Tensor[]"] 中
            assert type in ["Tensor", "Tensor?", "Tensor[]"]
            # 替换类型字符串中的 Tensor 为 Tensor(注解信息)
            type = type.replace("Tensor", f"Tensor({self.annotation})")
        # 如果名称为 None，则返回类型字符串
        if self.name is None:
            return type
        else:
            # 否则，返回类型和名称以及可能的默认值的字符串表示形式
            mb_default = ""
            if self.default:
                mb_default = f"={self.default}"
            return f"{type} {self.name}{mb_default}"
# 使用 dataclass 装饰器定义一个名为 Return 的数据类，其实例是不可变的
@dataclass(frozen=True)
class Return:
    # 声明数据类的属性：name 表示名称（可选，可能为 None），type 表示类型，annotation 表示注解（可选）
    name: str | None
    type: Type
    annotation: Annotation | None

    # 定义 alias_info 属性，返回注解信息或 None
    @property
    def alias_info(self) -> Annotation | None:
        return self.annotation

    # 定义静态方法 parse，用于解析传入的字符串参数 arg，并返回一个 Return 实例
    @staticmethod
    def parse(arg: str) -> Return:
        name: str | None
        # 如果参数中包含空格，则以最后一个空格为分隔符，分割出类型和名称
        if " " in arg:
            type_and_annot, name = arg.rsplit(" ", 1)
        else:
            type_and_annot = arg
            name = None
        # 使用正则表达式匹配以 "Tensor(...)" 开头的类型和注解
        match = re.match(r"Tensor\((.+)\)(.*)", type_and_annot)
        annotation: Annotation | None
        if match:
            # 如果匹配成功，则解析注解部分，并将类型设为 Tensor 类型
            assert match.group(2) in [
                "",
                "?",
                "[]",
            ], "unrecognized alias analysis form with Tensor"
            type_s = "Tensor" + match.group(2)
            annotation = Annotation.parse(match.group(1))
        else:
            # 如果匹配不成功，则将类型设为 type_and_annot，并注解设为 None
            type_s = type_and_annot
            annotation = None
        # 解析类型字符串，并创建 Return 实例 r
        type = Type.parse(type_s)
        r = Return(
            name=name,
            type=type,
            annotation=annotation,
        )
        # 确保创建的 Return 实例的字符串表示与输入的参数 arg 相同
        assert str(r) == arg, f"{str(r)} != {arg}"
        return r

    # 定义 is_write 属性，用于判断该 Return 实例是否为写操作
    @property
    def is_write(self) -> bool:
        return self.annotation is not None and self.annotation.is_write

    # 定义 __str__ 方法，返回该 Return 实例的字符串表示
    def __str__(self) -> str:
        type = f"{self.type}"
        if self.annotation:
            # 如果有注解，则替换类型中的 "Tensor" 为 "Tensor(注解)"
            assert type in ["Tensor", "Tensor?", "Tensor[]"]
            type = type.replace("Tensor", f"Tensor({self.annotation})")
        if self.name is None:
            return type
        else:
            return f"{type} {self.name}"


# 用 dataclass 装饰器定义一个名为 SelfArgument 的数据类，其实例是不可变的
@dataclass(frozen=True)
class SelfArgument:
    # 声明数据类的属性：argument 表示自变量
    argument: Argument


# 用 dataclass 装饰器定义一个名为 TensorOptionsArguments 的数据类，其实例是不可变的
@dataclass(frozen=True)
class TensorOptionsArguments:
    # 声明数据类的属性：dtype、layout、device、pin_memory 分别表示数据类型、布局、设备、是否固定内存
    dtype: Argument
    layout: Argument
    device: Argument
    pin_memory: Argument

    # 定义 all 方法，返回所有属性构成的列表
    def all(self) -> Sequence[Argument]:
        return [self.dtype, self.layout, self.device, self.pin_memory]


# 用 dataclass 装饰器定义一个名为 Arguments 的数据类，其实例是不可变的
@dataclass(frozen=True)
class Arguments:
    # 声明数据类的属性：pre_self_positional 表示在 self 之前的位置参数元组
    pre_self_positional: tuple[Argument, ...]
    # self_arg 表示 SelfArgument 实例或 None
    self_arg: SelfArgument | None
    # post_self_positional 表示在 self 之后的位置参数元组
    post_self_positional: tuple[Argument, ...]

    # pre_tensor_options_kwarg_only 表示仅限于 tensor options 前的关键字参数元组
    pre_tensor_options_kwarg_only: tuple[Argument, ...]
    # tensor_options 表示 TensorOptionsArguments 实例或 None
    tensor_options: TensorOptionsArguments | None
    # post_tensor_options_kwarg_only 表示仅限于 tensor options 后的关键字参数元组
    post_tensor_options_kwarg_only: tuple[Argument, ...]

    # 与之前的代码生成不同，这里已经将 'out' 参数分离出来
    # 在规范表示中，将其从 kwarg 参数中移除
    # 这个选择是基于许多后续变换，它们会特别处理 out 参数；此外，
    # 您可以看到规范性并没有被违反！
    out: tuple[Argument, ...]  # 这些也仅限于 kwarg

    @property
    def flat_non_out(self) -> Sequence[Argument]:
        # 返回一个不包含 out 参数的扁平化列表
        ret: list[Argument] = []
        ret.extend(self.flat_positional)  # 添加所有位置参数
        ret.extend(self.flat_kwarg_only)  # 添加所有仅限于 kwarg 的参数
        return ret

    @property
    def flat_positional(self) -> Sequence[Argument]:
        # 返回一个扁平化的位置参数列表
        ret: list[Argument] = []
        ret.extend(self.pre_self_positional)  # 添加所有 self 之前的位置参数
        if self.self_arg is not None:
            ret.append(self.self_arg.argument)  # 添加 self 参数
        ret.extend(self.post_self_positional)  # 添加所有 self 之后的位置参数
        return ret

    @property
    def post_self_positional_mutable(self) -> Sequence[Argument]:
        # 返回一个包含可变的 post_self_positional 参数列表
        return [a for a in self.post_self_positional if a.is_write]

    # 注意：不包含 out 参数
    @property
    def flat_kwarg_only(self) -> Sequence[Argument]:
        # 返回一个扁平化的仅限于 kwarg 的参数列表
        ret: list[Argument] = []
        ret.extend(self.pre_tensor_options_kwarg_only)  # 添加所有 tensor options 之前的 kwarg 参数
        if self.tensor_options is not None:
            ret.extend(self.tensor_options.all())  # 添加 tensor options 中的所有参数
        ret.extend(self.post_tensor_options_kwarg_only)  # 添加所有 tensor options 之后的 kwarg 参数
        return ret

    @property
    def flat_all(self) -> Sequence[Argument]:
        # 返回一个扁平化的所有参数列表，包括 positional、kwarg_only 和 out 参数
        ret: list[Argument] = []
        ret.extend(self.flat_positional)  # 添加所有位置参数
        ret.extend(self.flat_kwarg_only)  # 添加所有仅限于 kwarg 的参数
        ret.extend(self.out)  # 添加所有 out 参数
        return ret

    @property
    def non_out(
        self,
    ) -> Sequence[Argument | SelfArgument | TensorOptionsArguments]:
        # 返回一个不包含 out 参数的列表，包括 positional 和 kwarg_only 参数
        ret: list[Argument | SelfArgument | TensorOptionsArguments] = []
        ret.extend(self.positional)  # 添加所有位置参数
        ret.extend(self.kwarg_only)  # 添加所有仅限于 kwarg 的参数
        return ret

    @property
    def positional(self) -> Sequence[Argument | SelfArgument]:
        # 返回一个位置参数的列表，包括 pre_self_positional、self_arg 和 post_self_positional 参数
        ret: list[Argument | SelfArgument] = []
        ret.extend(self.pre_self_positional)  # 添加所有 self 之前的位置参数
        if self.self_arg is not None:
            ret.append(self.self_arg)  # 添加 self 参数
        ret.extend(self.post_self_positional)  # 添加所有 self 之后的位置参数
        return ret

    @property
    def kwarg_only(self) -> Sequence[Argument | TensorOptionsArguments]:
        # 返回一个仅限于 kwarg 的参数列表，包括 pre_tensor_options_kwarg_only、tensor_options 和 post_tensor_options_kwarg_only 参数
        ret: list[Argument | TensorOptionsArguments] = []
        ret.extend(self.pre_tensor_options_kwarg_only)  # 添加所有 tensor options 之前的 kwarg 参数
        if self.tensor_options is not None:
            ret.append(self.tensor_options)  # 添加 tensor options 参数
        ret.extend(self.post_tensor_options_kwarg_only)  # 添加所有 tensor options 之后的 kwarg 参数
        return ret

    @property
    def all(self) -> Sequence[Argument | SelfArgument | TensorOptionsArguments]:
        # 返回一个包含所有参数的列表，包括 positional、kwarg_only 和 out 参数
        ret: list[Argument | SelfArgument | TensorOptionsArguments] = []
        ret.extend(self.positional)  # 添加所有位置参数
        ret.extend(self.kwarg_only)  # 添加所有仅限于 kwarg 的参数
        ret.extend(self.out)  # 添加所有 out 参数
        return ret

    def mutable_arg_names(self) -> list[str]:
        # 返回一个包含可变参数名称的列表，这些参数被标注为可写
        return [
            a.name
            for a in self.flat_all
            if a.annotation is not None and a.annotation.is_write
        ]
    def has_tensor_arg(self) -> bool:
        # 检查在 self.flat_non_out 中是否有任何参数类型为张量（tensor）的参数
        return any(a.type.is_tensor_like() for a in self.flat_non_out)

    def has_symint_arg(self) -> bool:
        # 检查在 self.flat_non_out 中是否有任何参数类型为符号整数（symint）的参数
        return any(a.type.is_symint_like() for a in self.flat_non_out)

    def has_generator_arg(self) -> bool:
        # 检查在 self.flat_non_out 中是否有任何参数类型为生成器（generator）的参数
        return any(a.type.is_generator_like() for a in self.flat_non_out)

    def signature(self, *, strip_default: bool = False) -> Arguments:
        # 构造方法签名，返回 Arguments 类型对象
        # 可选择是否剥去默认值的注解
        def strip_arg_annotation(a: Argument) -> Argument:
            # 剥去参数的注解信息
            return Argument(
                name=a.name,
                type=a.type,
                default=a.default if not strip_default else None,
                annotation=None,
            )

        return Arguments(
            # 处理前自变量位置参数，剥去参数的注解信息
            pre_self_positional=tuple(
                map(strip_arg_annotation, self.pre_self_positional)
            ),
            # 剥去 self 参数的注解信息
            self_arg=SelfArgument(strip_arg_annotation(self.self_arg.argument))
            if self.self_arg is not None
            else None,
            # 处理后自变量位置参数，剥去参数的注解信息
            post_self_positional=tuple(
                map(strip_arg_annotation, self.post_self_positional)
            ),
            # 将后张量选项关键字参数转换为前张量选项关键字参数
            pre_tensor_options_kwarg_only=tuple(
                map(strip_arg_annotation, self.pre_tensor_options_kwarg_only)
            )
            + tuple(map(strip_arg_annotation, self.post_tensor_options_kwarg_only)),
            # 张量选项在签名中被丢弃
            tensor_options=None,
            # 空元组，因为在签名中不包括后张量选项关键字参数
            post_tensor_options_kwarg_only=tuple(),
            # 在签名中被丢弃的 out 参数
            out=(),
        )

    def remove_self_annotation(self) -> Arguments:
        # 断言 self.self_arg 不为 None，然后剥去其注解信息
        assert self.self_arg is not None
        return dataclasses.replace(
            self,
            self_arg=SelfArgument(
                dataclasses.replace(self.self_arg.argument, annotation=None)
            ),
        )

    def with_out_args(self, outs: list[Argument]) -> Arguments:
        # 断言 self.out 的长度为 0，然后将 outs 参数作为 out 参数添加到返回的 Arguments 对象中
        assert len(self.out) == 0
        return dataclasses.replace(
            self,
            out=tuple(outs),
        )

    @staticmethod
    # 定义一个私有方法 _preparse，用于预解析传入的参数字符串，返回三个列表：位置参数列表、仅关键字参数列表和输出参数列表
    def _preparse(args: str) -> tuple[list[Argument], list[Argument], list[Argument]]:
        positional: list[Argument] = []  # 存储位置参数的列表
        kwarg_only: list[Argument] = []  # 存储仅关键字参数的列表
        out: list[Argument] = []          # 存储输出参数的列表
        arguments_acc = positional        # 当前正在处理的参数列表，默认从位置参数列表开始

        # TODO: 在此处使用一个真正的解析器；当前的实现会受到像 std::array<bool, 2>（注意空格）这样的签名影响
        # 如果参数字符串中存在空字符串，则跳过
        # 如果参数字符串为 "*", 则将当前处理列表切换到仅关键字参数列表
        for arg in args.split(", "):
            if not arg:
                continue
            if arg == "*":
                assert (
                    arguments_acc is positional
                ), "invalid syntax: kwarg-only specifier * can only occur once"
                arguments_acc = kwarg_only
                continue
            # 解析参数字符串为 Argument 对象
            parg = Argument.parse(arg)
            # 当前实现依赖于这样一个不变条件：没有仅关键字参数的可变参数。如果要放宽这一限制，需要更语义化的匹配方法，
            # 考虑到返回参数。在那种情况下，需要在 FunctionSchema 的更高层管理输出计算。参见注释 [is_out_fn]
            if parg.annotation is not None and parg.annotation.is_write:
                # 如果参数具有写入注解
                if arguments_acc is positional:
                    pass  # 什么也不做，因为位置参数列表不变
                elif arguments_acc is kwarg_only:
                    arguments_acc = out  # 切换到输出参数列表
            else:
                assert arguments_acc is not out  # 确保不在输出参数列表中
            arguments_acc.append(parg)  # 将解析后的参数对象添加到当前处理列表中

        return positional, kwarg_only, out  # 返回三个参数列表的元组

    @staticmethod
    def __str__(self) -> str:
        all_arguments: list[str] = []  # 存储所有参数描述的列表
        all_arguments.extend(map(str, self.flat_positional))  # 将平铺的位置参数描述添加到列表中
        # 如果存在平铺的仅关键字参数或输出参数，则添加一个 "*"
        if self.flat_kwarg_only or self.out:
            all_arguments.append("*")
        # 将平铺的仅关键字参数描述和输出参数描述添加到列表中
        all_arguments.extend(map(str, self.flat_kwarg_only))
        all_arguments.extend(map(str, self.out))
        # 返回所有参数描述组成的字符串，用逗号分隔
        return ", ".join(all_arguments)

    def __post_init__(self) -> None:
        # TODO: 这些不变条件看起来奇怪地不对称？
        # TODO: 更复杂的类型？
        if self.self_arg is None:
            assert not self.pre_self_positional  # 确保没有预自变量位置参数
        if self.tensor_options is None:
            assert not self.post_tensor_options_kwarg_only  # 确保没有后张量选项的仅关键字参数

        # 我们不允许以下任何一项具有参数注解，以保持简单性。
        # 查找具有写入注解的可变预自变量位置参数
        mutable_pre_self_positionals = [
            a
            for a in self.pre_self_positional
            if a.annotation is not None and a.annotation.is_write
        ]
        assert (
            len(mutable_pre_self_positionals) == 0
        ), "mutable pre_self_positional arguments are not currently supported in the schema"
# 定义一组有效的原地操作名称列表，这些名称以 "__iXXX__" 形式表示。
# 详情请参考 https://www.python.org/dev/peps/pep-0203/#new-methods
# 注意：PyTorch 实际上并未实现所有这些操作。
AUGMENTED_ASSIGNMENT_NAMES = [
    "add",
    "sub",
    "mul",
    "div",
    "mod",
    "pow",
    "lshift",
    "rshift",
    "and",
    "xor",
    "or",
]

# BaseOperatorName 表示操作符名称的基本信息，不包括重载名称。
# 与通常的字符串不同，我们直接表示从字符串中提取的几个重要的语义信息：
# 是否为原地操作（如 add_）以及是否为双下划线方法（__add__）。
@dataclass(frozen=True)
class BaseOperatorName:
    base: str  # 操作符的基本名称
    inplace: bool  # 是否为原地操作
    dunder_method: bool  # 是否为双下划线方法
    # 注释 [Overload Ambiguity With Functional Variants]
    # 一些操作符同时具有“可变”和“函数式”变体（native_batch_norm 是一个很好的例子，尽管目前不是这种情况）。
    # 对于这些操作符，可变和函数式变体接受相同的参数集，但具有不同的别名注解。
    # 当尝试将 OverloadPacket 解析为一个重载时，使用一组输入参数时会产生歧义。
    #
    # 因此，与其将函数式变体作为真正的重载，例如：
    #   native_batch_norm（可变变体）
    #   native_batch_norm.functional（函数式变体）
    # 我们将其作为一个新的基本操作符：
    #   native_batch_norm_functional（函数式变体）
    #
    # 在理想情况下，我们可能会反转这一点，使操作符为：
    #   native_batch_norm.mutable（可变变体）
    #   native_batch_norm（函数式变体）
    #
    # 但由于这样做会破坏向后兼容性，所以我们只能采取上述的建模方式。
    functional_overload: bool = False

    @staticmethod
    # 定义一个函数 parse，接受一个字符串参数 op，返回一个 BaseOperatorName 对象
    def parse(op: str) -> BaseOperatorName:
        # 断言 op 不为空字符串
        assert op != ""
        # 断言 op 不以 "_out" 结尾，否则输出指定信息
        assert not op.endswith("_out"), (
            "_out suffix is reserved and not permitted for operator names; "
            "did you mean to specify an out overload name instead?"
        )
        # 使用正则表达式匹配 op 是否为双下划线开头和结尾的格式
        m = re.match(r"^__([^_]+)__$", op)
        # 如果匹配成功
        if m is not None:
            # op 是双下划线方法
            dunder_method = True
            # 获取双下划线方法的基础名
            base = m.group(1)
            # 如果 base 的形式为 i+某个增强赋值操作名，则将 inplace 设为 True，并从 base 中去除首字符
            if any(base == f"i{n}" for n in AUGMENTED_ASSIGNMENT_NAMES):
                inplace = True
                base = base[1:]
            else:
                # 否则 inplace 设为 False
                inplace = False
                # 临时的，虽然这并非固有真实，但对于双下划线方法历史上是真实的
                # 我们支持这种形式（但是，如果我们得到例如 __int__，这将是错误的！）
                assert base[0] != "i"
        else:
            # 如果不是双下划线方法
            dunder_method = False
            # 基础名设为 op
            base = op
            # 如果基础名以 "_" 结尾，则将 inplace 设为 True，并从 base 中去除末字符
            if base[-1] == "_":
                inplace = True
                base = base[:-1]
            else:
                # 否则 inplace 设为 False
                inplace = False

        # 定义功能性后缀为 "_functional"
        functional_suffix = "_functional"
        # 如果基础名以功能性后缀结尾
        if base.endswith(functional_suffix):
            # 是功能性重载
            functional_overload = True
            # 从基础名中去除功能性后缀
            base = base[: -len(functional_suffix)]
            # 对于有功能性和可变变体（如 native_batch_norm）的操作，目前禁止使用双下划线方法
            assert not dunder_method and not inplace
        else:
            # 否则不是功能性重载
            functional_overload = False

        # 创建一个 BaseOperatorName 对象 r，包含 base、inplace、dunder_method 和 functional_overload 属性
        r = BaseOperatorName(
            base=base,
            inplace=inplace,
            dunder_method=dunder_method,
            functional_overload=functional_overload,
        )
        # 断言创建的 BaseOperatorName 对象的字符串表示与原始字符串 op 相等，否则输出指定信息
        assert str(r) == op, f"{str(r)} != {op}"
        # 返回 BaseOperatorName 对象 r
        return r

    # 定义一个方法 __str__，返回实例对象的字符串表示
    def __str__(self) -> str:
        # 如果是双下划线方法
        if self.dunder_method:
            # 如果 inplace 为 True，则 i 设为 "i"，否则为空字符串
            i = "i" if self.inplace else ""
            # 返回双下划线方法的字符串表示，格式为 "__{i}{self.base}__"
            return f"__{i}{self.base}__"
        else:
            # 如果不是双下划线方法
            # 如果 inplace 为 True，则 i 设为 "_"，否则为 "_functional"（如果 functional_overload 为 True）
            i = (
                "_"
                if self.inplace
                else "_functional"
                if self.functional_overload
                else ""
            )
            # 返回非双下划线方法的字符串表示，格式为 "{self.base}{i}"
            return f"{self.base}{i}"
# Operator name is a dataclass representing an operator name along with an overload string.
# It is frozen to make instances immutable after creation.
@dataclass(frozen=True)
class OperatorName:
    # `name` stores the base operator name.
    # `overload_name` stores a string representing an optional overload name.
    name: BaseOperatorName
    overload_name: str

    # Static method to parse an operator name string `op_name` into an OperatorName object.
    @staticmethod
    def parse(op_name: str) -> OperatorName:
        if "." in op_name:
            # Split the operator name into `name` and `overload_name` parts based on the first dot.
            name, overload_name = op_name.split(".", 1)
        else:
            # If no dot is found, the entire name is considered as `name`.
            name = op_name
            overload_name = ""
        # Create an OperatorName object using BaseOperatorName's parse method for `name`.
        r = OperatorName(name=BaseOperatorName.parse(name), overload_name=overload_name)
        # Assertion to ensure the reconstructed string matches the original `op_name`.
        assert str(r) == op_name, f"{str(r)} != {op_name}"
        return r

    # Method to return the string representation of an OperatorName object.
    def __str__(self) -> str:
        if self.overload_name:
            # If there is an overload name, format the string as `name.overload_name`.
            return f"{self.name}.{self.overload_name}"
        else:
            # Otherwise, return just `name`.
            return f"{self.name}"

    # Method to return an unambiguous name for the operator.
    # This method is synchronized with naming schemes in specific source files.
    def unambiguous_name(self) -> str:
        if self.overload_name:
            # Format the name as `name_overload_name` if overload is present.
            return f"{self.name}_{self.overload_name}"
        else:
            # Otherwise, return just `name`.
            return f"{self.name}"

    # Method to create a new OperatorName instance with inplace=False in the base name.
    def remove_inplace(self) -> OperatorName:
        return OperatorName(
            name=BaseOperatorName(
                base=self.name.base,
                inplace=False,
                dunder_method=self.name.dunder_method,
            ),
            overload_name=self.overload_name,
        )

    # Method to create a new OperatorName instance with a specified overload name.
    def with_overload(self, overload: str) -> OperatorName:
        return OperatorName(
            name=BaseOperatorName(
                base=self.name.base,
                inplace=False,
                dunder_method=self.name.dunder_method,
            ),
            overload_name=overload,
        )


# Function to determine if a NativeFunction `f` gets generated as an inplace wrapper.
# It checks conditions related to function kinds and backend kernel availability.
def gets_generated_out_inplace_wrapper(
    f: NativeFunction, g: NativeFunctionsGroup, b: BackendIndex
) -> bool:
    return (
        f.func.kind() is not SchemaKind.functional
        and not b.has_kernel(f)
        and b.has_kernel(g.functional)
    )


# Dataclass representing a group of NativeFunction objects that are views.
# It pairs a view NativeFunction with an optional `view_copy` counterpart.
@dataclass(frozen=True)
class NativeFunctionsViewGroup:
    view: NativeFunction
    # Note: the {view}_copy operator is optional because we currently don't generate copy variants
    # 定义一个类型注解，表示这个变量可以是 NativeFunction 类型或者 None
    view_copy: NativeFunction | None
    # 定义另一个类型注解，表示这个变量可以是 NativeFunction 类型或者 None
    view_inplace: NativeFunction | None

    # 初始化方法，用于验证视图操作是否正确设置
    def __post_init__(self) -> None:
        # 断言当前视图操作必须是视图操作
        assert self.view.is_view_op
        # 如果 view_copy 为 None，则验证其生成的视图副本是否存在
        if self.view_copy is None:
            # 断言不应生成复合隐式自动微分视图
            assert not gets_generated_view_copy(self.view), (
                f"{str(self.view.func.name)} appears to be a new operator that aliases its inputs."
                " The codegen expects you to add a corresponding operator to native_functions.yaml:"
                f" {get_view_copy_name(self.view)!s}."
                " See Note [view_copy NativeFunctions] for details."
            )
        else:
            # 否则，验证 view_copy 的函数名是否以 "_copy" 或 "_scatter" 结尾
            assert self.view_copy.func.name.name.base.endswith(("_copy", "_scatter"))
            # 验证 view 和 view_copy 的函数签名是否匹配（去除视图副本名称）
            assert self.view.func.signature() == self.view_copy.func.signature(
                strip_view_copy_name=True,
            )
            # 断言 view_copy 必须标记为 'view_copy'
            assert "view_copy" in self.view_copy.tags, (
                f"{str(self.view_copy.func.name), str(self.view.tags)} appears to be a view_copy operator. The codegen expects"
                " view_copy operators to be annotated with the 'view_copy' tag in native_functions.yaml."
                " See Note [view_copy NativeFunction] for details."
            )
        
        # 如果存在 view_inplace，则验证其函数签名与 view 的函数签名是否匹配
        if self.view_inplace is not None:
            assert self.view.func.signature() == self.view_inplace.func.signature()

        # 如果 view 具有复合隐式自动微分内核
        if self.view.has_composite_implicit_autograd_kernel:
            # 如果存在 view_inplace，则验证其也具有复合隐式自动微分内核
            if self.view_inplace is not None:
                assert self.view_inplace.has_composite_implicit_autograd_kernel, (
                    f"{str(self.view.func.name)} and {str(self.view_inplace.func.name)} must either"
                    " both have CompositeImplicitAutograd kernels, or both not have composite kernels."
                )
        
        # 如果 view 具有复合隐式自动微分 NestedTensor 内核
        if self.view.has_composite_implicit_autograd_nested_tensor_kernel:
            # 如果存在 view_inplace，则验证其也具有复合隐式自动微分 NestedTensor 内核
            if self.view_inplace is not None:
                assert (
                    self.view_inplace.has_composite_implicit_autograd_nested_tensor_kernel
                ), (
                    f"{str(self.view.func.name)} and {str(self.view_inplace.func.name)} must either"
                    " both have CompositeImplicitAutogradNestedTensor kernels, or both not have composite kernels."
                )

    # 返回一个迭代器，包含视图操作函数
    def functions(self, *, include_copy: bool = True) -> Iterator[NativeFunction]:
        yield self.view
        # 如果 view_inplace 存在，则也包含其函数
        if self.view_inplace is not None:
            yield self.view_inplace
        # 如果 view_copy 存在且 include_copy 为 True，则包含其函数
        if self.view_copy is not None and include_copy:
            yield self.view_copy

    # 返回视图操作的根名称
    @property
    def root_name(self) -> str:
        return self.view.root_name

    # 下一个属性的注释在下一个代码块中添加
    def composite(self) -> bool:
        # 确保 "group" 是一致的断言条件。
        # 如果视图操作是复合的，则它的原地视图操作也是复合的。
        return self.view.has_composite_implicit_autograd_kernel
# 判断给定的 NativeFunction 是否可以生成视图拷贝操作
def gets_generated_view_copy(f: NativeFunction) -> bool:
    # 如果不是视图操作，则返回 False
    if not f.is_view_op:
        return False
    # 如果具有 CompositeImplicitAutograd 内核，则不生成拷贝变体
    if f.has_composite_implicit_autograd_kernel:
        return False
    # 如果是 inplace 视图，则不生成拷贝变体
    if "inplace_view" in f.tags:
        return False
    # 如果操作名称以 _inverse 结尾，则假定手动定义了拷贝变体，不生成
    if f.func.name.name.base.endswith("_inverse"):
        return False
    # 否则返回 True，表示生成拷贝变体
    return True


# 给定一个对应视图操作的 NativeFunction，返回对应的 "copy" 变体的 OperatorName
def get_view_copy_name(f: NativeFunction) -> OperatorName:
    # 确保操作名称在已有显式视图拷贝操作的列表中，或者可以生成视图拷贝操作
    list_of_ops_with_explicit_view_copy_operators = ["narrow"]
    if str(f.func.name) not in list_of_ops_with_explicit_view_copy_operators:
        assert gets_generated_view_copy(f)
    
    # 构造基础名称，添加 '_copy' 后缀
    base_name = f"{f.func.name.name.base}_copy"
    # 构造 OperatorName 对象，指定不是原地操作且保留原操作的特殊方法
    view_copy_name = OperatorName(
        name=BaseOperatorName(
            base=base_name, inplace=False, dunder_method=f.func.name.name.dunder_method
        ),
        overload_name=f.func.name.overload_name,
    )
    return view_copy_name


# 解析返回值声明字符串，返回一个元组列表
def parse_returns(return_decl: str) -> tuple[Return, ...]:
    """
    输入: '()'
    输出: []
    """
    # 如果返回声明为 '()'，则返回空元组
    if return_decl == "()":
        return ()
    # 去掉声明的括号后，按逗号分隔解析每个返回值声明，返回解析后的元组
    if return_decl[0] == "(" and return_decl[-1] == ")":
        return_decl = return_decl[1:-1]
    return tuple(Return.parse(arg) for arg in return_decl.split(", "))


# Precompute 实例由内核参数名映射到应在 impl 函数中替换该内核参数的 Argument 实例列表组成
@dataclass(frozen=True)
class Precompute:
    # 映射，从内核参数名 -> 替换/取代它的预计算元素列表
    replace: dict[str, list[Argument]]
    # 添加的未替换预计算参数列表
    add: list[Argument]

    @staticmethod
    def parse(src: object) -> Precompute:
        # 断言确保 src 是一个列表对象
        assert isinstance(src, list)

        # src 是一个字符串列表，每个字符串的格式为:
        #   {kernel param name} -> {replacement decl}[, {replacement decl}, ...]
        #   [{add decl}[, {add decl}, ...]]
        # 最后一行是可选的，包含直接添加到预计算参数的内容，无需替换。
        # 其他行被解析用于获取哪些预计算元素应该替换哪些内核参数。
        add_args = []
        if " -> " not in src[-1]:
            # 将最后一行按逗号分隔，解析为 Argument 对象列表，添加到 add_args 中
            add_list = src[-1].split(",")
            add_args = [Argument.parse(name.strip()) for name in add_list]
            # 将 src 更新为除了最后一行的部分
            src = src[:-1]

        replace = {}
        for raw_replace_item in src:
            assert isinstance(raw_replace_item, str)
            assert " -> " in raw_replace_item, (
                "precomputed parameters without replacement"
                " are allowed only in the last line"
            )

            # 将每个替换条目按 " -> " 分隔，解析出内核参数名和替换参数列表
            arg, with_list_raw = raw_replace_item.split(" -> ")
            assert (
                " " not in arg
            ), f"illegal kernel param name '{arg}' in precomputed parameters'"
            with_list = with_list_raw.split(",")
            with_list_args = [Argument.parse(name.strip()) for name in with_list]
            replace[arg] = with_list_args

        # 创建 Precompute 对象，传入替换字典和添加参数列表
        r = Precompute(replace=replace, add=add_args)
        # 断言 Precompute 对象转换为列表后与原始输入 src 相同
        assert r.to_list() == src, "r.to_list() != src"
        return r

    def __post_init__(self) -> None:
        # 检查添加参数列表中的每个参数，确保其名称不全为大写
        for a in self.add:
            assert a.name.upper() != a.name

        # 检查替换字典中每组替换参数列表中的每个参数，确保其名称不全为大写
        for args in self.replace.values():
            for a in args:
                assert a.name.upper() != a.name

    def to_list(self) -> list[str]:
        replace_list = []
        # 遍历替换字典，将每个内核参数和对应的替换参数列表格式化为字符串，添加到列表中
        for kernel_param, replacement_params in self.replace.items():
            replacements = ", ".join(str(param) for param in replacement_params)
            replace_list.append(f"{kernel_param} -> {replacements}")

        return replace_list
```