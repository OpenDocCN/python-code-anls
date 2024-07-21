# `.\pytorch\torchgen\dest\ufunc.py`

```
# 引入未来版本的注释，使得 Python 2.7 中的类型提示支持使用注解
from __future__ import annotations

# 导入用于数据类的装饰器
from dataclasses import dataclass
# 导入 Sequence 类型用于声明序列
from typing import Sequence, TYPE_CHECKING

# 导入 torchgen 库中的相关模块和函数
import torchgen.api.ufunc as ufunc
from torchgen.api.translate import translate
from torchgen.api.types import (
    BaseCType,
    Binding,
    CType,
    Expr,
    NamedCType,
    opmath_t,
    scalar_t,
    StructuredImplSignature,
    VectorizedCType,
)
# 导入处理本地函数上下文的工具
from torchgen.context import with_native_function
# 导入模型相关的类和函数
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    DispatchKey,
    NativeFunctionsGroup,
    ScalarType,
    UfuncKey,
)
# 导入有序集合工具
from torchgen.utils import OrderedSet

# 如果只是类型检查，导入 UfunctorBindings 类型
if TYPE_CHECKING:
    from torchgen.api.ufunc import UfunctorBindings


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                                  CUDA STUFF
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #

# 注意：此处并未在头文件中生成调度存根的前向声明，
# 我们可以随时将其粘贴到必要的地方

# TODO: 使用 BackendIndex
# dispatch_key: DispatchKey  # 目前仅支持 CPU/CUDA


# 表示用于实现 CUDA ufuncs 的函数对象。
# 这些函数对象根据 scalar_t 进行模板化，因为在用户实例化函数对象时会进行模板化。
# 一个函数对象的示例可能如下所示：
#
#   template <typename scalar_t>
#   struct CUDAFunctorOnSelf_add {
#     using opmath_t = at::opmath_type<scalar_t>;
#     opmath_t other_;
#     opmath_t alpha_;
#     CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha)
#         : other_(other), alpha_(alpha) {}
#     __device__ scalar_t operator()(scalar_t self) {
#       return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
#     }
#   };
#
@dataclass(frozen=True)
class UfunctorSignature:
    # 表示本地函数组
    g: NativeFunctionsGroup
    # 标量张量索引，可能为空
    scalar_tensor_idx: int | None
    # 函数对象的名称
    name: str

    # 返回函数对象的参数绑定
    def arguments(self) -> UfunctorBindings:
        return ufunc.ufunctor_arguments(
            self.g, scalar_tensor_idx=self.scalar_tensor_idx, scalar_t=scalar_t
        )

    # 返回函数对象的字段列表
    def fields(self) -> list[Binding]:
        # 按照约定，将字段重命名为带下划线后缀的形式
        return [b.rename(f"{b.name}_") for b in self.arguments().ctor]

    # 返回函数对象的返回类型
    def returns_type(self) -> CType:
        # TODO: 不要硬编码；返回类型将基于本地函数的标签进行推断
        return BaseCType(scalar_t)

    # 返回函数对象字段的声明字符串
    def decl_fields(self) -> str:
        return "\n".join(f"{f.type} {f.name};" for f in self.fields())

    # 返回函数对象的内联定义构造函数字符串
    def inline_defn_ctor(self) -> str:
        args_str = ", ".join(a.decl() for a in self.arguments().ctor)
        # 注意：理论上可以使用 translate 来完成这一步骤，
        # 但这里的转换过程非常规则，因此直接实现了构造函数
        init_str = ", ".join(f"{a.name}_({a.name})" for a in self.arguments().ctor)
        return f"{self.name}({args_str}) : {init_str} {{}}"
    # 定义一个方法名为 decl_apply 的方法，返回类型为字符串
    def decl_apply(self) -> str:
        # 将参数列表中每个参数的声明字符串连接成一个逗号分隔的字符串
        args_str = ", ".join(a.decl() for a in self.arguments().apply)
        # 构造并返回一个字符串，表示重载运算符 () 的声明
        return f"{self.returns_type().cpp_type()} operator()({args_str}) const"
# 使用 Python 的 dataclass 装饰器定义一个不可变的数据类 UfuncSignature，表示通用函数的签名
@dataclass(frozen=True)
class UfuncSignature:
    # 包含 NativeFunctionsGroup 对象的字段 g，表示函数所属的原生函数组
    g: NativeFunctionsGroup
    # 字符串字段 name，表示函数的名称
    name: str
    # 表示计算类型的 CType 对象的字段 compute_t
    compute_t: CType

    # 方法 arguments() 返回一个 Binding 类型的列表，表示函数的参数绑定
    def arguments(self) -> list[Binding]:
        return ufunc.ufunc_arguments(self.g, compute_t=self.compute_t)

    # 方法 call(ctx) 接受一个 Sequence[Binding | Expr] 类型的参数 ctx，返回函数调用的字符串表示
    def call(self, ctx: Sequence[Binding | Expr]) -> str:
        # 使用 f-string 构建函数调用的字符串，其中包括函数名称和参数表达式
        return f"{self.name}({', '.join(a.expr for a in translate(ctx, self.arguments()))})"


# steps:
#   1. 取得功能签名
#   2. 使用 api.ufunc 将其转换为模板签名。这确定了模板函数的类型
#   3. 使用 api.ufunc (II) 生成分离的结构 / operator() 签名。
#      这为我们调用模板签名建立了上下文

# StructuredImplSignature 上下文
#   ~> 函数对象构造函数签名

# Functor 构造函数上下文
#   ~> 函数对象字段签名

# Functor 应用上下文（函数对象字段 + 函数对象应用签名）
#   ~> 模板签名


def eligible_for_binary_scalar_specialization(g: NativeFunctionsGroup) -> bool:
    # 计算 g.functional.func.arguments.flat_non_out 中类型为张量的参数个数
    num_tensors = sum(
        1 for a in g.functional.func.arguments.flat_non_out if a.type.is_tensor_like()
    )
    # 返回是否有两个张量参数
    return num_tensors == 2


def compute_ufunc_cuda_functors(
    g: NativeFunctionsGroup,
) -> tuple[dict[ScalarType, dict[UfuncKey, UfunctorSignature]], str]:
    # 首先，构建函数对象（functors）。
    ufunctor_sigs: dict[ScalarType, dict[UfuncKey, UfunctorSignature]] = {}
    # 初始化空列表 ufunctors，用于存储函数对象的名称
    ufunctors: list[str] = []
    # 获取 g.out.ufunc_inner_loop，表示函数对象内部循环的键值对
    loops = g.out.ufunc_inner_loop
    # 标量张量索引查找表，指定在 CUDA 函数对象上的自身和其他张量的索引
    scalar_tensor_idx_lookup = {
        UfuncKey.CUDAFunctorOnSelf: 1,
        UfuncKey.CUDAFunctorOnOther: 0,
        UfuncKey.CUDAFunctor: None,
    }
    # 如果函数符合二元标量特殊化条件
    if eligible_for_binary_scalar_specialization(g):
        # 设置 keys 为三个可能的 UfuncKey
        keys = [
            UfuncKey.CUDAFunctorOnSelf,
            UfuncKey.CUDAFunctorOnOther,
            UfuncKey.CUDAFunctor,
        ]
    else:
        # 否则，只使用单一的 UfuncKey.CUDAFunctor
        keys = [UfuncKey.CUDAFunctor]
        # 对于 UfuncKey.CUDAFunctorOnSelf 和 UfuncKey.CUDAFunctorOnOther，断言不可用于非二元函数
        for k in [UfuncKey.CUDAFunctorOnSelf, UfuncKey.CUDAFunctorOnOther]:
            assert k not in loops, f"cannot use {k} on non-binary function"
        for k in keys:
            # 如果键直接定义了，跳过函数代码生成；我们假设用户已经为我们完成了这部分工作
            if k in loops:
                # 创建一个 UfunctorSignature 对象，用于表示特定的函数签名
                ufunctor_sig = UfunctorSignature(
                    g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=loops[k].name
                )
                # 将支持的数据类型加入到对应键的字典中
                for dtype in loops[k].supported_dtypes:
                    ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig
                continue

            # 注释 [ScalarOnly and Generic must match names for CUDA]
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # 否则，查看任何通用条目。为了简化代码生成，ScalarOnly 和 Generic
            # 都已定义，ufunc 名称必须匹配（如果它们不匹配，我们将不得不为每个数据类型生成不同的函数器，这是糟糕的，所以我们不打算这样做，除非有人真的逼我们）
            ufunc_name = None
            supported_dtypes: OrderedSet[ScalarType] = OrderedSet()
            for lk in [UfuncKey.ScalarOnly, UfuncKey.Generic]:
                if lk not in loops:
                    continue
                if ufunc_name is None:
                    ufunc_name = loops[lk].name
                else:
                    # 见注释 [ScalarOnly and Generic must match names for CUDA]
                    assert (
                        ufunc_name == loops[lk].name
                    ), "ScalarOnly and Generic must have same ufunc name"
                supported_dtypes |= loops[lk].supported_dtypes
            # 确保 ufunc_name 不为 None
            assert ufunc_name is not None

            # 按照特定格式生成名称
            name = f"{k}_{ufunc_name}"
            # 创建 UfunctorSignature 对象，用于表示特定的函数签名
            ufunctor_sig = UfunctorSignature(
                g, scalar_tensor_idx=scalar_tensor_idx_lookup[k], name=name
            )
            # 将支持的数据类型加入到对应键的字典中
            for dtype in supported_dtypes:
                ufunctor_sigs.setdefault(dtype, {})[k] = ufunctor_sig

            # 创建 UfuncSignature 对象，用于表示特定的函数签名
            ufunc_sig = UfuncSignature(
                g, name=f"ufunc::{ufunc_name}", compute_t=BaseCType(opmath_t)
            )
            # 应用上下文是 ufunctor_sig 的字段和参数的应用
            apply_ctx = ufunctor_sig.fields() + ufunctor_sig.arguments().apply
            # 将生成的函数器添加到列表中
            ufunctors.append(
                f"""
template <typename scalar_t>
struct {ufunctor_sig.name} {{
  using opmath_t = at::opmath_type<scalar_t>;
  {ufunctor_sig.decl_fields()}  // 声明结构体字段，使用了模板类型scalar_t和操作类型opmath_t

  {ufunctor_sig.inline_defn_ctor()}  // 内联构造函数的定义，用于初始化结构体的成员变量

  __device__ {ufunctor_sig.decl_apply()} {{  // 在设备上定义apply函数，返回ufunc_sig.call(apply_ctx)的结果
    return {ufunc_sig.call(apply_ctx)};
  }}
}};
"""
        )

    return ufunctor_sigs, "\n".join(ufunctors)  // 返回ufunctor_sigs和ufunctors，ufunctor_sigs是一个结构体签名的字典，ufunctors是所有ufunctor的字符串连接起来

@dataclass(frozen=True)
class BinaryScalarSpecializationConfig:
    scalar_idx: int  // 标量索引，表示该配置用于哪个标量
    ctor_tensor: str  // 构造函数的张量，指示该配置构造函数使用的是self还是other
    ufunc_key: UfuncKey  // UfuncKey枚举类型的实例，表示二元标量专用配置的键


BinaryScalarSpecializationConfigs = [  // 二元标量专用配置的列表
    BinaryScalarSpecializationConfig(
        scalar_idx=0,
        ctor_tensor="self",
        ufunc_key=UfuncKey.CUDAFunctorOnOther,
    ),
    BinaryScalarSpecializationConfig(
        scalar_idx=1,
        ctor_tensor="other",
        ufunc_key=UfuncKey.CUDAFunctorOnSelf,
    ),
]


def compute_ufunc_cuda_dtype_body(
    g: NativeFunctionsGroup,
    dtype: ScalarType,
    inner_loops: dict[UfuncKey, UfunctorSignature],
    parent_ctx: Sequence[Binding],
) -> str:
    body = "using opmath_t = at::opmath_type<scalar_t>;"  // 使用模板类型scalar_t定义opmath_t类型

    body += "if (false) {}\n"  // 为了便于代码生成，插入一个假的条件语句

    for config in BinaryScalarSpecializationConfigs:
        if config.ufunc_key not in inner_loops:
            continue
        ufunctor_sig = inner_loops[config.ufunc_key]  // 根据配置键获取内部循环中对应的ufunctor签名
        scalar_idx = config.scalar_idx + 1  // 标量索引加1，用于获取标量值
        // 复制父上下文并扩展类型（不允许没有复制，不希望改变输入参数）
        ctx: list[Expr | Binding] = list(parent_ctx)  // 复制父上下文
        ctx.append(  // 将表达式和命名的C类型添加到上下文中
            Expr(
                expr=f"iter.scalar_value<opmath_t>({scalar_idx})",
                type=NamedCType(config.ctor_tensor, BaseCType(opmath_t)),
            )
        )
        ufunctor_ctor_exprs_str = ", ".join(  // 构造函数表达式字符串连接
            a.expr for a in translate(ctx, ufunctor_sig.arguments().ctor)
        )

        // 注意：在调用iter.remove_operand之前必须分配ufunctor，
        // 因为它依赖于iter
        body += f"""\
else if (iter.is_cpu_scalar({scalar_idx})) {{
  {ufunctor_sig.name}<scalar_t> ufunctor({ufunctor_ctor_exprs_str});
  iter.remove_operand({scalar_idx});
  gpu_kernel(iter, ufunctor);  // 在GPU上运行内核，传递iter和ufunctor
}}"""

    ufunctor_sig = inner_loops[UfuncKey.CUDAFunctor]
    ufunctor_ctor_exprs_str = ", ".join(  // 构造函数表达式字符串连接
        a.expr for a in translate(parent_ctx, ufunctor_sig.arguments().ctor)
    )
    body += f"""
else {{
  gpu_kernel(iter, {ufunctor_sig.name}<scalar_t>({ufunctor_ctor_exprs_str}));
}}
    """
    return body  // 返回生成的body字符串


@with_native_function
def compute_ufunc_cuda(g: NativeFunctionsGroup) -> str:
    // 首先，构建ufuncs并通过dtype索引它们
    ufunctor_sigs, ufunctors = compute_ufunc_cuda_functors(g)

    // 接下来，构建条件语句
    sig = StructuredImplSignature(g, ufunc.kernel_name(g, DispatchKey.CUDA))  // 使用CUDA分发键构建结构化实现签名
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunctor_sigs.items():
        dtype_cases.append(
            f"""
AT_DISPATCH_CASE(at::ScalarType::{dtype},
  [&]() {{
    {{compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunc_sigs, sig.arguments())}
  }}


注释：


    {{compute_ufunc_cuda_dtype_body(g, dtype, inner_ufunc_sigs, sig.arguments())}
  }}


这行代码是一个模板字符串，使用双大括号 `{{ }}` 包围，其中包含一个函数调用和一个块结束符号 `}}`。这个模板字符串可能被动态地填充和执行，具体的行为和上下文有待进一步的代码和运行时环境来确定。
# 创建一个字符串列表，包含所有的数据类型案例
dtype_cases_str = "\n".join(dtype_cases)

# 创建一个 StubSignature 对象，用于生成函数签名
stub_sig = StubSignature(g)

# 返回一个包含多行字符串的格式化字符串，用于定义 ufunctors 和相应的函数实现
return f"""
{ufunctors}

{stub_sig.type_defn()};
{stub_sig.dispatch_decl()};

{stub_sig.kernel_defn()} {{
  AT_DISPATCH_SWITCH(iter.common_dtype(), "{sig.name}",
    {dtype_cases_str}
  );
}}
REGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name});

{sig.defn()} {{
  {stub_sig.direct_call(sig.arguments())};
}}
"""
    if UfuncKey.CPUVector in inner_loops:
        # 检查是否在内部循环中存在 CPU 向量化的键
        vec_loop = inner_loops[UfuncKey.CPUVector]

    # NB: We DON'T use translate here, because translate is
    # incapable of CSE'ing the scalar accesses in case it is also
    # used by Vectorized; also, the unpacking here is very simple
    # and only affects Scalar; everything else is implicitly captured
    # by the lambda

    # Setup scalar in scope
    body = []
    ctx = []
    for b in parent_ctx:
        if isinstance(b.argument, Argument) and b.argument.type != BaseType(
            BaseTy.Scalar
        ):
            continue
        # 如果参数是标量类型，则添加标量转换的代码
        body.append(f"auto _s_{b.name} = {b.name}.to<scalar_t>();")
        ctx.append(Expr(f"_s_{b.name}", NamedCType(b.nctype.name, BaseCType(scalar_t))))
    if vec_loop is not None:
        # 如果存在 CPU 向量化的内部循环，则添加向量化转换的代码
        for b in parent_ctx:
            if isinstance(b.argument, Argument) and b.argument.type != BaseType(
                BaseTy.Scalar
            ):
                continue
            body.append(
                f"auto _v_{b.name} = at::vec::Vectorized<scalar_t>(_s_{b.name});"
            )
            ctx.append(
                Expr(
                    f"_v_{b.name}",
                    NamedCType(b.nctype.name, VectorizedCType(BaseCType(scalar_t))),
                )
            )

    # Setup lambda signature
    # NB: simplified version of ufunctor_arguments
    scalar_bindings = []
    vec_bindings = []
    for a in g.functional.func.arguments.flat_non_out:
        if not a.type.is_tensor_like():
            continue
        assert a.type == BaseType(BaseTy.Tensor)
        # 如果参数是张量类型，则添加标量和向量化绑定
        scalar_bindings.append(
            Binding(
                name=a.name,
                nctype=NamedCType(a.name, BaseCType(scalar_t)),
                argument=a,
            )
        )
        if vec_loop is not None:
            vec_bindings.append(
                Binding(
                    name=a.name,
                    nctype=NamedCType(a.name, VectorizedCType(BaseCType(scalar_t))),
                    argument=a,
                )
            )

    def with_ctx(b: Sequence[Binding]) -> list[Expr | Binding]:
        # 将上下文和参数绑定合并返回
        r: list[Expr | Binding] = []
        r.extend(ctx)
        r.extend(b)
        return r

    body_str = "\n".join(body)
    if vec_loop is not None:
        return f"""
@with_native_function
# 使用装饰器将当前函数与本地函数相关联，用于处理原生函数的分组操作
def compute_ufunc_cpu_kernel(g: NativeFunctionsGroup) -> str:
    # 创建一个存根签名对象，以处理给定的本地函数组
    stub_sig = StubSignature(g)

    # 按数据类型重新索引通用函数（ufunc）；处理通用的标量和仅标量情况
    loops = g.out.ufunc_inner_loop
    ufunc_sigs: dict[ScalarType, dict[UfuncKey, UfuncSignature]] = {}

    # 遍历可能的内部循环类型，按指定的优先顺序覆盖
    for k in [UfuncKey.CPUScalar, UfuncKey.CPUVector]:
        lks = []
        # 顺序很重要：这里指定了优先级覆盖的顺序
        if k in loops:  # 应该很少出现
            lks.append(k)
        if UfuncKey.ScalarOnly in loops and k is UfuncKey.CPUScalar:
            lks.append(UfuncKey.ScalarOnly)
        if UfuncKey.Generic in loops:
            lks.append(UfuncKey.Generic)
        
        # TODO: 不要在这里硬编码 ufunc:: 命名空间，应该集中管理
        for lk in lks:
            for dtype in loops[lk].supported_dtypes:
                compute_t: CType
                if k is UfuncKey.CPUScalar:
                    compute_t = BaseCType(scalar_t)
                elif k is UfuncKey.CPUVector:
                    compute_t = VectorizedCType(BaseCType(scalar_t))
                else:
                    raise AssertionError
                
                # 获取或设置当前数据类型和内部循环类型的通用函数签名
                inner_ufunc_sigs = ufunc_sigs.setdefault(dtype, {})
                if k not in inner_ufunc_sigs:
                    inner_ufunc_sigs[k] = UfuncSignature(
                        g, name=f"ufunc::{loops[lk].name}", compute_t=compute_t
                    )

    # 构建条件语句
    dtype_cases = []
    for dtype, inner_ufunc_sigs in ufunc_sigs.items():
        dtype_cases.append(
            f"""
            AT_DISPATCH_CASE(at::ScalarType::{dtype},
              [&]() {{
                {compute_ufunc_cpu_dtype_body(g, dtype, inner_ufunc_sigs, stub_sig.arguments())}
              }}
            )
            """
        )

    # 将数据类型情况连接为字符串
    dtype_cases_str = "\n".join(dtype_cases)
    
    # 返回生成的 C++ 代码字符串，包括匿名命名空间和调度注册
    return f"""
    namespace {{

    {stub_sig.kernel_defn()} {{
      AT_DISPATCH_SWITCH(iter.common_dtype(), "{stub_sig.name}",
        {dtype_cases_str}
      );
    }}

    }} // anonymous namespace

    {stub_sig.type_defn()};
    {stub_sig.dispatch_decl()};
    REGISTER_DISPATCH({stub_sig.name}, &{stub_sig.kernel_name});
    """
```