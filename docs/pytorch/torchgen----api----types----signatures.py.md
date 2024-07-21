# `.\pytorch\torchgen\api\types\signatures.py`

```
    """
    A CppSignature represents a single overload in the C++ API.  For
    any given function schema, there may be multiple CppSignatures
    corresponding to it, based on how we desugar to C++.  See also
    CppSignatureGroup.
    """

    # 这个类表示 C++ API 中的一个重载签名，对应于给定的函数模式，可能会有多个 CppSignatures
    # 与之对应，基于我们如何将其转换为 C++。参见 CppSignatureGroup。

    # 从这个签名派生的函数模式
    func: FunctionSchema

    # 是否是一个方法的 C++ 签名，例如 Tensor::my_op(...)?
    method: bool

    # 是否是一个忠实的 C++ 签名（即遵循 JIT 模式），还是一个便利的 API
    # （例如带有潜在的 TensorOptions 参数和前置的输出参数）
    faithful: bool

    # 是否是一个 symint 的 C++ 签名。出于兼容性原因，接受 SymInt 的函数在 C++ 中仍然
    # 表示为 int64_t，而 SymInt 变体在不同的重载名称下提供
    symint: bool

    # 一组在 C++ 参数中不应用默认值的参数
    cpp_no_default_args: set[str]

    # 是否是一个回退的 C++ 绑定？回退绑定由 manual_cpp_binding: True 启用，
    # 是一个替代的非公共 API，允许手动实现 C++ 绑定的开发者访问自动生成的绑定
    fallback_binding: bool = False

    # 返回此签名的解包参数结构，丢弃关于哪些参数在语义上相关的信息。
    def arguments(self) -> Sequence[Binding]:
        return cpp.arguments(
            self.func.arguments,
            faithful=self.faithful,
            symint=self.symint,
            method=self.method,
            cpp_no_default_args=self.cpp_no_default_args,
        )

    # 返回此签名的名称，根据需要抑制 symint 后缀
    def name(self, *, suppress_symint_suffix: bool = False) -> str:
        n = cpp.name(
            self.func,
            faithful_name_for_out_overloads=self.faithful,
            symint_overload=False if suppress_symint_suffix else self.symint,
        )
        if self.fallback_binding:
            n = f"__dispatch_{n}"
        return n

    # 渲染此签名的 C++ 声明
    def decl(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
        suppress_symint_suffix: bool = False,
    # 定义一个方法，返回此方法的签名的C++声明字符串
    def __str__(self) -> str:
        # 获取此方法返回类型的C++表示形式
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        # 获取此方法所有参数的C++声明字符串列表
        cpp_args = [a.decl() for a in self.arguments()]
        # 如果是重新调度函数，则在参数列表开头加上dispatchKeySet参数
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        # 将参数列表转换为逗号分隔的字符串形式
        cpp_args_str = ", ".join(cpp_args)
        # 如果未提供name参数，则根据prefix和方法名生成默认的方法名
        if name is None:
            name = prefix + self.name(suppress_symint_suffix=suppress_symint_suffix)
        # 返回此方法的完整C++声明字符串，不包括方法体（花括号部分）
        return f"{returns_type} {name}({cpp_args_str})"

    # 返回此方法的C++定义，不包括方法体（花括号部分）
    def defn(
        self,
        *,
        name: str | None = None,
        prefix: str = "",
        is_redispatching_fn: bool = False,
    ) -> str:
        # 获取此方法返回类型的C++表示形式
        returns_type = cpp.returns_type(
            self.func.returns, symint=self.symint
        ).cpp_type()
        # 获取此方法所有参数的C++定义字符串列表
        cpp_args = [a.defn() for a in self.arguments()]
        # 如果是重新调度函数，则在参数列表开头加上dispatchKeySet参数
        if is_redispatching_fn:
            cpp_args = ["c10::DispatchKeySet dispatchKeySet"] + cpp_args
        # 将参数列表转换为逗号分隔的字符串形式
        cpp_args_str = ", ".join(cpp_args)
        # 如果未提供name参数，则根据prefix和方法名生成默认的方法名
        if name is None:
            name = prefix + self.name()
        # 返回此方法的完整C++定义字符串，不包括方法体（花括号部分）
        return f"{returns_type} {name}({cpp_args_str})"

    # 返回此方法的指针类型的C++表示形式
    def ptr_type(self) -> str:
        # 获取此方法所有参数类型的逗号分隔字符串形式
        args_types_str = ", ".join(a.type for a in self.arguments())
        # 返回此方法的指针类型的C++表示形式，形如返回类型 (*) (参数类型列表)
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_types_str})"

    # 返回此方法的C++函数类型的字符串表示形式，形如返回类型 (参数类型列表)
    def type(self) -> str:
        # 获取此方法所有参数类型的逗号分隔字符串形式
        args_types_str = ", ".join(a.type for a in self.arguments())
        # 返回此方法的C++函数类型的字符串表示形式，形如返回类型 (参数类型列表)
        return f"{cpp.returns_type(self.func.returns, symint=self.symint).cpp_type()} ({args_types_str})"
# 定义了一个数据类 CppSignatureGroup，表示与 FunctionSchema 相关的所有 CppSignatures 组合。
# 当前包含常规的用户可见签名、以及一个无分组的“忠实”签名。
@dataclass(frozen=True)
class CppSignatureGroup:
    func: FunctionSchema  # 函数的函数模式对象
    signature: CppSignature  # 常规签名的 CppSignature 对象
    faithful_signature: CppSignature | None  # 可能为空的“忠实”签名的 CppSignature 对象
    symint_signature: CppSignature | None  # 可能为空的对称整数签名的 CppSignature 对象
    symint_faithful_signature: CppSignature | None  # 可能为空的对称整数“忠实”签名的 CppSignature 对象

    # 返回最符合预期的签名
    def most_faithful_signature(self) -> CppSignature:
        if self.faithful_signature:
            return self.faithful_signature
        else:
            return self.signature

    # 生成签名的迭代器，根据指定的 symint 参数返回不同的签名
    def signatures(self, *, symint: bool = True) -> Iterator[CppSignature]:
        yield self.signature  # 返回常规签名
        if self.faithful_signature:
            yield self.faithful_signature  # 返回“忠实”签名
        if symint:
            if self.symint_signature:
                yield self.symint_signature  # 返回对称整数签名
            if self.symint_faithful_signature:
                yield self.symint_faithful_signature  # 返回对称整数“忠实”签名

    # 从本地函数创建 CppSignatureGroup 对象的静态方法
    @staticmethod
    def from_native_function(
        f: NativeFunction, *, method: bool, fallback_binding: bool = False
    ) -> CppSignatureGroup:
        func = f.func

        # 创建 CppSignature 对象的函数
        def make_sig(*, faithful: bool, symint: bool) -> CppSignature:
            return CppSignature(
                func=func,
                faithful=faithful,
                symint=symint,
                method=method,
                fallback_binding=fallback_binding,
                cpp_no_default_args=f.cpp_no_default_args,
            )

        # 创建签名的元组的函数
        def make_sigs(*, symint: bool) -> tuple[CppSignature, CppSignature | None]:
            faithful_signature: CppSignature | None = None
            # 如果函数包含 tensor_options 或有输出参数，则创建“忠实”签名
            if func.arguments.tensor_options is not None or len(func.arguments.out) > 0:
                faithful_signature = make_sig(faithful=True, symint=symint)
            signature = make_sig(faithful=False, symint=symint)
            return signature, faithful_signature

        # 创建常规签名及其“忠实”签名
        signature, faithful_signature = make_sigs(symint=False)
        symint_signature: CppSignature | None = None
        symint_faithful_signature: CppSignature | None = None
        # 如果函数具有对称整数特性，则创建对应的对称整数签名及其“忠实”签名
        if func.has_symint():
            symint_signature, symint_faithful_signature = make_sigs(symint=True)

        # 返回 CppSignatureGroup 对象
        return CppSignatureGroup(
            func=func,
            signature=signature,
            faithful_signature=faithful_signature,
            symint_signature=symint_signature,
            symint_faithful_signature=symint_faithful_signature,
        )


# 定义了一个数据类 DispatcherSignature，表示派发器签名对象
@dataclass(frozen=True)
class DispatcherSignature:
    func: FunctionSchema  # 派发器签名对象关联的函数模式

    # 允许在签名名称前添加任意前缀，用于生成包装器的代码生成部分，以避免命名冲突
    prefix: str = ""

    symint: bool = True  # 是否包含对称整数特性

    # 返回与签名相关的参数列表
    def arguments(self) -> list[Binding]:
        return dispatcher.arguments(self.func, symint=self.symint)
    # 返回函数的名称，由前缀和调度器返回的函数名称组成
    def name(self) -> str:
        return self.prefix + dispatcher.name(self.func)

    # 返回函数的声明字符串，包括函数名和参数列表
    def decl(self, name: str | None = None) -> str:
        # 生成参数列表的字符串表示，每个参数通过空格分隔
        args_str = ", ".join(a.decl() for a in self.arguments())
        # 如果未提供函数名，则使用默认的函数名称
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    # 返回函数的定义字符串，包括函数名和参数列表
    def defn(
        self, name: str | None = None, *, is_redispatching_fn: bool = False
    ) -> str:
        # 生成参数列表的定义字符串表示
        args = [a.defn() for a in self.arguments()]
        # 如果是重新调度的函数，则添加额外的参数
        if is_redispatching_fn:
            args = ["c10::DispatchKeySet dispatchKeySet"] + args
        args_str = ", ".join(args)
        # 如果未提供函数名，则使用默认的函数名称
        if name is None:
            name = self.name()
        return f"{self.returns_type().cpp_type()} {name}({args_str})"

    # 返回表达式列表，由函数参数生成的表达式对象组成
    def exprs(self) -> list[Expr]:
        return [Expr(a.name, a.nctype) for a in self.arguments()]

    # 返回函数的返回类型，使用调度器确定返回类型
    def returns_type(self) -> CType:
        return dispatcher.returns_type(self.func.returns, symint=self.symint)

    # 返回函数指针的类型字符串表示
    def ptr_type(self) -> str:
        # 生成函数指针类型的字符串表示，包括返回类型和参数类型列表
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} (*)({dispatcher_args_types_str})"

    # 返回函数的类型字符串表示，包括返回类型和参数类型列表
    # 示例：int(bool), 表示返回类型为int，参数类型为bool
    def type(self) -> str:
        dispatcher_args_types_str = ", ".join(a.type for a in self.arguments())
        return f"{self.returns_type().cpp_type()} ({dispatcher_args_types_str})"

    # 根据函数模式生成一个调度器签名对象
    @staticmethod
    def from_schema(
        func: FunctionSchema, *, prefix: str = "", symint: bool = True
    ) -> DispatcherSignature:
        return DispatcherSignature(func, prefix, symint)
# 定义一个名为 NativeSignature 的数据类，使用 @dataclass 装饰器进行声明，实例对象不可变
@dataclass(frozen=True)
class NativeSignature:
    # 表示此签名派生自的函数模式
    func: FunctionSchema

    # 标志是否为符号整数类型
    symint: bool

    # 可选的前缀字符串，默认为空字符串
    prefix: str = ""

    # 返回由前缀和函数名组成的字符串
    def name(self) -> str:
        return self.prefix + native.name(self.func)

    # 返回包含函数声明的字符串
    def decl(self, name: str | None = None) -> str:
        # 获取参数声明字符串，格式为逗号分隔的参数列表
        args_str = ", ".join(a.decl() for a in self.arguments())
        if name is None:
            name = self.name()
        # 返回完整的函数声明，格式为返回类型 + 函数名 + 参数列表
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    # 返回包含函数定义的字符串
    def defn(self, name: str | None = None) -> str:
        # 获取参数定义字符串，格式为逗号分隔的参数列表
        args_str = ", ".join(a.defn() for a in self.arguments())
        if name is None:
            name = self.name()
        # 返回完整的函数定义，格式为返回类型 + 函数名 + 参数列表
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} {name}({args_str})"

    # 返回函数指针类型的字符串表示
    def ptr_type(self) -> str:
        # 获取参数定义字符串，格式为逗号分隔的参数列表
        args_str = ", ".join(a.defn() for a in self.arguments())
        # 返回函数指针类型的字符串，格式为返回类型 + (*指针符号) + 参数列表
        return f"{native.returns_type(self.func.returns, symint=self.symint).cpp_type()} (*)({args_str})"

    # 返回函数参数列表
    def arguments(self) -> list[Binding]:
        # 使用 native 模块获取函数参数列表，根据 symint 参数决定是否为符号整数
        return native.arguments(self.func, symint=self.symint)

    # 返回函数返回类型
    def returns_type(self) -> CType:
        # 使用 native 模块获取函数返回类型，根据 symint 参数决定是否为符号整数
        return native.returns_type(self.func.returns, symint=self.symint)

    # 返回函数分发器表达式列表
    def dispatcher_exprs(self) -> list[Expr]:
        # 使用 translate 模块将参数列表转换为分发器表达式列表
        return translate.translate(
            self.arguments(), dispatcher.arguments(self.func), method=False
        )


# 定义一个名为 ViewInverseSignature 的数据类，使用 @dataclass 装饰器进行声明，实例对象不可变
@dataclass(frozen=True)
class ViewInverseSignature:
    # 表示此签名属于的 NativeFunctionsViewGroup 对象
    g: NativeFunctionsViewGroup

    # 返回反向函数名字符串，不包括命名空间
    def name(self) -> str:
        return functionalization.reverse_name(self.g.view, include_namespace=False)

    # 返回静态函数声明字符串
    def decl(self) -> str:
        # 获取返回类型的字符串表示
        return_type = functionalization.returns_type(self.g.view.func)
        # 获取内部参数的声明列表，用于反向函数
        decls = [
            a.decl()
            for a in functionalization.inner_arguments(
                self.g.view.func, is_reverse=True
            )
        ]
        # 返回静态函数声明，格式为 static + 返回类型 + 函数名 + 参数列表
        return f"static {return_type.cpp_type()} {self.name()}({', '.join(decls)});"


# 定义一个名为 FunctionalizationLambda 的数据类，使用 @dataclass 装饰器进行声明，实例对象不可变
@dataclass(frozen=True)
class FunctionalizationLambda:
    # 表示此 Lambda 属于的 NativeFunctionsViewGroup 对象
    g: NativeFunctionsViewGroup

    # 表示是否生成正向 Lambda 还是反向 Lambda
    is_reverse: bool
    # 返回一个表达式列表，这些表达式是函数化内核捕获的参数表达式
    def captures(self) -> list[Expr]:
        # lambda 函数位于一个遵循调度程序 API 的内核中，因此其外部上下文是调度程序的参数
        outer_ctx = dispatcher.arguments(self.g.view.func) + [
            functionalization.reapply_views_binding,
            functionalization.inverse_return_mode_binding,
        ]
        # 获取捕获参数的绑定列表
        capture_bindings = functionalization.capture_arguments(
            self.g.view.func, is_reverse=self.is_reverse
        )
        # 使用外部上下文和捕获的参数绑定，将其翻译为表达式列表
        capture_exprs = translate.translate(
            outer_ctx, capture_bindings, method=False, allow_expensive_conversions=True
        )
        return capture_exprs

    # 返回函数化内核的声明字符串
    def decl(self) -> str:
        # 获取函数化内核返回的类型
        return_type = functionalization.returns_type(self.g.view.func)
        # 构建捕获参数的字符串表示，形如 "类型 = 表达式"
        capture_str = ", ".join(
            f"{val.type.name} = {val.expr}" for val in self.captures()
        )
        # 获取外部参数的声明列表
        decls = [
            a.decl()
            for a in functionalization.outer_arguments(is_reverse=self.is_reverse)
        ]
        # 返回函数化内核的完整声明字符串，包括捕获参数和外部参数
        return f"[{capture_str}]({', '.join(decls)}) -> {return_type.cpp_type()}"

    # 返回内部调用的字符串表示
    def inner_call(self, *, reapply_views: bool | None = None) -> str:
        # 获取内部调用的名称，包括命名空间等信息
        inner_call_name = functionalization.name(
            self.g,
            is_reverse=self.is_reverse,
            include_namespace=True,
            reapply_views=reapply_views,
        )
        # 获取外部参数的上下文
        arg_ctx = functionalization.outer_arguments(is_reverse=self.is_reverse)
        # 获取捕获参数的上下文
        capture_ctx = functionalization.capture_arguments(
            self.g.view.func, is_reverse=self.is_reverse
        )
        # 将外部参数和捕获参数的上下文合并
        full_ctx = arg_ctx + capture_ctx

        # 确保视图副本不为空
        assert self.g.view_copy is not None
        # 获取内部调用的参数绑定
        call_bindings = functionalization.inner_arguments(
            self.g.view_copy.func, is_reverse=self.is_reverse
        )
        # 获取内部调用的索引，如果有的话
        maybe_index = functionalization.inner_call_index(self.g.view_copy.func)
        # 将参数绑定翻译为表达式列表
        call_exprs = [
            e.expr for e in translate.translate(full_ctx, call_bindings, method=False)
        ]
        # 构建内部调用的字符串表示，包括索引（如果存在）
        if not self.is_reverse and maybe_index is not None:
            return f'{inner_call_name}({", ".join(call_exprs)})[{maybe_index.name}];'
        else:
            return f'{inner_call_name}({", ".join(call_exprs)});'

    # 根据给定的 NativeFunctionsViewGroup 和 is_reverse 参数创建 FunctionalizationLambda 对象
    @staticmethod
    def from_func(
        g: NativeFunctionsViewGroup, *, is_reverse: bool
    ) -> FunctionalizationLambda:
        return FunctionalizationLambda(g, is_reverse)
# 使用 @dataclass 装饰器创建一个不可变数据类 StructuredImplSignature，用于表示结构化实现的签名
@dataclass(frozen=True)
class StructuredImplSignature:
    # g 属性表示本地函数组 NativeFunctionsGroup
    g: NativeFunctionsGroup
    # name 属性表示签名的名称，为字符串类型
    name: str

    # 定义 defn 方法，生成该签名的定义字符串
    def defn(self, name: str | None = None) -> str:
        # 获取所有参数的定义字符串，并用逗号分隔
        args_str = ", ".join(a.defn() for a in self.arguments())
        return f"TORCH_IMPL_FUNC({self.name})({args_str})"

    # arguments 方法返回与此签名相关的参数列表，每个参数都是 Binding 对象
    def arguments(self) -> list[Binding]:
        return structured.impl_arguments(self.g)


# Helper functions


# 定义 kernel_signature 函数，用于生成内核签名
def kernel_signature(
    f: NativeFunction, backend_index: BackendIndex, *, prefix: str = ""
) -> NativeSignature | DispatcherSignature:
    # 注意 [External Backends Follow Dispatcher API] 部分的注释
    # 内核签名针对内部后端遵循 "native" API，而针对外部后端遵循 dispatcher API
    # 详细内容见 `native.py` 中的注释，但是历史上它们在模式约定上有些小差异
    # 任何需要在两者之间转换的差异都会导致运行时成本，因此我们希望将差异尽量减少
    # 对于外部后端，我们希望强制它们使用与 Dispatcher API 直接匹配的模式编写其内核
    meta = backend_index.get_kernel(f)
    symint = meta is not None and meta.supports_symint()
    if symint:
        # 断言确保如果 meta 支持符号整数，那么函数 f 必须具有符号整数
        assert (
            f.func.has_symint()
        ), f"attempted to define symint kernel for {backend_index.dispatch_key} without SymInt in schema"
    # 如果是外部后端，则返回由 schema 生成的 DispatcherSignature
    if backend_index.external:
        return DispatcherSignature.from_schema(f.func, prefix=prefix, symint=symint)
    else:
        # 对于内部后端，返回由函数对象和其他参数生成的 NativeSignature
        return NativeSignature(f.func, prefix=prefix, symint=symint)


# Functions only, no types
# 导入 torchgen.api 中的一系列模块，用于后续功能实现
from torchgen.api import (
    cpp,
    dispatcher,
    functionalization,
    native,
    structured,
    translate,
)
```