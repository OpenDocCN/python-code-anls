# `.\pytorch\torchgen\executorch\api\custom_ops.py`

```py
# 从未来版本导入注释类型，用于类型检查
from __future__ import annotations

# 导入默认字典
from collections import defaultdict
# 导入数据类支持
from dataclasses import dataclass
# 导入类型检查
from typing import Sequence, TYPE_CHECKING

# 导入目标生成器
from torchgen import dest

# 禁止导入排序以避免循环依赖
# 导入调度器签名
from torchgen.api.types import DispatcherSignature  # usort:skip
# 导入原生函数相关方法
from torchgen.context import method_with_native_function
# 导入基础类型和派发键
from torchgen.model import BaseTy, BaseType, DispatchKey, NativeFunction, Variant
# 导入工具函数：concatMap 和 Target
from torchgen.utils import concatMap, Target

# 如果是类型检查阶段，则导入以下类型
if TYPE_CHECKING:
    # 导入ETKernelIndex类
    from torchgen.executorch.model import ETKernelIndex
    # 导入选择性构建器
    from torchgen.selective_build.selector import SelectiveBuilder


# 生成 RegisterKernelStub.cpp 文件，为自定义操作提供占位符内核。这将在模型作者端使用。
@dataclass(frozen=True)
class ComputeNativeFunctionStub:
    # 使用原生函数装饰器
    @method_with_native_function
    # 调用方法，接受一个 NativeFunction 对象，并返回一个字符串或 None
    def __call__(self, f: NativeFunction) -> str | None:
        # 如果函数变体中不包含 Variant.function，则返回 None
        if Variant.function not in f.variants:
            return None

        # 根据函数的 schema 创建调度器签名
        sig = DispatcherSignature.from_schema(
            f.func, prefix=f"wrapper_CPU_{f.func.name.overload_name}_", symint=False
        )
        # 确保签名不为空
        assert sig is not None

        # 根据函数的返回值数量决定返回值的名称
        if len(f.func.returns) == 0:
            ret_name = ""
        elif len(f.func.returns) == 1:
            # 如果函数有输出参数，则返回第一个输出参数的名称；否则返回第一个非输出参数的名称
            if f.func.arguments.out:
                ret_name = f.func.arguments.out[0].name
            else:
                ret_name = next(
                    (
                        a.name
                        for a in f.func.arguments.flat_non_out
                        if a.type == f.func.returns[0].type
                    ),
                    "",
                )
            # 如果返回值名称为空
            if not ret_name:
                # 如果返回类型是 Tensor
                if f.func.returns[0].type == BaseType(BaseTy.Tensor):
                    # 返回一个空的 Tensor
                    ret_name = "at::Tensor()"
                else:
                    # 抛出异常，无法处理此返回类型
                    raise Exception(
                        f"Can't handle this return type {f.func}"
                    )
        elif len(f.func.arguments.out) == len(f.func.returns):
            # 返回一个输出参数元组
            tensor_type = "at::Tensor &"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join([r.name for r in f.func.arguments.out])}
            )"""
        else:
            # 确保所有返回值类型均为 Tensor
            assert all(
                a.type == BaseType(BaseTy.Tensor) for a in f.func.returns
            ), f"Only support tensor returns but got {f.func.returns}"
            # 返回一个空 Tensor 元组
            tensor_type = "at::Tensor"
            comma = ", "
            ret_name = f"""::std::tuple<{comma.join([tensor_type] * len(f.func.returns))}>(
                {comma.join(["at::Tensor()" for _ in f.func.returns])}
            )"""
        
        # 构造返回语句
        ret_str = f"return {ret_name};" if len(f.func.returns) > 0 else ""
        # 返回完整的函数定义字符串
        return f"""
{sig.defn()} {{
    # 使用格式化字符串将变量 ret_str 的值插入到当前位置
    {ret_str}
    """
    生成自定义操作注册的代码，用于 dest.RegisterDispatchKey。

    :param native_functions: 一个 `NativeFunction` 序列
    :param selector: 选择性构建
    :param kernel_index: 所有操作的内核
    :param rocm: 布尔值，用于 dest.RegisterDispatchKey
    :return: 生成的 C++ 代码，用于将自定义运算符注册到 PyTorch 中
    """

    # 将内核索引转换为 BackendIndex。这是因为我们目前无法处理 ETKernelIndex。
    backend_index = kernel_index._to_backend_index()

    # 初始化静态初始化调度注册字符串为空
    static_init_dispatch_registrations = ""

    # 使用 defaultdict 将 native_functions 按命名空间分组
    ns_grouped_native_functions: dict[str, list[NativeFunction]] = defaultdict(list)
    for native_function in native_functions:
        ns_grouped_native_functions[native_function.namespace].append(native_function)

    # 遍历每个命名空间及其函数列表
    for namespace, functions in ns_grouped_native_functions.items():
        # 如果函数列表为空，则跳过当前命名空间
        if len(functions) == 0:
            continue
        
        # 将函数列表中的函数映射为注册调度键的代码字符串，并以换行符连接起来
        dispatch_registrations_body = "\n".join(
            list(
                concatMap(
                    dest.RegisterDispatchKey(
                        backend_index,
                        Target.REGISTRATION,
                        selector,
                        rocm=rocm,
                        symint=False,
                        class_method_name=None,
                        skip_dispatcher_op_registration=False,
                    ),
                    functions,
                )
            )
        )

        # 将生成的调度注册代码添加到静态初始化调度注册字符串中
        static_init_dispatch_registrations += f"""
TORCH_LIBRARY_IMPL({namespace}, {DispatchKey.CPU}, m) {{
{dispatch_registrations_body}
}};"""

    # 将所有 native_functions 映射为匿名定义的注册调度键的代码字符串，并以换行符连接起来
    anonymous_definition = "\n".join(
        list(
            concatMap(
                dest.RegisterDispatchKey(
                    backend_index,
                    Target.ANONYMOUS_DEFINITION,
                    selector,
                    rocm=rocm,
                    symint=False,
                    class_method_name=None,
                    skip_dispatcher_op_registration=False,
                ),
                native_functions,
            )
        )
    )

    # 返回匿名定义和静态初始化调度注册字符串
    return anonymous_definition, static_init_dispatch_registrations
```