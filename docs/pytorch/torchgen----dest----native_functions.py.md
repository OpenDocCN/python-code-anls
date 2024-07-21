# `.\pytorch\torchgen\dest\native_functions.py`

```
# 引入新的语言特性，即类型注解，这里是为了支持 Python 3.9 之前的版本
from __future__ import annotations

# 导入 torchgen 库中的各种模块和函数
import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe

# 使用装饰器，为函数添加本地函数和索引的上下文
@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> str | None:
    # 获取本地函数的内核签名
    sig = kernel_signature(f, backend_index)
    # 获取与本地函数关联的元数据
    metadata = backend_index.get_kernel(f)
    # 如果找不到元数据，则返回 None
    if metadata is None:
        return None
    # 如果元数据中包含 "legacy::" 字符串，则返回 None
    if "legacy::" in metadata.kernel:
        return None
    else:
        # 根据外部/内部标志选择前缀
        prefix = "static" if backend_index.external else "TORCH_API"
        # 返回格式化后的函数声明字符串
        return f"{prefix} {sig.decl(name=metadata.kernel)};"

# 使用装饰器，为函数添加本地函数组和索引的上下文
@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> list[str]:
    # 获取本地函数组的元数据名称
    meta_name = meta.name(g)
    # 获取结构化实现的参数列表
    out_args = structured.impl_arguments(g)
    # 获取与本地函数组关联的元数据
    metadata = backend_index.get_kernel(g)
    # 如果找不到元数据，则返回空列表
    if metadata is None:
        return []
    # 根据外部/内部标志选择前缀
    prefix = "" if backend_index.external else "TORCH_API "
    # 返回结构化实现的声明字符串列表
    return [
        f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
void impl({', '.join(a.decl() for a in out_args)});
}};
"""
    ]

# 生成 NativeFunctions.h 文件，包含所有实际内核定义的前向声明
@with_native_function_and_index
def compute_native_function_declaration(
    g: NativeFunctionsGroup | NativeFunction, backend_index: BackendIndex
) -> list[str]:
    # 获取本地函数或函数组的元数据
    metadata = backend_index.get_kernel(g)
    # 如果是本地函数组
    if isinstance(g, NativeFunctionsGroup):
        # 如果元数据存在且为结构化实现
        if metadata is not None and metadata.structured:
            # 对于外部后端，结构化实现尚未实现
            if backend_index.external:
                raise AssertionError(
                    "Structured external backend functions are not implemented yet."
                )
            else:
                # 生成结构化实现的声明列表
                return gen_structured(g, backend_index)
        else:
            # 对函数组中的每个函数生成非结构化实现的声明列表
            return list(
                mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions())
            )
    else:
        # 对于单个本地函数，生成非结构化实现的声明列表
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]
```