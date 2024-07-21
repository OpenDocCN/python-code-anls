# `.\pytorch\torchgen\executorch\parse.py`

```
from __future__ import annotations

from collections import defaultdict, namedtuple
from typing import Any

import yaml

from torchgen.executorch.model import ETKernelIndex, ETKernelKey
from torchgen.gen import LineLoader, parse_native_yaml
from torchgen.model import (
    BackendMetadata,
    DispatchKey,
    FunctionSchema,
    NativeFunction,
    OperatorName,
)
from torchgen.utils import NamespaceHelper


# 定义一个命名元组，用于存储从 YAML 中解析出的原生函数和 ET 后端索引
ETParsedYaml = namedtuple("ETParsedYaml", ["native_functions", "et_kernel_indices"])

# native_functions.yaml 中用于确定使用哪些内核的字段
ET_FIELDS = ["kernels", "type_alias", "dim_order_alias"]


def parse_from_yaml(ei: dict[str, object]) -> dict[ETKernelKey, BackendMetadata]:
    """根据加载的 YAML 表示的内核分配信息，提取从 'kernel keys' 到 'BackendMetadata' 的映射。
    
    Args:
        ei: 字典类型，包含键 {kernels, type_alias, dim_order_alias}
            参见 ETKernelKey 以获取参数的描述
    """
    e = ei.copy()
    if (kernels := e.pop("kernels", None)) is None:
        return {}

    type_alias: dict[str, list[str]] = e.pop("type_alias", {})  # 忽略类型检查，因为变量可以为 None
    dim_order_alias: dict[str, list[str]] = e.pop("dim_order_alias", {})  # 忽略类型检查，因为变量可以为 None
    dim_order_alias.pop("__line__", None)

    kernel_mapping: dict[ETKernelKey, BackendMetadata] = {}

    for entry in kernels:  # 忽略属性定义的类型检查
        arg_meta = entry.get("arg_meta")
        if arg_meta is not None:
            arg_meta.pop("__line__")

        kernel_name = entry.get("kernel_name")
        # 使用 NamespaceHelper 根据内核名称生成命名空间
        namespace_helper = NamespaceHelper.from_namespaced_entity(
            kernel_name, max_level=3
        )
        kernel_namespace = namespace_helper.get_cpp_namespace(default="at")
        backend_metadata = BackendMetadata(
            kernel=namespace_helper.entity_name,
            structured=False,
            cpp_namespace=(kernel_namespace + "::native"),
        )

        # 从 YAML 中生成 ETKernelKey 列表
        kernel_keys = (
            [ETKernelKey((), default=True)]
            if arg_meta is None
            else ETKernelKey.gen_from_yaml(arg_meta, type_alias, dim_order_alias)  # 忽略参数类型检查
        )

        for kernel_key in kernel_keys:
            assert kernel_key not in kernel_mapping, (
                "Duplicate kernel key: " + str(kernel_key) + " " + str(e)
            )
            kernel_mapping[kernel_key] = backend_metadata

    return kernel_mapping


def parse_et_yaml_struct(es: object) -> ETKernelIndex:
    """根据加载的 YAML 表示的操作符列表，提取每个操作符的 'kernel keys' 到 'BackendMetadata' 的映射。
    """
    indices: dict[OperatorName, dict[ETKernelKey, BackendMetadata]] = {}
    # 遍历列表 es 中的每个元素 ei
    for ei in es:  # type: ignore[attr-defined]
        # 复制 ei 的内容到 e
        e = ei.copy()

        # 从 e 中弹出键为 "func" 的值，并确保其为字符串类型
        funcs = e.pop("func")
        assert isinstance(funcs, str), f"not a str: {funcs}"
        
        # 从 funcs 创建一个 NamespaceHelper 对象，最大层级为 1
        namespace_helper = NamespaceHelper.from_namespaced_entity(
            namespaced_entity=funcs, max_level=1
        )
        
        # 解析函数名称并获取操作名称
        opname = FunctionSchema.parse(namespace_helper.entity_name).name

        # 确保操作名称不在 indices 中，避免重复
        assert opname not in indices, f"Duplicate func found in yaml: {opname} already"

        # 从 e 中解析索引信息，并将结果存储在 index 中
        if len(index := parse_from_yaml(e)) != 0:
            indices[opname] = index

    # 返回包含索引信息的 ETKernelIndex 对象
    return ETKernelIndex(indices)
# 给定一个表示操作符列表的加载了的 YAML 对象，提取与操作符名称索引相关的内核键相关字段。
def extract_kernel_fields(es: object) -> dict[OperatorName, dict[str, Any]]:
    """给定一个加载了操作符列表的 YAML 对象，提取与操作符名称索引相关的内核键相关字段。"""
    # 使用 defaultdict 创建一个空的嵌套字典，用于存储操作符名称到字段字典的映射
    fields: dict[OperatorName, dict[str, Any]] = defaultdict(dict)
    # 遍历 es 中的每个条目
    for ei in es:  # type: ignore[attr-defined]
        # 获取当前条目的 "func" 字段
        funcs = ei.get("func")
        # 断言该字段为字符串类型，否则抛出异常
        assert isinstance(funcs, str), f"not a str: {funcs}"
        # 根据函数名创建 NamespaceHelper 对象，最大嵌套层级为 1
        namespace_helper = NamespaceHelper.from_namespaced_entity(
            namespaced_entity=funcs, max_level=1
        )
        # 解析函数模式，获取操作符名称
        opname = FunctionSchema.parse(namespace_helper.entity_name).name

        # 遍历预定义的 ET_FIELDS 列表
        for field in ET_FIELDS:
            # 如果当前条目中存在该字段，则将其值存入 fields 字典对应的操作符名称下的字段中
            if (value := ei.get(field)) is not None:
                fields[opname][field] = value

    # 返回包含操作符名称索引相关内核字段的字典
    return fields


# 解析 native_functions.yaml 文件，返回 NativeFunctions 列表和操作符名称到字段字典的元组
def parse_et_yaml(
    path: str,
    tags_yaml_path: str,
    ignore_keys: set[DispatchKey] | None = None,
    skip_native_fns_gen: bool = False,
) -> tuple[list[NativeFunction], dict[OperatorName, dict[str, Any]]]:
    """解析 native_functions.yaml 文件为 NativeFunctions 列表，并返回一个操作符名称到字段字典的元组"""
    # 打开指定路径的 YAML 文件
    with open(path) as f:
        # 使用 LineLoader 加载 YAML 文件
        es = yaml.load(f, Loader=LineLoader)

    # 提取操作符名称索引相关的内核字段
    et_kernel = extract_kernel_fields(es)

    # 为了向后兼容性，从条目中移除 ET 特定字段
    strip_et_fields(es)

    # 解析 native_functions.yaml 文件，返回 NativeFunctions 列表和操作符名称到字段字典的元组
    native_yaml = parse_native_yaml(
        path,
        tags_yaml_path,
        ignore_keys,
        skip_native_fns_gen=skip_native_fns_gen,
        loaded_yaml=es,
    )
    return native_yaml.native_functions, et_kernel


# 给定一个加载了的 YAML 对象，移除每个条目中的 ET 特定字段以确保向后兼容性
def strip_et_fields(es: object) -> None:
    """给定一个加载了的 YAML 对象，移除每个条目中的 ET 特定字段以确保向后兼容性"""
    # 遍历 es 中的每个条目
    for entry in es:  # type: ignore[attr-defined]
        # 遍历预定义的 ET_FIELDS 列表
        for field in ET_FIELDS:
            # 从当前条目中移除 ET 特定字段
            entry.pop(field, None)
```