# `.\pytorch\torchgen\executorch\model.py`

```py
# 表示 Executorch 模型使用的所有内核。
# 维护一个 Dict[OperatorName, Dict[ETKernelKey, BackendMetadata]] 结构。

from __future__ import annotations

import itertools  # 导入 itertools 库，用于迭代操作
from collections import defaultdict, namedtuple  # 导入 defaultdict 和 namedtuple 用于创建默认字典和命名元组
from dataclasses import dataclass  # 导入 dataclass 用于创建数据类
from enum import IntEnum  # 导入 IntEnum 用于创建枚举类型

from torchgen.model import (
    BackendIndex,  # 导入 BackendIndex 类
    BackendMetadata,  # 导入 BackendMetadata 类
    DispatchKey,  # 导入 DispatchKey 类
    NativeFunction,  # 导入 NativeFunction 类
    NativeFunctionsGroup,  # 导入 NativeFunctionsGroup 类
    OperatorName,  # 导入 OperatorName 类
)
from torchgen.utils import assert_never  # 导入 assert_never 函数用于断言终止

KERNEL_KEY_VERSION = 1  # 设置 KERNEL_KEY_VERSION 常量为 1


# TODO: 从 codegen.tool.gen_oplist 中移除声明，因为是重复的子集
class ScalarType(IntEnum):
    Byte = 0
    Char = 1
    Short = 2
    Int = 3
    Long = 4
    Float = 6
    Double = 7
    Bool = 11
    # 定义枚举 ScalarType，包含不同的数据类型和对应的整数值


ETParsedYaml = namedtuple("ETParsedYaml", ["native_functions", "kernel_index"])
# 创建命名元组 ETParsedYaml，包含 native_functions 和 kernel_index 两个字段


@dataclass(frozen=True)
class ETKernelKeyOpArgMeta:
    arg_name: str  # 参数名称
    dtype: str  # 数据类型
    dim_order: tuple[int, ...]  # 如果是 Tensor，表示维度顺序的元组

    def to_native_string(self) -> str:
        # 将参数元数据转换为本地字符串表示形式
        dtype_str = ScalarType[self.dtype].value  # 获取数据类型的枚举值
        dim_str = str(self.dim_order)[1:-1].replace(" ", "")  # 将维度顺序转换为字符串形式
        return f"{dtype_str};{dim_str}"  # 返回拼接后的字符串表示


@dataclass(frozen=True)
class ETKernelKey:
    arg_meta: tuple[ETKernelKeyOpArgMeta, ...] = ()  # 参数元数据元组，默认为空
    default: bool = False  # 是否作为通用内核的指示器，默认为 False
    version: int = KERNEL_KEY_VERSION  # 版本号，默认为 KERNEL_KEY_VERSION

    @staticmethod
    def gen_from_yaml(
        args: dict[str, tuple[str, str]],  # 参数字典，键为参数名称，值为元组 (数据类型, 维度顺序)
        type_alias_map: dict[str, list[str]],  # 类型别名映射，键为类型别名，值为类型列表
        dim_order_alias_map: dict[str, list[int]],  # 维度顺序别名映射，键为别名，值为维度顺序列表
        # TODO: 支持未包装的字符串值
    ) -> list[ETKernelKey]:
        """Generate ETKernelKeys from arg kernel specs
        Multiple ETKernelKeys are returned due to dtype permutations from utilizing
        type_alias_map (actualizing each potential type permutation as a KernelKey)

        Args:
            args: Mapping from argument name to kernel specs
                Kernel specs are a tuple of (dtype, dim_order).
                Currently tuple entries must be aliased via the alias map arguments
            type_alias_map: Mapping from type alias to potential type enums
                i.e { T0 : [Double, Int] } means T0 can be either Double or Int
                Used for lookup by args
            dim_order_alias_map: Mapping from alias to a list of dimension orders
                Used for lookup by args
        """
        # Cast to dim order to int
        dim_order_alias_map = {
            k: [int(alias) for alias in v] for k, v in dim_order_alias_map.items()
        }
        # Initialize an empty list to store generated kernel keys
        kernel_keys = []

        # Get all used Dtype Alias
        dtype_alias_used = set()
        # Iterate over argument values to enforce type alias usage
        for type_alias, dim_order in args.values():
            # Enforce usage of alias initially
            # TODO: Support inlined arguments
            assert type_alias in type_alias_map, "Undefined type alias: " + str(
                type_alias
            )
            assert (
                dim_order in dim_order_alias_map
            ), "Undefined dim_order alias: " + str(dim_order)
            dtype_alias_used.add(type_alias)

        # Generate all permutations of dtype alias values
        alias_dtypes = [
            [(alias, dtype) for dtype in type_alias_map[alias]]
            for alias in dtype_alias_used
        ]
        alias_permutations = [
            dict(permutation) for permutation in list(itertools.product(*alias_dtypes))
        ]

        # Using each alias value permutation, generate kernel keys
        op_arg_cache = {}
        for permutation in alias_permutations:
            arg_list = []
            # Iterate over arguments and their specifications
            for arg_name, arg_spec in args.items():
                dtype = permutation[arg_spec[0]]
                dim_order = dim_order_alias_map[arg_spec[1]]  # type: ignore[assignment]
                # Check if the cache key exists; if not, create a new ETKernelKeyOpArgMeta
                if (
                    cache_key := (arg_name, dtype, tuple(dim_order))
                ) not in op_arg_cache:
                    op_arg_cache[cache_key] = ETKernelKeyOpArgMeta(*cache_key)  # type: ignore[arg-type]

                arg_list.append(op_arg_cache[cache_key])
            # Append a new ETKernelKey constructed from arg_list to kernel_keys
            kernel_keys.append(ETKernelKey(tuple(arg_list)))

        return kernel_keys

    def to_native_string(self) -> str:
        # Return "default" if self.default is True; otherwise, construct a string representation
        if self.default:
            return "default"
        return (
            "v"
            + str(KERNEL_KEY_VERSION)
            + "/"
            + "|".join([arg.to_native_string() for arg in self.arg_meta])
        )
    @dataclass(frozen=True)
    class ETKernelIndex:
        # 定义一个数据类 ETKernelIndex，包含一个 index 字典，其键是 OperatorName，值是另一个字典，其键是 ETKernelKey，值是 BackendMetadata
        index: dict[OperatorName, dict[ETKernelKey, BackendMetadata]]

        # 判断是否存在与给定函数 g 相关的内核
        def has_kernels(self, g: NativeFunction | NativeFunctionsGroup) -> bool:
            # 调用 get_kernels 方法获取内核字典 m
            m = self.get_kernels(g)
            return m is not None

        # 获取与给定函数 g 相关的内核字典
        def get_kernels(
            self, g: NativeFunction | NativeFunctionsGroup
        ) -> dict[ETKernelKey, BackendMetadata]:
            if isinstance(g, NativeFunction):
                f = g
            elif isinstance(g, NativeFunctionsGroup):
                f = g.functional
            else:
                assert_never(g)
            # 如果函数名称不在索引中，则返回空字典
            if f.func.name not in self.index:
                return {}
            # 否则返回与函数名称对应的内核字典
            return self.index[f.func.name]

        # 从后端索引中生成 ETKernelIndex 对象
        @staticmethod
        def grow_from_backend_indices(
            kernel_index: dict[OperatorName, dict[ETKernelKey, BackendMetadata]],
            backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]],
        ) -> None:
            for dk in backend_indices:
                index = backend_indices[dk]
                for op, backend_metadata in index.items():
                    if op in kernel_index:
                        kernel_index[op][ETKernelKey(default=True)] = backend_metadata
                    else:
                        kernel_index[op] = {ETKernelKey(default=True): backend_metadata}

        # 从后端索引中生成 ETKernelIndex 对象
        @staticmethod
        def from_backend_indices(
            backend_indices: dict[DispatchKey, dict[OperatorName, BackendMetadata]]
        ) -> ETKernelIndex:
            # 创建默认字典存储内核索引
            kernel_index: dict[
                OperatorName, dict[ETKernelKey, BackendMetadata]
            ] = defaultdict(dict)
            # 调用静态方法 grow_from_backend_indices 填充内核索引
            ETKernelIndex.grow_from_backend_indices(kernel_index, backend_indices)
            return ETKernelIndex(kernel_index)

        # 将 ETKernelIndex 转换为 BackendIndex
        def _to_backend_index(self) -> BackendIndex:
            """
            WARNING: this will be deprecated once all the codegen places know how to handle ETKernelIndex.
            """
            # 创建空字典存储操作符名称与后端元数据的映射
            index: dict[OperatorName, BackendMetadata] = {}
            # 遍历当前索引中的每个操作符
            for op in self.index:
                kernel_dict = self.index[op]
                # 断言每个操作符只有一个内核
                assert (
                    len(kernel_dict.values()) == 1
                ), f"Can't convert ETKernelIndex to BackendIndex because {op} has more than one kernels. Got {kernel_dict}"
                # 将操作符名称与默认内核元数据添加到 index 字典中
                index[op] = kernel_dict.get(
                    ETKernelKey(default=True),
                    BackendMetadata(kernel="", structured=False, cpp_namespace=""),
                )
            # 返回 BackendIndex 对象
            return BackendIndex(
                dispatch_key=DispatchKey.CPU,
                use_out_as_primary=False,
                device_guard=False,
                external=False,
                index=index,
            )

        # 注意：从索引 b 中重复的 ETKernelKey 将覆盖索引 a 中的元数据
        @staticmethod
    # 定义一个函数，用于合并两个 ETKernelIndex 类型的索引对象，并返回合并后的结果
    def merge_indices(index_a: ETKernelIndex, index_b: ETKernelIndex) -> ETKernelIndex:
        # 创建一个新的 defaultdict 对象 combined，使用 index_a 的内容进行初始化
        combined = defaultdict(dict, index_a.index.copy())
    
        # 遍历 index_b 中的每个操作 op 和其对应的条目 entry
        for op, entry in index_b.index.items():
            # 遍历 entry 中的每个键 key 和其对应的元数据 metadata
            for key, metadata in entry.items():
                # 将 index_b 中的数据合并到 combined 中对应的操作 op 和键 key 下
                combined[op][key] = metadata
    
        # 返回一个新的 ETKernelIndex 对象，其索引为合并后的 combined
        return ETKernelIndex(combined)
```