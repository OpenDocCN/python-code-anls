# `.\pytorch\torchgen\selective_build\selector.py`

```py
# 从未来模块导入 annotations，确保代码能在旧版本的 Python 中运行
from __future__ import annotations

# 导入 defaultdict 类型，用于创建默认值字典
from collections import defaultdict
# 导入 Iterable 类型，用于判断对象是否可迭代
from collections.abc import Iterable
# 导入 dataclass 装饰器，用于创建不可变数据类
from dataclasses import dataclass
# 导入 TYPE_CHECKING，用于类型检查
from typing import TYPE_CHECKING

# 导入 yaml 库，用于处理 YAML 格式数据
import yaml

# 导入 torchgen.selective_build.operator 模块中的相关函数和类
from torchgen.selective_build.operator import (
    merge_debug_info,
    merge_operator_dicts,
    SelectiveBuildOperator,
    strip_operator_overload_name,
)

# 如果是类型检查阶段，导入 NativeFunction 类型
if TYPE_CHECKING:
    from torchgen.model import NativeFunction


# SelectiveBuilder 类表示从选择性构建 YAML 规范中提取的信息
#
# 它包括构建的选择性信息、与此选择性构建相关的调试信息（不透明字符串）和应该包括在构建中的操作符集合。
#
@dataclass(frozen=True)
class SelectiveBuilder:
    # 如果为 True，则构建不是选择性的，包括所有操作符。
    include_all_operators: bool

    # 选择性/自定义构建级别的调试信息。
    _debug_info: tuple[str, ...] | None

    # 操作符到操作符元数据的字典。
    operators: dict[str, SelectiveBuildOperator]

    # 选定的内核标签和数据类型的字典。通常 PyTorch 操作符内核（函数）可能有许多为多种张量数据类型专门化的代码路径，
    # 因此不是每个内核函数一个，而是每个内核函数可能有多个。
    kernel_metadata: dict[str, list[str]]

    # ExecuTorch 专用。内核标签到（张量类似输入参数的数据类型列表的列表）的字典。
    # 这来自 selective.yaml。
    et_kernel_metadata: dict[str, list[str]]

    # 所有选定模型使用的自定义 torch 绑定类的集合。内部存储为集合以预先删除重复项，但以列表形式写入到 YAML。
    custom_classes: set[str]

    # 所有选定模型使用的构建特性的集合。内部存储为集合以预先删除重复项，但以列表形式写入到 YAML。
    build_features: set[str]

    # 如果为 True，则包括所有内核函数的所有数据类型片段以及所有自定义类。
    # 当任何一个操作符列表不是基于跟踪的选择性构建机制生成时，通常设置此标志。
    include_all_non_op_selectives: bool

    @staticmethod
    def get_nop_selector() -> SelectiveBuilder:
        # 返回一个包含 include_all_operators 为 True 的 SelectiveBuilder 实例。
        return SelectiveBuilder.from_yaml_dict({"include_all_operators": True})

    @staticmethod
    # 从一个 YAML 字典数据构建一个 SelectiveBuilder 对象
    def from_yaml_dict(data: dict[str, object]) -> SelectiveBuilder:
        # 定义有效的顶层键集合
        valid_top_level_keys = {
            "include_all_non_op_selectives",
            "include_all_operators",
            "debug_info",
            "operators",
            "kernel_metadata",
            "et_kernel_metadata",
            "custom_classes",
            "build_features",
        }
        # 获取实际顶层键的集合
        top_level_keys = set(data.keys())
        # 如果有未知的顶层键，抛出异常
        if len(top_level_keys - valid_top_level_keys) > 0:
            raise Exception(  # noqa: TRY002
                "Got unexpected top level keys: {}".format(
                    ",".join(top_level_keys - valid_top_level_keys),
                )
            )
        
        # 获取是否包括所有运算符的设置，并确保其为布尔类型
        include_all_operators = data.get("include_all_operators", False)
        assert isinstance(include_all_operators, bool)

        # 处理 debug_info，将其转换为元组类型
        debug_info = None
        if "debug_info" in data:
            di_list = data["debug_info"]
            assert isinstance(di_list, list)
            debug_info = tuple(str(x) for x in di_list)

        # 处理 operators，将其转换为 SelectiveBuildOperator 对象的字典
        operators = {}
        operators_dict = data.get("operators", {})
        assert isinstance(operators_dict, dict)
        for k, v in operators_dict.items():
            operators[k] = SelectiveBuildOperator.from_yaml_dict(k, v)

        # 处理 kernel_metadata，将其转换为字符串类型的列表字典
        kernel_metadata = {}
        kernel_metadata_dict = data.get("kernel_metadata", {})
        assert isinstance(kernel_metadata_dict, dict)
        for k, v in kernel_metadata_dict.items():
            kernel_metadata[str(k)] = [str(dtype) for dtype in v]

        # 处理 et_kernel_metadata，确保其为字典类型
        et_kernel_metadata = data.get("et_kernel_metadata", {})
        assert isinstance(et_kernel_metadata, dict)

        # 处理 custom_classes，确保其为可迭代类型，并转换为集合
        custom_classes = data.get("custom_classes", [])
        assert isinstance(custom_classes, Iterable)
        custom_classes = set(custom_classes)

        # 处理 build_features，确保其为可迭代类型，并转换为集合
        build_features = data.get("build_features", [])
        assert isinstance(build_features, Iterable)
        build_features = set(build_features)

        # 获取是否包括所有非操作选择性的设置，并确保其为布尔类型
        include_all_non_op_selectives = data.get("include_all_non_op_selectives", False)
        assert isinstance(include_all_non_op_selectives, bool)

        # 返回 SelectiveBuilder 对象，传入相应的参数
        return SelectiveBuilder(
            include_all_operators,
            debug_info,
            operators,
            kernel_metadata,
            et_kernel_metadata,
            custom_classes,  # type: ignore[arg-type]
            build_features,  # type: ignore[arg-type]
            include_all_non_op_selectives,
        )

    # 从 YAML 字符串内容构建 SelectiveBuilder 对象
    @staticmethod
    def from_yaml_str(config_contents: str) -> SelectiveBuilder:
        # 解析 YAML 字符串内容
        contents = yaml.safe_load(config_contents)
        # 调用 from_yaml_dict 方法构建 SelectiveBuilder 对象
        return SelectiveBuilder.from_yaml_dict(contents)

    # 从 YAML 文件路径构建 SelectiveBuilder 对象
    @staticmethod
    def from_yaml_path(config_path: str) -> SelectiveBuilder:
        # 打开 YAML 文件并解析其内容
        with open(config_path) as f:
            contents = yaml.safe_load(f)
            # 调用 from_yaml_dict 方法构建 SelectiveBuilder 对象
            return SelectiveBuilder.from_yaml_dict(contents)

    # 从旧版操作注册允许列表构建 SelectiveBuilder 对象
    @staticmethod
    def from_legacy_op_registration_allow_list(
        allow_list: set[str], is_root_operator: bool, is_used_for_training: bool
    ):
    ) -> SelectiveBuilder:
        # 初始化一个空字典用于存放操作符信息
        operators = {}
        # 遍历允许列表中的操作符，为每个操作符创建一个包含相关信息的字典条目
        for op in allow_list:
            operators[op] = {
                "name": op,
                "is_root_operator": is_root_operator,
                "is_used_for_training": is_used_for_training,
                "include_all_overloads": True,
            }
        # 使用从YAML字典创建SelectiveBuilder对象，并传入操作符信息和设置
        return SelectiveBuilder.from_yaml_dict(
            {
                "operators": operators,
                "include_all_non_op_selectives": True,
            }
        )

    def is_operator_selected(self, name: str) -> bool:
        # 如果标记为包含所有操作符，则返回True
        if self.include_all_operators:
            return True

        # 如果操作符在操作符字典中，则返回True；否则检查其是否为重载操作符名并再次检查
        if name in self.operators:
            return True
        name = strip_operator_overload_name(name)
        return name in self.operators and self.operators[name].include_all_overloads

    def is_native_function_selected(self, func: NativeFunction) -> bool:
        # 获取本地函数的操作符名，并调用is_operator_selected方法进行检查
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected(op_name)

    def is_operator_selected_for_training(self, name: str) -> bool:
        # 如果操作符未被选中，则返回False
        if not self.is_operator_selected(name):
            return False
        # 如果标记为包含所有操作符，则返回True
        if self.include_all_operators:
            return True

        # 创建一个非训练操作符对象，用于比较操作符的训练属性
        not_training_op = SelectiveBuildOperator(
            name="",
            is_root_operator=False,
            is_used_for_training=False,
            include_all_overloads=False,
            _debug_info=None,
        )
        op = not_training_op
        # 如果操作符在操作符字典中，则使用该操作符对象
        if name in self.operators:
            op = self.operators[name]

        name = strip_operator_overload_name(name)
        base_op = not_training_op
        # 如果操作符名在操作符字典中，则使用该操作符对象
        if name in self.operators:
            base_op = self.operators[name]

        # 返回操作符是否用于训练的判断结果
        return op.is_used_for_training or (
            base_op.include_all_overloads and base_op.is_used_for_training
        )

    def is_native_function_selected_for_training(self, func: NativeFunction) -> bool:
        # 获取本地函数的操作符名，并调用is_operator_selected_for_training方法进行检查
        op_name = op_name_from_native_function(func)
        return self.is_operator_selected_for_training(op_name)

    def is_root_operator(self, name: str) -> bool:
        # 如果操作符未被选中，则返回False
        if not self.is_operator_selected(name):
            return False
        # 如果标记为包含所有操作符，则返回True
        if self.include_all_operators:
            return True

        # 如果操作符在操作符字典中，则返回其是否为根操作符的属性
        if name in self.operators:
            op: SelectiveBuildOperator = self.operators[name]
            return op.is_root_operator
        name = strip_operator_overload_name(name)
        # 如果操作符名不在操作符字典中，则检查其是否包含所有重载并且为根操作符
        if name not in self.operators:
            return False
        base_op: SelectiveBuildOperator = self.operators[name]
        return base_op.include_all_overloads and base_op.is_root_operator

    def is_kernel_dtype_selected(self, kernel_tag: str, dtype: str) -> bool:
        # 如果标记为包含所有操作符或所有非操作选择性，则返回True
        if self.include_all_operators or self.include_all_non_op_selectives:
            return True

        # 检查内核标签和数据类型是否在内核元数据中
        return (
            kernel_tag in self.kernel_metadata
            and dtype in self.kernel_metadata[kernel_tag]
        )
    # 返回一个列表，其中包含涵盖所使用操作的内核键列表
    def et_get_selected_kernels(self, op_name: str, kernel_key: list[str]) -> list[str]:
        """
        Return a list of kernel keys that cover the used ops
        """
        # 如果没有内核元数据，要么是由 include_all_operators=True 隐含了，要么是该操作未被使用。
        if op_name not in self.et_kernel_metadata:
            return kernel_key if self.include_all_operators else []
        
        # 否则，只返回特定的内核键。
        result_set = set()

        for model_kernel_keys in self.et_kernel_metadata[op_name]:
            key_found = False
            for key in kernel_key:
                # 暂时不比较版本
                if (
                    key != "default"
                    and key.split("/")[1] == model_kernel_keys.split("/")[1]
                ):
                    result_set.add(key)
                    key_found = True
                    break
            if not key_found:
                if "default" not in kernel_key:
                    raise Exception("Missing kernel for the model")  # noqa: TRY002
                else:
                    result_set.add("default")

        return list(result_set)

    # 将对象转换为字典表示
    def to_dict(self) -> dict[str, object]:
        ret: dict[str, object] = {
            "include_all_non_op_selectives": self.include_all_non_op_selectives,
            "include_all_operators": self.include_all_operators,
        }
        operators = {}
        for op_name, op in self.operators.items():
            operators[op_name] = op.to_dict()
        ret["operators"] = operators

        if self._debug_info is not None:
            ret["debug_info"] = sorted(self._debug_info)

        # 对内核元数据进行排序后放入字典
        ret["kernel_metadata"] = {
            k: sorted(v) for (k, v) in self.kernel_metadata.items()
        }

        ret["et_kernel_metadata"] = self.et_kernel_metadata

        # 对自定义类进行排序后放入字典
        ret["custom_classes"] = sorted(self.custom_classes)

        # 对构建特性进行排序后放入字典
        ret["build_features"] = sorted(self.build_features)

        return ret
# 合并两个字典 `lhs` 和 `rhs` 中的数据，返回一个新的字典 `kernel_metadata`
def merge_kernel_metadata(
    lhs: dict[str, list[str]],
    rhs: dict[str, list[str]],
) -> dict[str, list[str]]:
    # 初始化一个空的字典 `kernel_metadata`
    kernel_metadata: dict[str, list[str]] = {}
    
    # 遍历 `lhs` 和 `rhs` 字典中的每个键值对
    for tag_name, dtypes in list(lhs.items()) + list(rhs.items()):
        # 创建 `dtypes` 的副本，转换为集合类型
        dtypes_copy = set(dtypes)
        
        # 如果 `tag_name` 已经在 `kernel_metadata` 中存在
        if tag_name in kernel_metadata:
            # 将 `kernel_metadata` 中对应键的值也转换为集合，与当前 `dtypes_copy` 合并
            dtypes_copy |= set(kernel_metadata[tag_name])

        # 将合并后的 `dtypes_copy` 转换回列表，更新 `kernel_metadata` 中的值
        kernel_metadata[tag_name] = list(dtypes_copy)

    # 返回合并后的 `kernel_metadata` 字典
    return kernel_metadata


# 合并两个字典 `lhs` 和 `rhs` 中的数据，返回一个新的字典 `et_kernel_metadata`
def merge_et_kernel_metadata(
    lhs: dict[str, list[str]],
    rhs: dict[str, list[str]],
) -> dict[str, list[str]]:
    # 使用 defaultdict 创建一个 `merge_et_kernel_metadata` 字典，其值为集合类型的默认字典
    merge_et_kernel_metadata: dict[str, set[str]] = defaultdict(set)
    
    # 遍历 `lhs` 和 `rhs` 字典中的所有键
    for op in list(lhs.keys()) + list(rhs.keys()):
        # 更新 `merge_et_kernel_metadata[op]` 中的值，合并 `lhs[op]` 和 `rhs[op]` 的列表
        merge_et_kernel_metadata[op].update(lhs.get(op, []))
        merge_et_kernel_metadata[op].update(rhs.get(op, []))

    # 返回将所有值列表排序后的 `merge_et_kernel_metadata` 字典
    return {op: sorted(val) for op, val in merge_et_kernel_metadata.items()}


# 合并两个 SelectiveBuilder 对象 `lhs` 和 `rhs` 的数据，返回一个新的 SelectiveBuilder 对象
def combine_selective_builders(
    lhs: SelectiveBuilder, rhs: SelectiveBuilder
) -> SelectiveBuilder:
    # 判断是否包含所有运算符
    include_all_operators = lhs.include_all_operators or rhs.include_all_operators
    # 合并调试信息
    debug_info = merge_debug_info(lhs._debug_info, rhs._debug_info)
    # 合并操作符字典
    operators = merge_operator_dicts(lhs.operators, rhs.operators)
    # 合并核心元数据
    kernel_metadata = merge_kernel_metadata(lhs.kernel_metadata, rhs.kernel_metadata)
    # 合并 ET 核心元数据
    et_kernel_metadata = merge_et_kernel_metadata(
        lhs.et_kernel_metadata, rhs.et_kernel_metadata
    )
    # 判断是否包含所有非操作选择器
    include_all_non_op_selectives = (
        lhs.include_all_non_op_selectives or rhs.include_all_non_op_selectives
    )
    # 合并自定义类集合
    custom_classes = lhs.custom_classes.union(rhs.custom_classes)
    # 合并构建特性集合
    build_features = lhs.build_features.union(rhs.build_features)
    
    # 返回一个新的 SelectiveBuilder 对象，以合并后的数据初始化
    return SelectiveBuilder(
        include_all_operators,
        debug_info,
        operators,
        kernel_metadata,
        et_kernel_metadata,
        custom_classes,
        build_features,
        include_all_non_op_selectives,
    )


# 获取给定 NativeFunction 对象 `f` 的操作名称，返回格式化后的字符串
def op_name_from_native_function(f: NativeFunction) -> str:
    # 根据原始数据中的 'operator_name_with_overload' 字段，返回 `schema_string` 中第一个 '(' 之前的部分
    return f"{f.namespace}::{f.func.name}"
```