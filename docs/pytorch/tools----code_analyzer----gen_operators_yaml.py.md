# `.\pytorch\tools\code_analyzer\gen_operators_yaml.py`

```
# 使用 /usr/bin/env python3 解释器运行脚本
#!/usr/bin/env python3

# 引入未来的类型注解兼容
from __future__ import annotations

# 引入命令行解析库 argparse，JSON 序列化库 json，系统库 sys，以及类型相关的 Any
import argparse
import json
import sys
from typing import Any

# 引入 YAML 序列化与反序列化库 yaml
import yaml

# 从 gen_op_registration_allowlist 模块中导入函数和类
from gen_op_registration_allowlist import (
    canonical_name,
    gen_transitive_closure,
    load_op_dep_graph,
)

# 从 torchgen.selective_build.operator 模块中导入函数和类
from torchgen.selective_build.operator import (
    merge_operator_dicts,
    SelectiveBuildOperator,
)

# 从 torchgen.selective_build.selector 模块中导入函数
from torchgen.selective_build.selector import merge_kernel_metadata

# 生成 YAML 文件，包含用于特定 PyTorch 模型的操作符。
# ---------------------------------------------------
#
# 该二进制文件负责从 pt_operator_library() BUCK 宏调用中为每个模型生成 model_operators.yaml 文件。
#
# 输出的 YAML 文件格式:
# ------------------------
#
# <BEGIN FILE CONTENTS>
# include_all_non_op_selectives: False
# include_all_operators: False
# debug_info:
#   - model1@v100
#   - model2@v50
# operators:
#   aten::add:
#     is_root_operator: Yes
#     is_used_for_training: Yes
#     include_all_overloads: No
#     debug_info:
#       - model1@v100
#       - model2@v50
#   aten::add.int:
#     is_root_operator: No
#     is_used_for_training: No
#     include_all_overloads: Yes
# kernel_metadata:
#   add_kernel:
#     - Int8
#     - UInt32
#   sub_kernel:
#     - Int16
#     - Float
# <END FILE CONTENTS>
#
# 该应用程序的几个主要输入:
# -----------------------------------------------
#
# 1. 推理根操作符 (--root-ops): 推理用例中直接调用的根操作符（从 TorchScript 中调用）。
#
# 2. 训练根操作符 (--training-root-ops): 训练用例中使用的根操作符。当前，此列表是训练中使用的所有操作符的列表，而不仅仅是根操作符。
#    所有训练操作符也被认为是推理操作符，因此这些操作符会合并到推理操作符中。
#
# 3. 操作符依赖图 (--dep-graph-yaml-path): 用于确定哪些操作符依赖于其他操作符以正确运行的操作符依赖图的路径。
#    这用于基于静态选择性构建时根据根操作符生成模型使用的所有操作符的传递闭包。
#    对于基于跟踪的选择性构建，我们不需要执行这种传递闭包。
#
# 4. 模型元数据 (--model-name, --model-versions, --model-assets, --model-backends): 自描述。这些用于告知脚本从模型构建元数据 YAML 文件中获取哪些模型操作符列表。
#
# 5. 模型 YAML 文件 (--models-yaml-path): 这些 YAML 文件包含（对于每个模型/版本/资产/后端）用于推断和跟踪的操作符集。
#    这用于提取需要包含在构建中的实际操作符集。
#

# 函数：规范化操作符名称列表
def canonical_opnames(opnames: list[str]) -> list[str]:
    return [canonical_name(opname) for opname in opnames]

# 函数：从选项创建过滤器
def make_filter_from_options(
    model_name: str,
    # 定义一个变量 model_name，表示模型的名称，类型为字符串

    model_versions: list[str],
    # 定义一个变量 model_versions，表示模型的版本列表，每个版本为字符串

    model_assets: list[str] | None,
    # 定义一个变量 model_assets，表示模型的资产列表，每个资产为字符串或者为 None

    model_backends: list[str] | None,
    # 定义一个变量 model_backends，表示模型的后端列表，每个后端为字符串或者为 None
# 返回一个函数，该函数检查给定的模型信息是否符合指定的条件
def is_model_included(model_info) -> bool:
    # 获取模型信息中的模型数据
    model = model_info["model"]
    # 如果模型名称不匹配指定的模型名称，返回 False
    if model["name"] != model_name:
        return False
    # 如果模型版本不在指定的版本列表中，返回 False
    if str(model["version"]) not in model_versions:
        return False
    # 如果模型资产列表不为空且模型的资产不在指定的资产列表中，返回 False
    if model_assets is not None and model["asset"] not in model_assets:
        return False
    # TODO: 后续处理后端
    # 符合所有条件，返回 True
    return True


# 返回指定规则是否是新风格或旧风格的 pt_operator_library
def is_new_style_rule(model_name: str, model_versions: list[str] | None):
    # 如果模型名称和版本列表都不为空，则认为是新风格的规则
    return model_name is not None and model_versions is not None


# 验证指定的模型名称、所有指定的版本和资产是否至少出现在一个模型的 YAML 中
# 在验证失败时抛出异常，在成功时返回 None
def verify_all_specified_present(
    model_assets: list[str] | None,
    model_versions: list[str],
    selected_models_yaml: list[dict[str, Any]],
    rule_name: str,
    model_name: str,
    new_style_rule: bool,
) -> None:
    # 在指定的模型项中查找缺失的项目
    def find_missing_items(model_items, key, selected_models_yaml):
        missing_items = []
        # 如果不是新风格规则或者模型项为空，则直接返回空的缺失项目列表
        if not new_style_rule or not model_items:
            return missing_items
        # 遍历模型项，查找是否在选定的模型 YAML 中存在
        for item in model_items:
            found = False
            for model in selected_models_yaml:
                # 如果模型的指定键的字符串表示与当前项相等，则找到匹配
                if str(model["model"][key]) == item:
                    found = True
            # 如果未找到匹配，则将当前项添加到缺失项目列表中
            if not found:
                missing_items.append(item)
        # 返回缺失的项目列表
        return missing_items

    # 查找缺失的资产列表
    missing_assets = find_missing_items(model_assets, "asset", selected_models_yaml)
    # 查找缺失的版本列表
    missing_versions = find_missing_items(model_versions, "version", selected_models_yaml)
    # 如果缺少版本或资产列表中的任何一个，抛出异常
    if len(missing_versions) > 0 or len(missing_assets) > 0:  # 至少有一个缺失
        # 初始化警告消息字符串
        name_warning = ""
        # 如果选定的模型 YAML 文件数为零，生成警告消息
        if len(selected_models_yaml) == 0:
            name_warning = (
                "WARNING: 0 yaml's were found for target rule. This could be because the "
                + "provided model name: {name} is incorrect. Please check that field as well as "
                + "the assets and versions."
            ).format(name=model_name)
        # 抛出运行时异常，指明缺失的模型相关信息
        raise RuntimeError(
            (
                "Error: From the pt_operator_library rule for Rule: {name}, at least one entry for the "
                + "following fields was expected -- Model: {model_name} Expected Assets: {expected_assets}, Expected Versions: "
                + "{expected_versions}. {name_warning} In all_mobile_models.yaml either no assets were on one of the "
                + "specified versions, one of the specified assets was not present on any of the specified "
                + "versions, or both. Assets not found: {missing_assets}, Versions not found: {missing_versions} "
                + "For questions please ask in https://fb.workplace.com/groups/2148543255442743/"
            ).format(
                name=rule_name,
                model_name=model_name,
                expected_versions=model_versions,
                expected_assets=model_assets
                if model_assets
                else "<All model assets present on specified versions>",
                name_warning=name_warning,
                missing_versions=missing_versions
                if len(missing_versions) > 0
                else "<All specified versions had at least one asset>",
                missing_assets=missing_assets
                if len(missing_assets) > 0
                else "<All specified assets are present on at least 1 version>",
            )
        )
# 使用选定的模型配置，并将它们组合成一个字典，格式化为字符串，
# 将该字符串作为顶层 debug_info 放入 output 中
def create_debug_info_from_selected_models(
    output: dict[str, object],
    selected_models: list[dict],
    new_style_rule: bool,
) -> None:
    # 创建一个空的模型字典，用于存储资产信息和是否使用新风格规则的标志
    model_dict = {
        "asset_info": {},  # 映射资产名称 -> 包含哈希等资产元数据的字典
        "is_new_style_rule": new_style_rule,
    }

    # 遍历选定的模型列表
    for model in selected_models:
        # 获取模型信息字典
        model_info = model["model"]
        # 获取模型对应的资产名称和 MD5 哈希值
        asset = model_info["asset"]
        hash = model_info["md5_hash"]

        # 获取或创建特定资产名称的资产信息字典，并将当前模型的哈希值添加到列表中
        asset_info = model_dict["asset_info"].setdefault(asset, {})
        asset_info.setdefault("md5_hash", []).append(hash)

    # 将模型字典转换为 JSON 字符串，并存入 debug_info 中
    output["debug_info"] = [json.dumps(model_dict)]


def fill_output(output: dict[str, object], options: object) -> None:
    """Populate the output dict with the information required to serialize
    the YAML file used for selective build.
    """
    # 加载操作依赖图数据
    dept_graph = load_op_dep_graph(options.dep_graph_yaml_path)

    # 将模型版本和模型资产列表根据逗号拆分为字符串列表，如果为空则置为 None
    model_versions = (
        options.model_versions.split(",") if options.model_versions is not None else []
    )
    model_assets = (
        options.model_assets.split(",") if options.model_assets is not None else None
    )

    # 如果存在模型 YAML 文件路径列表，则逐一加载并存入 all_models_yaml 列表中
    all_models_yaml = []
    if options.models_yaml_path:
        for yaml_path in options.models_yaml_path:
            with open(yaml_path, "rb") as f:
                all_models_yaml.append(yaml.safe_load(f))

    # 根据选项创建模型过滤函数
    model_filter_func = make_filter_from_options(
        options.model_name, model_versions, model_assets, options.model_backends
    )

    # 过滤并获取符合条件的模型 YAML 数据
    selected_models_yaml = list(filter(model_filter_func, all_models_yaml))

    # 验证所有指定的模型资产和版本是否都存在
    verify_all_specified_present(
        model_assets=model_assets,
        model_versions=model_versions,
        selected_models_yaml=selected_models_yaml,
        rule_name=options.rule_name,
        model_name=options.model_name,
        new_style_rule=is_new_style_rule(options.model_name, options.model_versions),
    )

    # 创建 debug_info，将选定的模型 YAML 数据转换为字符串并存入 output
    create_debug_info_from_selected_models(
        output,
        selected_models_yaml,
        is_new_style_rule(options.model_name, options.model_versions),
    )

    # 如果存在根操作列表选项，则将非空的根操作添加到 static_root_ops 中
    if options.root_ops is not None:
        static_root_ops = set(filter(lambda x: len(x) > 0, options.root_ops.split(",")))
    else:
        static_root_ops = set()

    # 将静态训练根操作从逗号拆分的字符串列表中筛选出非空的操作并加入 static_training_root_ops
    static_training_root_ops = set(
        filter(
            lambda x: len(x) > 0,
            (options.training_root_ops or "").split(","),
        )
    )

    # 如果静态训练根操作集合不为空，则将其合并到 static_root_ops 中
    if len(static_training_root_ops) > 0:
        static_root_ops = static_root_ops | static_training_root_ops
    # end if

    # 初始化未展开的根操作和跟踪操作集合
    root_ops_unexpand = set()
    traced_ops = set()
    training_root_ops_unexpand = set()
    traced_training_ops = set()
    all_kernel_metadata = []
    # 初始化空集合，用于存储所有自定义类和构建特性的名称
    all_custom_classes = set()
    all_build_features = set()

    # 遍历选定的模型信息列表中的每个模型信息
    for model_info in selected_models_yaml:
        # 如果模型信息中没有指定追踪的操作符
        if "traced_operators" not in model_info:
            # 使用静态分析选择性构建方法，找到传递使用的操作符，更新静态根操作符集合
            static_root_ops = static_root_ops | set(model_info["root_operators"])
        else:
            # 使用基于追踪的选择性构建方法，更新对应的根操作符集合
            if model_info["train"]:
                # 如果设置了训练标志，则更新训练用的根操作符集合和追踪到的训练操作符集合
                training_root_ops_unexpand = training_root_ops_unexpand | set(
                    model_info["root_operators"]
                )
                traced_training_ops = traced_training_ops | set(
                    model_info["traced_operators"]
                )
            else:
                # 否则更新静态根操作符集合和追踪到的操作符集合
                root_ops_unexpand = root_ops_unexpand | set(model_info["root_operators"])
                traced_ops = traced_ops | set(model_info["traced_operators"])

        # 如果模型信息中包含内核元数据，则将其添加到所有内核元数据列表中
        if "kernel_metadata" in model_info:
            all_kernel_metadata.append(model_info["kernel_metadata"])

        # 如果模型信息中包含自定义类名称，则将其添加到所有自定义类名称集合中
        if "custom_classes" in model_info:
            all_custom_classes = all_custom_classes | set(model_info["custom_classes"])

        # 如果模型信息中包含构建特性名称，则将其添加到所有构建特性名称集合中
        if "build_features" in model_info:
            all_build_features = all_build_features | set(model_info["build_features"])

    # 下面的部分涉及静态构建中的传递闭包，仅适用于静态构建
    # 生成静态根操作符的规范名称集合
    canonical_root_ops = canonical_opnames(static_root_ops)
    # 如果存在静态根操作符的规范名称集合，则计算其传递闭包
    # 否则，将传递闭包设置为空集合
    if len(canonical_root_ops) > 0:
        closure_op_list = gen_transitive_closure(dept_graph, canonical_root_ops)
    else:
        closure_op_list = set()

    # 生成静态训练用根操作符的规范名称集合
    canonical_training_root_ops = canonical_opnames(static_training_root_ops)
    # 如果存在静态训练用根操作符的规范名称集合，则计算其传递闭包
    # 否则，将传递闭包设置为空集合
    if len(canonical_training_root_ops) > 0:
        closure_training_op_list = gen_transitive_closure(dept_graph, canonical_training_root_ops)
    else:
        closure_training_op_list = set()
    # 如果 canonical_training_root_ops 列表长度大于 0，则生成 canonical_training_root_ops 列表中所有操作符的传递闭包
    closure_training_op_list = gen_transitive_closure(
        dept_graph, canonical_training_root_ops, train=True
    )
else:
    # 如果 canonical_training_root_ops 列表为空，则设置闭包训练操作列表为空集合
    closure_training_op_list = set()

# bucketed_ops 存储了对应特定语义桶的操作符集合。例如：
#
# 1. Root Operators not used for training w/o full overload inclusion
# 2. Root Operators not used for training w/ full overload inclusion
# 3. Root Operators used for training w/o full overload inclusion
# 4. Root Operators used for training w/ full overload inclusion
# 5. Non-root Operators not used for training w/o full overload inclusion
# 等等...
#
# 基本上，对于每个布尔条件，都有两个选项（True/False）。
#
bucketed_ops = []

# START STATIC BUILD OPS
# static_root_ops_bucket 存储了静态根操作符的字典，每个操作符根据 YAML 字典生成一个 SelectiveBuildOperator 对象
static_root_ops_bucket = {}
for op_name in static_root_ops:
    op = SelectiveBuildOperator.from_yaml_dict(
        op_name,
        {
            "is_root_operator": True,
            "is_used_for_training": False,
            "include_all_overloads": not options.not_include_all_overloads_static_root_ops,
            "debug_info": [options.model_name],
        },
    )
    static_root_ops_bucket[op_name] = op
bucketed_ops.append(static_root_ops_bucket)

# closure_ops_bucket 存储了闭包操作列表的字典，每个操作符根据 YAML 字典生成一个 SelectiveBuildOperator 对象
closure_ops_bucket = {}
for op_name in closure_op_list:
    op = SelectiveBuildOperator.from_yaml_dict(
        op_name,
        {
            "is_root_operator": False,
            "is_used_for_training": False,
            "include_all_overloads": not options.not_include_all_overloads_closure_ops,
            "debug_info": [options.model_name],
        },
    )
    closure_ops_bucket[op_name] = op
bucketed_ops.append(closure_ops_bucket)

# static_training_root_ops_bucket 存储了静态训练根操作符的字典，每个操作符根据 YAML 字典生成一个 SelectiveBuildOperator 对象
static_training_root_ops_bucket = {}
for op_name in static_training_root_ops:
    op = SelectiveBuildOperator.from_yaml_dict(
        op_name,
        {
            "is_root_operator": True,
            "is_used_for_training": True,
            "include_all_overloads": True,
            "debug_info": [options.model_name],
        },
    )
    static_training_root_ops_bucket[op_name] = op
bucketed_ops.append(static_training_root_ops_bucket)

# closure_training_ops_bucket 存储了闭包训练操作列表的字典，每个操作符根据 YAML 字典生成一个 SelectiveBuildOperator 对象
closure_training_ops_bucket = {}
for op_name in closure_training_op_list:
    op = SelectiveBuildOperator.from_yaml_dict(
        op_name,
        {
            "is_root_operator": False,
            "is_used_for_training": True,
            "include_all_overloads": True,
            "debug_info": [options.model_name],
        },
    )
    closure_training_ops_bucket[op_name] = op
bucketed_ops.append(closure_training_ops_bucket)
# END STATIC BUILD OPS

# START TRACING BASED BUILD OPS
root_ops_unexpand_bucket = {}
    # 遍历未扩展的根操作符列表
    for op_name in root_ops_unexpand:
        # 使用给定的 YAML 字典创建 SelectiveBuildOperator 对象，设置为根操作符
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": False,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        # 将操作符添加到根操作符未扩展的字典中
        root_ops_unexpand_bucket[op_name] = op
    # 将根操作符未扩展的字典添加到操作符桶列表中
    bucketed_ops.append(root_ops_unexpand_bucket)

    # 初始化跟踪操作符桶
    traced_ops_bucket = {}
    # 遍历跟踪操作符列表
    for op_name in traced_ops:
        # 使用给定的 YAML 字典创建 SelectiveBuildOperator 对象，设置为非根操作符
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": False,
                "is_used_for_training": False,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        # 将操作符添加到跟踪操作符桶中
        traced_ops_bucket[op_name] = op
    # 将跟踪操作符桶添加到操作符桶列表中
    bucketed_ops.append(traced_ops_bucket)

    # 初始化训练根操作符未扩展的桶
    training_root_ops_unexpand_bucket = {}
    # 遍历训练根操作符未扩展的列表
    for op_name in training_root_ops_unexpand:
        # 使用给定的 YAML 字典创建 SelectiveBuildOperator 对象，设置为训练根操作符
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": True,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        # 将操作符添加到训练根操作符未扩展的桶中
        training_root_ops_unexpand_bucket[op_name] = op
    # 将训练根操作符未扩展的桶添加到操作符桶列表中
    bucketed_ops.append(training_root_ops_unexpand_bucket)

    # 初始化跟踪训练操作符桶
    traced_training_ops_bucket = {}
    # 遍历跟踪训练操作符列表
    for op_name in traced_training_ops:
        # 使用给定的 YAML 字典创建 SelectiveBuildOperator 对象，设置为非根操作符且用于训练
        op = SelectiveBuildOperator.from_yaml_dict(
            op_name,
            {
                "is_root_operator": False,
                "is_used_for_training": True,
                "include_all_overloads": False,
                "debug_info": [options.model_name],
            },
        )
        # 将操作符添加到跟踪训练操作符桶中
        traced_training_ops_bucket[op_name] = op
    # 将跟踪训练操作符桶添加到操作符桶列表中
    bucketed_ops.append(traced_training_ops_bucket)
    # END TRACING BASED BUILD OPS

    # 合并所有桶中的字典以去除重复的操作符
    operators: dict[str, SelectiveBuildOperator] = {}
    for ops_dict in bucketed_ops:
        operators = merge_operator_dicts(operators, ops_dict)

    # 遍历所有操作符，如果任何操作符指定需要包含所有重载，则设置 include_all_non_op_selectives 为 True
    include_all_non_op_selectives = False
    for op_name, op_info in operators.items():
        include_all_non_op_selectives = (
            include_all_non_op_selectives or op_info.include_all_overloads
        )

    # 将操作符字典转换为字典形式
    operators_as_dict = {}
    for k, v in operators.items():
        operators_as_dict[k] = v.to_dict()

    # 将结果存入输出字典中对应的字段
    output["operators"] = operators_as_dict
    output["custom_classes"] = all_custom_classes
    output["build_features"] = all_build_features
    output["include_all_non_op_selectives"] = include_all_non_op_selectives
    # 检查 all_kernel_metadata 列表的长度是否大于 0
    if len(all_kernel_metadata) > 0:
        # 如果条件成立，创建一个空的 kernel_metadata 字典
        kernel_metadata = {}
        # 遍历 all_kernel_metadata 列表中的每个元素 kt
        for kt in all_kernel_metadata:
            # 调用 merge_kernel_metadata 函数将 kernel_metadata 和当前的 kt 合并
            kernel_metadata = merge_kernel_metadata(kernel_metadata, kt)
        # 将合并后的 kernel_metadata 字典赋值给输出字典 output 的 "kernel_metadata" 键
        output["kernel_metadata"] = kernel_metadata
# 定义函数，用于向指定的参数解析器添加命令行参数
def add_arguments_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # 添加命令行选项 "--root-ops" 和 "--root_ops"
    # 用法：指定模型使用的根操作符的逗号分隔列表
    parser.add_argument(
        "--root-ops",
        "--root_ops",
        help="A comma separated list of root operators used by the model",
        required=False,
    )
    # 添加命令行选项 "--training-root-ops" 和 "--training_root_ops"
    # 用法：指定用于训练的根操作符的逗号分隔列表
    parser.add_argument(
        "--training-root-ops",
        "--training_root_ops",
        help="A comma separated list of root operators used for training",
        required=False,
    )
    # 添加命令行选项 "--output-path" 和 "--output_path"
    # 用法：指定输出 YAML 文件的位置，此选项为必需
    parser.add_argument(
        "--output-path",
        "--output_path",
        help="The location of the output yaml file.",
        required=True,
    )
    # 添加命令行选项 "--dep-graph-yaml-path" 和 "--dep_graph_yaml_path"
    # 用法：指定操作依赖图 YAML 文件的路径，此选项为必需
    parser.add_argument(
        "--dep-graph-yaml-path",
        "--dep_graph_yaml_path",
        type=str,
        help="A path to the Operator Dependency Graph YAML file.",
        required=True,
    )
    # 添加命令行选项 "--model-name" 和 "--model_name"
    # 用法：指定使用指定根操作符的模型名称，此选项为必需
    parser.add_argument(
        "--model-name",
        "--model_name",
        type=str,
        help="The name of the model that uses the specified root operators.",
        required=True,
    )
    # 添加命令行选项 "--model-versions" 和 "--model_versions"
    # 用法：指定模型版本的逗号分隔列表
    parser.add_argument(
        "--model-versions",
        "--model_versions",
        type=str,
        help="A comma separated list of model versions.",
        required=False,
    )
    # 添加命令行选项 "--model-assets" 和 "--model_assets"
    # 用法：指定模型资产名称的逗号分隔列表（如果未提供，则默认为该模型的所有资产）
    parser.add_argument(
        "--model-assets",
        "--model_assets",
        type=str,
        help="A comma separate list of model asset names (if absent, defaults to all assets for this model).",
        required=False,
    )
    # 添加命令行选项 "--model-backends" 和 "--model_backends"
    # 用法：指定模型后端的逗号分隔列表，默认为 "CPU"
    parser.add_argument(
        "--model-backends",
        "--model_backends",
        type=str,
        default="CPU",
        help="A comma separated list of model backends.",
        required=False,
    )
    # 添加命令行选项 "--models-yaml-path" 和 "--models_yaml_path"
    # 用法：指定移动模型配置 YAML 文件的路径，支持多个路径
    parser.add_argument(
        "--models-yaml-path",
        "--models_yaml_path",
        type=str,
        help="The paths to the mobile model config YAML files.",
        required=False,
        nargs="+",
    )
    # 添加命令行选项 "--include-all-operators" 和 "--include_all_operators"
    # 用法：设置此标志以请求包含所有操作符（即构建不是选择性的）
    parser.add_argument(
        "--include-all-operators",
        "--include_all_operators",
        action="store_true",
        default=False,
        help="Set this flag to request inclusion of all operators (i.e. build is not selective).",
        required=False,
    )
    # 添加命令行选项 "--rule-name" 和 "--rule_name"
    # 用法：指定生成此命令的 pt_operator_library 规则的名称
    parser.add_argument(
        "--rule-name",
        "--rule_name",
        type=str,
        help="The name of pt_operator_library rule resulting in this generation",
        required=True,
    )
    # 添加命令行选项 "--not-include-all-overloads-static-root-ops" 和 "--not_include_all_overloads_static_root_ops"
    # 用法：设置此标志以在 fill_output() 子程序中不包括静态根操作符桶中的所有重载操作符
    parser.add_argument(
        "--not-include-all-overloads-static-root-ops",
        "--not_include_all_overloads_static_root_ops",
        action="store_true",
        default=False,
        help="Set this flag to not include all overloaded operators for static root ops bucket in fill_output() subroutine",
        required=False,
    )
    parser.add_argument(
        # 添加一个命令行参数
        "--not-include-all-overloads-closure-ops",
        "--not_include_all_overloads_closure_ops",
        # 设置参数为布尔类型，存储 True 或 False
        action="store_true",
        # 默认为 False，如果不指定该参数，则为 False
        default=False,
        # 帮助信息，解释该参数的作用
        help="Set this flag to not include all overloaded operators for closure ops bucket in fill_output() subroutine",
        # 参数非必需
        required=False,
    )
    # 返回配置好命令行参数的解析器对象
    return parser
# 解析命令行参数并返回命令行选项的命名空间对象
def parse_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    return parser.parse_args()


# 获取解析器配置的命令行选项，并返回解析后的命名空间对象
def get_parser_options(parser: argparse.ArgumentParser) -> argparse.Namespace:
    # 调用添加参数配置的函数，更新解析器对象
    parser = add_arguments_parser(parser)
    # 调用解析命令行参数函数，返回解析后的选项命名空间对象
    return parse_options(parser)


# 主程序入口函数
def main(argv) -> None:
    # 创建命令行参数解析器对象，设置程序描述
    parser = argparse.ArgumentParser(description="Generate used operators YAML")
    # 获取命令行参数选项的命名空间对象
    options = get_parser_options(parser)

    # 构建模型字典
    model_dict = {
        "model_name": options.model_name,  # 设置模型名
        "asset_info": {},  # 初始化资产信息为空字典
        "is_new_style_rule": False,  # 设置新样式规则为假
    }
    # 构建输出字典
    output = {
        "debug_info": [json.dumps(model_dict)],  # 调试信息包含模型字典的 JSON 字符串
    }

    # 根据命令行选项设置是否包含所有运算符
    if options.include_all_operators:
        output["include_all_operators"] = True  # 设置包含所有运算符为真
        output["operators"] = {}  # 初始化运算符字典为空
        output["kernel_metadata"] = {}  # 初始化内核元数据字典为空
    else:
        fill_output(output, options)  # 根据选项填充输出字典

    # 打开指定路径的输出文件，写入序列化后的输出字典内容
    with open(options.output_path, "wb") as out_file:
        out_file.write(
            yaml.safe_dump(
                output,
                default_flow_style=False,
            ).encode("utf-8")
        )


# 如果作为脚本直接运行
if __name__ == "__main__":
    # 执行主程序，并使用命令行参数作为参数传递
    sys.exit(main(sys.argv))
```