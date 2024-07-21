# `.\pytorch\test\forward_backward_compatibility\check_forward_backward_compatibility.py`

```py
import argparse
import datetime
import re
import sys
import warnings
from collections import defaultdict

import torch
from torch._C import parse_schema


# How to run this test locally:
# 1 Have two virtual environments (eg conda env), one without PyTorch installed (venv_nightly)
#   one with your local changes (venv_yours).
# In venv_nightly:
# 2. First ensure that Pytorch is uninstalled, but all prereqs are installed
# 3. Install torch nightly build with
#    `pip install --pre torch -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html`
# 4. Generate original schemas with
#    `python test/forward_backward_compatibility/dump_all_function_schemas.py --filename nightly_schemas.txt`
# Now in venv_yours:
# 5. Run this test with
#    `python test/forward_backward_compatibility/check_forward_backward_compatibility.py --existing-schemas nightly_schemas.txt`

# The date specifies how long the allowlist exclusion should apply to.
#
#   - If we NEVER give BC guarantee for an operator, you can put the
#     date arbitrarily far in the future.
#   - Otherwise, pick a date that is far enough in the future that you
#     believe you can land your diff before then.
#
# Allowlist entries can be removed after the date listed on them passes.
#
# Allowlist item format:
# [
#   0: function name regex
#   1: date until which the allowlist entry is valid
#   2: (optional) function argument regex
# ]
#
# NB: function name DOES NOT include overload name!
ALLOW_LIST = [
    ("c10_experimental", datetime.date(9999, 1, 1)),
    # Internal
    ("static", datetime.date(9999, 1, 1)),
    ("prim::ModuleDictIndex", datetime.date(9999, 1, 1)),
    ("prim::MKLDNNRelu6", datetime.date(9999, 1, 1)),
    ("prim::MKLDNNRelu6_", datetime.date(9999, 1, 1)),
    ("prim::is_ort", datetime.date(9999, 1, 1)),
    ("prim::Concat", datetime.date(9999, 1, 1)),
    ("aten::_NestedTensor_GeneralizedBMM", datetime.date(9999, 1, 1)),
    # Internal, profiler-specific ops
    ("profiler::_call_end_callbacks_on_jit_fut*", datetime.date(9999, 1, 1)),
    ("profiler::_record_function_enter", datetime.date(9999, 1, 1)),
    ("aten::_cholesky_helper", datetime.date(9999, 1, 1)),
    ("aten::_lstsq_helper", datetime.date(9999, 1, 1)),
    ("aten::_syevd_helper", datetime.date(9999, 1, 1)),
    ("aten::_linalg_solve_out_helper_", datetime.date(9999, 1, 1)),
    ("aten::select_backward", datetime.date(9999, 1, 1)),
    ("aten::lstsq", datetime.date(9999, 1, 1)),
    ("aten::lstsq.X", datetime.date(9999, 1, 1)),
    ("aten::slice_backward", datetime.date(9999, 1, 1)),
    ("aten::diagonal_backward", datetime.date(9999, 1, 1)),
    ("aten::rowwise_prune", datetime.date(9999, 1, 1)),
    ("aten::eig", datetime.date(9999, 1, 1)),
    ("aten::eig.e", datetime.date(9999, 1, 1)),
    ("aten::adaptive_avg_pool3d_backward", datetime.date(9999, 1, 1)),
    ("aten::_embedding_bag_dense_backward", datetime.date(9999, 1, 1)),
    ("aten::matrix_rank", datetime.date(9999, 1, 1)),
]


注释：
这段代码定义了一个ALLOW_LIST列表，其中包含了一系列元组，每个元组包含一个函数名的正则表达式和一个截至日期，用于指定允许的函数名列表。这些条目用于控制特定函数在未来是否应该在测试中排除。
    ("aten::matrix_rank.tol", datetime.date(9999, 1, 1)),  # Define a tuple with operation name and a far future date
    ("aten::randperm", datetime.date(9999, 1, 1)),  # Another tuple for a different operation and date
    ("aten::solve", datetime.date(9999, 1, 1)),  # Yet another tuple for another operation and date
    ("aten::solve.solution", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_solve_helper", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_convolution_nogroup", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_backward_bias", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_backward_input", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_backward_weight", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_transpose_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_transpose_backward_input", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_convolution_transpose_backward_weight", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_depthwise_convolution_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_depthwise_convolution_backward_input", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::miopen_depthwise_convolution_backward_weight", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_nested_tensor", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("prepacked::unpack_prepacked_sizes_conv2d", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("prepacked::unpack_prepacked_sizes_linear", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_symeig_helper", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::symeig", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::symeig.e", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::native_multi_head_self_attention", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_native_multi_head_self_attention", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::grid_sampler_3d_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_transform_bias_rescale_qkv", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("prim::infer_squeeze_size.dim", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("prim::infer_squeeze_size", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_weight_norm_cuda_interface", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_weight_norm_cuda_interface_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::empty.SymInt", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    # Define a comment about the following operations being temporary auxiliary ops for nested tensors
    ("aten::_reshape_nested", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_reshape_nested_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::mps_linear", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_mps_linear", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_mps_max_pool2d", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_mps_max_pool2d.out", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::mps_max_pool2d_backward", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::mps_max_pool2d_backward.out", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    # Define a comment indicating that prims shouldn't be checked
    ("prims::.*", datetime.date(9999, 1, 1)),  # Tuple with operation pattern and date
    ("aten::_flash_attention_forward", datetime.date(2023, 12, 30)),  # Tuple with operation and a specific date
    ("aten::_flash_attention_backward", datetime.date(2023, 12, 30)),  # Tuple with operation and a specific date
    ("aten::_scaled_dot_product_cudnn_attention", datetime.date(9999, 1, 1)),  # Tuple with operation and date
    ("aten::_sparse_mask_helper", datetime.date(2023, 3, 15)),
    # 定义了操作名为 aten::_sparse_mask_helper 的时间戳为 2023 年 3 月 15 日
    ("aten::_transformer_decoder_only_layer_fwd", datetime.date(9999, 1, 1)),
    # 定义了操作名为 aten::_transformer_decoder_only_layer_fwd 的时间戳为永久，用于 BetterTransformer 1.0 内部操作
    ("aten::_native_decoder_only_multi_head_attention", datetime.date(9999, 1, 1)),
    # 定义了操作名为 aten::_native_decoder_only_multi_head_attention 的时间戳为永久，用于 BetterTransformer 1.0 内部操作
    ("c10d::_allgather_base_", datetime.date(2023, 12, 30)),
    # 定义了操作名为 c10d::_allgather_base_ 的时间戳为 2023 年 12 月 30 日
    ("c10d::_reduce_scatter_base_", datetime.date(2023, 12, 30)),
    # 定义了操作名为 c10d::_reduce_scatter_base_ 的时间戳为 2023 年 12 月 30 日
    ("c10d::broadcast_", datetime.date(2023, 12, 30)),
    # 定义了操作名为 c10d::broadcast_ 的时间戳为 2023 年 12 月 30 日
    ("c10d::scatter_", datetime.date(2023, 12, 30)),
    # 定义了操作名为 c10d::scatter_ 的时间戳为 2023 年 12 月 30 日
    # 这些操作已经移至 c10d_functional 命名空间下的 Python 中
    ("aten::wait_tensor", datetime.date(9999, 1, 30)),
    # 定义了操作名为 aten::wait_tensor 的时间戳为永久
    ("aten::reduce_scatter_tensor", datetime.date(9999, 1, 30)),
    # 定义了操作名为 aten::reduce_scatter_tensor 的时间戳为永久
    ("aten::all_gather_into_tensor", datetime.date(9999, 1, 30)),
    # 定义了操作名为 aten::all_gather_into_tensor 的时间戳为永久
    ("aten::all_reduce", datetime.date(9999, 1, 30)),
    # 定义了操作名为 aten::all_reduce 的时间戳为永久
    ("aten::to_sparse.out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::to_sparse.out 的时间戳为 2023 年 12 月 31 日
    ("aten::to_sparse.sparse_dim_out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::to_sparse.sparse_dim_out 的时间戳为 2023 年 12 月 31 日
    ("aten::to_sparse_bsc.out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::to_sparse_bsc.out 的时间戳为 2023 年 12 月 31 日
    ("aten::to_sparse_bsr.out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::to_sparse_bsr.out 的时间戳为 2023 年 12 月 31 日
    ("aten::to_sparse_csc.out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::to_sparse_csc.out 的时间戳为 2023 年 12 月 31 日
    ("aten::to_sparse_csr.out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::to_sparse_csr.out 的时间戳为 2023 年 12 月 31 日
    ("aten::_structured_sparse_linear", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::_structured_sparse_linear 的时间戳为 2023 年 12 月 31 日
    ("aten::batch_norm_backward_elemt.out", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::batch_norm_backward_elemt.out 的时间戳为 2023 年 12 月 31 日
    ("aten::batch_norm_backward_elemt", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::batch_norm_backward_elemt 的时间戳为 2023 年 12 月 31 日
    ("aten::sym_constrain_range", datetime.date(2023, 12, 31)),
    # 定义了操作名为 aten::sym_constrain_range 的时间戳为 2023 年 12 月 31 日
    ("aten::_efficient_attention_forward", datetime.date(2024, 7, 1)),
    # 定义了操作名为 aten::_efficient_attention_forward 的时间戳为 2024 年 7 月 1 日
    ("aten::_efficient_attention_backward", datetime.date(2024, 7, 1)),
    # 定义了操作名为 aten::_efficient_attention_backward 的时间戳为 2024 年 7 月 1 日
    ("onednn::qconv1d_pointwise", datetime.date(2024, 12, 31)),
    # 定义了操作名为 onednn::qconv1d_pointwise 的时间戳为 2024 年 12 月 31 日
    ("onednn::qconv2d_pointwise", datetime.date(2024, 12, 31)),
    # 定义了操作名为 onednn::qconv2d_pointwise 的时间戳为 2024 年 12 月 31 日
    ("onednn::qconv3d_pointwise", datetime.date(2024, 12, 31)),
    # 定义了操作名为 onednn::qconv3d_pointwise 的时间戳为 2024 年 12 月 31 日
    ("onednn::qconv2d_pointwise.binary", datetime.date(2024, 12, 31)),
    # 定义了操作名为 onednn::qconv2d_pointwise.binary 的时间戳为 2024 年 12 月 31 日
    ("aten::_scaled_mm.out", datetime.date(2024, 12, 31)),
    # 定义了操作名为 aten::_scaled_mm.out 的时间戳为 2024 年 12 月 31 日
    ("aten::_scaled_mm", datetime.date(2024, 12, 31)),
    # 定义了操作名为 aten::_scaled_mm 的时间戳为 2024 年 12 月 31 日
    # can_cast 签名中的 BC-breaking 变更：'from' -> 'from_'
    ("aten::can_cast", datetime.date(2024, 5, 31)),
    # 定义了操作名为 aten::can_cast 的时间戳为 2024 年 5 月 31 日
# ALLOW_LIST_COMPILED 是一个列表，其中每个元素都是一个元组，元组中包含以下项：
# 1. re.compile(item[0])：编译成正则表达式的第一个元素
# 2. item[1]：日期，表示允许列表的有效日期
# 3. re.compile(item[2]) if len(item) > 2 else None：如果存在第三个元素，则编译成正则表达式；否则为 None
ALLOW_LIST_COMPILED = [
    (
        re.compile(item[0]),
        item[1],
        re.compile(item[2]) if len(item) > 2 else None,
    )
    for item in ALLOW_LIST
    if item[1] >= datetime.date.today()
]


def allow_listed(schema):
    # 遍历 ALLOW_LIST_COMPILED 列表中的每个条目
    for item in ALLOW_LIST_COMPILED:
        # 如果 schema 字符串匹配到 item[0] 编译的正则表达式
        if item[0].search(str(schema)):
            # 如果 item 的长度大于2并且 item[2] 不为 None，则返回 item[2] 编译的正则表达式是否匹配 schema
            if len(item) > 2 and item[2] is not None:
                # 如果存在第三个正则表达式参数，使用它来进行进一步匹配
                return bool(item[2].search(str(schema)))
            # 否则，只需返回 True
            return True
    # 如果没有匹配到任何条目，则返回 False
    return False


# dont_parse_list 是一个列表，其中每个元素都是一个元组，包含以下项：
# 1. 字符串：表示不应解析的模式或模式列表
# 2. datetime.date 对象：指示不解析此模式的日期
# 如果日期早于今天，则可以开始解析。
# 这些模式用于标识哪些模式在 nightly 构建中会失败。
dont_parse_list = [
    ("_TorchScriptTesting.*", datetime.date(2099, 9, 17)),
    ("test_backend", datetime.date(2099, 9, 17)),
    ("dist_c10d", datetime.date(2099, 9, 17)),
    ("__backends__.nnc", datetime.date(2099, 9, 17)),
]


def has_valid_upgraders(schema, version_map):
    # 检查给定 schema 是否存在有效的升级器
    # 获取 schema 的名称
    schema_name = schema.name

    # 如果 schema_name 不在 version_map 中，则返回 False
    if schema_name not in version_map:
        return False

    # 获取 schema_name 对应的条目
    entries = version_map[schema_name]

    # 初始化可能的重载列表和可能的模式列表
    possible_overloads = []
    possible_schemas = []

    # 遍历 entries，生成可能的重载列表和可能的模式列表
    for key, upgrader_schema_entries in entries.items():
        possible_overloads.append(key)
        possible_schemas.extend(upgrader_schema_entries)

    # 确保给定的 schema 存在于可能的模式列表中
    for old_schema in possible_schemas:
        if old_schema == schema:
            return True

    # 如果没有找到匹配的 schema，则返回 False
    return False


def dont_parse(schema_line):
    # 检查给定的 schema_line 是否在不解析列表中
    for item in dont_parse_list:
        # 如果 item[1] 的日期早于今天，则跳过此条目
        if item[1] < datetime.date.today():
            continue
        # 编译 item[0] 成为正则表达式对象
        regexp = re.compile(item[0])
        # 如果 regexp 匹配到 schema_line，则返回 True
        if regexp.search(schema_line):
            return True
    # 如果没有匹配的条目，则返回 False
    return False


def load_schemas_to_dict():
    # 获取所有 Torch 的模式和自定义类模式，并存储在 new_schema_dict 中
    new_schemas = torch._C._jit_get_all_schemas()
    new_schemas += torch._C._jit_get_custom_class_schemas()
    new_schema_dict = defaultdict(list)
    # 将模式按其名称存储在 new_schema_dict 中
    for s in new_schemas:
        new_schema_dict[s.name].append(s)
    return new_schema_dict


def process_version_map(version_map):
    # 处理版本映射，将其重新组织为更易于查找的结构
    # 初始化输出为 defaultdict(dict)
    output = defaultdict(dict)
    # 遍历 version_map 中的每个条目
    for key, entries in version_map.items():
        # 获取操作符名称
        operator_name = key.split(".")[0]
        # 解析每个条目的旧模式，并存储在 schema_entries 中
        schema_entries = [parse_schema(entry.old_schema) for entry in entries]
        # 将 schema_entries 存储在 output[operator_name][key] 中
        output[operator_name][key] = schema_entries
    return output


def check_bc(existing_schemas):
    # 检查兼容性
    # 获取所有新的模式并存储在 new_schema_dict 中
    new_schema_dict = load_schemas_to_dict()
    # 处理版本映射并存储在 version_map 中
    version_map = process_version_map(torch._C._get_operator_version_map())
    # 初始化 is_bc 为 True，broken_ops 为空列表
    is_bc = True
    broken_ops = []
    # 遍历现有模式列表中的每个模式对象
    for existing_schema in existing_schemas:
        # 如果当前模式在允许列表中，则跳过处理
        if allow_listed(existing_schema):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        # 如果当前模式有有效的升级器，则跳过处理
        if has_valid_upgraders(existing_schema, version_map):
            print("schema: ", str(existing_schema), " has valid upgrader, skipping")
            continue
        # 输出正在处理的现有模式信息
        print("processing existing schema: ", str(existing_schema))
        # 获取与当前模式名称匹配的新模式列表
        matching_new_schemas = new_schema_dict.get(existing_schema.name, [])
        found = False
        # 遍历匹配的新模式列表，查找是否存在向后兼容的模式
        for matching_new_schema in matching_new_schemas:
            if matching_new_schema.is_backward_compatible_with(existing_schema):
                found = True
                break
        # 如果未找到向后兼容的新模式，则输出未找到兼容的信息和相关候选模式
        if not found:
            print(
                "Can NOT find backward compatible schemas after changes "
                "for schema {} from the following candidates:\n[\n{}\n]".format(
                    str(existing_schema),
                    "\n\t".join(str(s) for s in matching_new_schemas),
                )
            )
            # TODO 打印更多关于为何候选模式不匹配的细节。
            # 将当前模式名称添加到已破坏的操作列表中
            broken_ops.append(str(existing_schema))
            # 设定当前模式不是向后兼容
            is_bc = False
    # 如果所有现有模式均有向后兼容的新模式，则输出向后兼容信息
    if is_bc:
        print("Found backward compatible schemas for all existing schemas")
    else:
        # 如果存在不向后兼容的操作，则输出向后不兼容的信息和已破坏的操作列表
        print(
            "The PR is introducing backward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops))
        )
    # 返回是否所有现有模式都有向后兼容的标志
    return is_bc
# 定义一个函数，用于检查现有的模式是否具有前向兼容性
def check_fc(existing_schemas):
    # 载入最新的模式信息到字典中
    new_schema_dict = load_schemas_to_dict()
    # 初始化是否具有前向兼容性的标志位
    is_fc = True
    # 用于存储不具有前向兼容性的模式列表
    broken_ops = []
    # 遍历每一个现有的模式
    for existing_schema in existing_schemas:
        # 如果该模式在允许列表中，则跳过检查
        if allow_listed(existing_schema):
            print("schema: ", str(existing_schema), " found on allowlist, skipping")
            continue
        # 打印正在处理的现有模式信息
        print("processing existing schema: ", str(existing_schema))
        # 获取与当前模式名称匹配的最新模式列表
        matching_new_schemas = new_schema_dict.get(existing_schema.name, [])
        # 标记是否找到了兼容的新模式
        found = False
        # 存储可能的兼容失败原因
        possible_failure_reasons = []
        # 遍历匹配的最新模式列表
        for matching_new_schema in matching_new_schemas:
            # 检查当前新模式是否与现有模式前向兼容
            is_compatible, reason = matching_new_schema.check_forward_compatible_with(
                existing_schema
            )
            # 如果找到了兼容的模式，则标记并退出循环
            if is_compatible:
                found = True
                break
            # 如果有具体的不兼容原因，则记录下来
            if reason != "":
                possible_failure_reasons.append(reason)
        # 如果未找到兼容的新模式
        if not found:
            # 打印未找到前向兼容的错误信息，并列出候选的新模式
            print(
                "Can NOT find forward compatible schemas after changes "
                "for schema {} from the following candidates:\n[\n{}\n]".format(
                    str(existing_schema),
                    "\n\t".join(str(s) for s in matching_new_schemas),
                )
            )
            # 打印兼容失败的具体原因列表
            print(
                "Refer to following reasons for failure "
                "to find FC schema:\n[\n{}\n]".format(
                    "\n\t".join(str(r) for r in possible_failure_reasons)
                )
            )
            # 将未兼容的模式名称添加到列表中
            broken_ops.append(str(existing_schema))
            # 设置整体检查结果为不具备前向兼容性
            is_fc = False
    # 如果所有现有模式均具有前向兼容性，则打印成功信息
    if is_fc:
        print("Found forward compatible schemas for all existing schemas")
    else:
        # 如果有不具备前向兼容性的模式存在，则发出警告信息
        warnings.warn(
            "The PR is introducing a potentially forward incompatible changes to the "
            "operator library. Please contact PyTorch team to confirm "
            "whether this change is wanted or not. \n\nBroken ops: "
            "[\n\t{}\n]".format("\n\t".join(broken_ops))
        )


if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--existing-schemas",
        help="filename to load existing schemas",
        type=str,
        default="schemas.txt",
    )
    args = parser.parse_args()
    # 初始化现有模式的字典
    existing_schema_dict = {}
    # 初始化模式列表
    slist = []
    # 打开并逐行读取指定文件中的模式信息
    with open(args.existing_schemas) as f:
        while True:
            line = f.readline()
            if not line:
                break
            # 如果该行不需要解析，则跳过
            if dont_parse(line.strip()):
                print("Not parsing schema line: ", line.strip())
                continue
            # 解析并添加有效的模式对象到列表中
            s = parse_schema(line.strip())
            slist.append(s)

    # 检查现有模式的前向兼容性
    check_fc(slist)

    # 检查是否存在后向不兼容的改变，并返回相应的退出码
    if not check_bc(slist):
        sys.exit(1)
```