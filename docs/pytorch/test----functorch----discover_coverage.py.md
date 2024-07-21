# `.\pytorch\test\functorch\discover_coverage.py`

```
import copy  # 导入 copy 模块，用于深拷贝和浅拷贝操作
import enum  # 导入 enum 枚举类型模块，用于定义枚举类
import pprint  # 导入 pprint 模块，用于漂亮打印数据结构
import unittest  # 导入 unittest 模块，用于编写和运行单元测试
from enum import Enum  # 从 enum 模块中导入 Enum 类

# Importing these files make modifications to the op_db that we need
import test_ops  # 导入 test_ops 模块，用于测试操作
import test_vmap  # 导入 test_vmap 模块，用于测试映射
from functorch_additional_op_db import additional_op_db  # 导入额外的操作数据库

import torch  # 导入 PyTorch 深度学习库
import torch._functorch.top_operators_github_usage as top_ops  # 导入 top_operators_github_usage 模块
from torch.testing._internal.common_device_type import toleranceOverride  # 导入 toleranceOverride 类
from torch.testing._internal.common_methods_invocations import op_db  # 导入 op_db 数据库

all_overridable = list(torch.overrides.get_testing_overrides().keys())  # 获取所有可重写的测试覆盖项键列表

public_docs = [  # 公共文档列表，包含元组 (模块对象, 模块名称, 源文件路径)
    (torch.nn.functional, "torch.nn.functional", "docs/source/nn.functional.rst"),
    (torch.fft, "torch.fft", "docs/source/fft.rst"),
    (torch.special, "torch.special", "docs/source/special.rst"),
    (torch.linalg, "torch.linalg", "docs/source/linalg.rst"),
    (torch, "torch", "docs/source/torch.rst"),
    (torch.Tensor, "torch.Tensor", "docs/source/tensors.rst"),
]

# torch.abs, Tensor.abs, Tensor.abs_ are all considered to be different

def get_public_overridable_apis(pytorch_root="/raid/rzou/pt/debug-cpu"):
    results = {}  # 初始化结果字典
    all_overridable_apis = set(torch.overrides.get_testing_overrides().keys())  # 获取所有可测试覆盖项的键集合
    for module, module_name, src in public_docs:  # 遍历公共文档列表
        with open(f"{pytorch_root}/{src}") as f:  # 打开源文件路径
            lines = f.readlines()  # 读取文件的所有行
        # APIs eitehr begin with 4 spaces or ".. autofunction::"
        api_lines1 = [line.strip() for line in lines if line.startswith(" " * 4)]  # 获取以四个空格开头的 API 行
        api_lines2 = [
            line.strip()[len(".. autofunction:: ") :]
            for line in lines
            if line.startswith(".. autofunction::")
        ]  # 获取以 ".. autofunction::" 开头的 API 行
        lines = api_lines1 + api_lines2  # 合并两种 API 行
        lines = [line[7:] if line.startswith("Tensor.") else line for line in lines]  # 处理以 "Tensor." 开头的 API 行
        lines = [line for line in lines if hasattr(module, line)]  # 过滤出模块中存在的 API 行
        for line in lines:
            api = getattr(module, line)  # 获取 API 对象
            if api in all_overridable_apis:  # 如果 API 在可测试覆盖项中
                results[f"{module_name}.{line}"] = api  # 将 API 添加到结果字典中
    return results  # 返回结果字典

denylist = {  # 拒绝列表，包含不希望关注的方法
    "torch.Tensor.data_ptr",
    "torch.Tensor.dim",
    "torch.Tensor.element_size",
    "torch.Tensor.backward",
    "torch.Tensor.as_strided",
    "torch.Tensor.register_hook",
    "torch.Tensor.record_stream",
    "torch.Tensor.qscheme",
    "torch.Tensor.ndimension",
    "torch.Tensor.smm",
    "torch.Tensor.sspaddmm",
    "torch.Tensor.retain_grad",
    "torch.Tensor.sparse_mask",
    "torch.Tensor.sparse_dim",
    "torch.Tensor.dense_dim",
    "torch.Tensor.values",
    "torch.Tensor.indices",
    "torch.Tensor.numel",
    "torch.Tensor.size",
    "torch.Tensor.nelement",
    "torch.Tensor.q_scale",
    "torch.Tensor.q_zero_point",
    "torch.Tensor.q_per_channel_scales",
    "torch.Tensor.q_per_channel_zero_points",
    "torch.Tensor.q_per_channel_axis",
    "torch.Tensor.int_repr",
    "torch.Tensor.to_sparse",
    "torch.Tensor.is_inference",
    "torch.Tensor.storage",
    "torch.Tensor.storage_type",
}

def get_method_only_ops_we_care_about():
    # 获取公开可重写的 API 列表
    apis = get_public_overridable_apis()
    # 初始化结果列表
    result = []
    # 遍历 API 字典的键
    for key in apis.keys():
        # 如果键不以 "torch.Tensor" 开头，则跳过当前循环
        if not key.startswith("torch.Tensor"):
            continue
        # 如果键在拒绝列表中，则跳过当前循环
        if key in denylist:
            continue
        # 获取 API 名称，通过分割键字符串获取第三部分
        api = key.split(".")[2]
        # 过滤掉末尾带下划线的 API 方法（即原地操作方法）
        # 如果 API 名称以 "_" 结尾，则跳过当前循环
        if api.endswith("_"):
            continue
        # 如果 "torch.{api}" 不在 API 字典的键中，则将 API 添加到结果列表中
        if f"torch.{api}" not in apis.keys():
            result.append(api)
    # 返回过滤后的 API 名称列表
    return result
# 获取公共的可覆盖操作集合
def get_public_overridable_ops():
    # 获取所有公共的可覆盖的 API
    results = get_public_overridable_apis()
    # 深拷贝结果，以便安全修改
    cpy = copy.deepcopy(results)
    # 遍历拷贝的键（API 名称）
    for key in cpy.keys():
        # 如果键不以 "torch.Tensor" 开头，则跳过
        if not key.startswith("torch.Tensor"):
            continue
        # 提取 API 名称的具体部分
        api = key.split(".")[2]
        # 如果 "torch.{api}" 存在于原始结果中，则从结果中删除当前键
        if f"torch.{api}" in results.keys():
            del results[key]
    # 返回处理后的结果字典
    return results


# 获取公共的可覆盖的非原地操作集合
def get_public_overridable_outplace_ops():
    # 获取所有公共的可覆盖操作集合
    results = get_public_overridable_ops()
    # 深拷贝结果，以便安全修改
    cpy = copy.deepcopy(results)
    # 遍历拷贝的键
    for key in cpy.keys():
        # 注意：我们不记录下划线结尾的双下划线方法
        if key.endswith("_"):
            del results[key]
    # 返回处理后的结果字典
    return results


# 获取我们关心的公共可覆盖的非原地操作集合
def get_public_overridable_outplace_we_care_about():
    # 获取所有公共的可覆盖非原地操作集合
    results = get_public_overridable_outplace_ops()
    # 深拷贝结果，以便安全修改
    cpy = copy.deepcopy(results)
    # 遍历拷贝的键
    for key in cpy.keys():
        # 如果键中包含 "quant" 或 ".q_"，则从结果中删除该键
        if "quant" in key or ".q_" in key:
            del results[key]
        # 如果键中包含 ".is_"，则从结果中删除该键
        if ".is_" in key:
            del results[key]
        # 如果键在 denylist 中且也在结果中，则从结果中删除该键
        if key in denylist and key in results:
            del results[key]
    # 返回处理后的结果字典
    return results


# 根据点分名称获取操作对象
def get_op(dotted_name):
    # 将点分名称拆分成名称列表
    names = dotted_name.split(".")
    # 初始化模块为 torch
    mod = torch
    # 遍历名称列表
    for name in names:
        # 如果当前模块不包含当前名称，则返回 None
        if not hasattr(mod, name):
            return None
        # 获取当前名称对应的属性或模块
        mod = getattr(mod, name)
    # 返回最终的模块或属性对象
    return mod


# 获取由 OpInfo 所覆盖的操作映射：函数 -> [OpInfo]
def get_ops_covered_by_opinfos():
    # 初始化空的操作映射字典
    ops = {}

    # 定义安全追加函数，用于向字典中的列表安全添加元素
    def safe_append(dct, key, val):
        if key in dct:
            dct[key].append(val)
        else:
            dct[key] = [val]

    # 遍历 OpInfo 列表
    for opinfo in op_db:
        # 根据 OpInfo 的名称获取操作对象
        func_op = get_op(opinfo.name)
        # 如果存在操作对象，则将 OpInfo 添加到对应操作对象的列表中
        if func_op:
            safe_append(ops, func_op, opinfo)
        # 如果存在方法变体，则也将 OpInfo 添加到方法变体对应操作对象的列表中
        if opinfo.method_variant:
            safe_append(ops, opinfo.method_variant, opinfo)
        # 如果存在原地变体，则也将 OpInfo 添加到原地变体对应操作对象的列表中
        if opinfo.inplace_variant:
            safe_append(ops, opinfo.inplace_variant, opinfo)
        # 遍历 OpInfo 的别名列表，将 OpInfo 添加到别名对应操作对象的列表中
        for alias in opinfo.aliases:
            safe_append(ops, alias.op, opinfo)
    # 返回操作映射字典
    return ops


# 工厂函数集合，用于快速创建张量和特定类型的张量
factory_fns = {
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "rand",
    "empty",
    "randperm",
    "linspace",
    "logspace",
    "hann_window",
    "full",
    "eye",
    "blackman_window",
    "bartlett_window",
    "randint",
    "range",
}


# 获取覆盖阈值以上的顶级操作集合和对应的操作计数（可选）
def get_top_ops(torch_threshold, nn_fn_threshold, with_counts=False):
    denylist = set(
        {
            # 这些是不是真正的 "operators"，而是工厂函数或未记录的操作。
            "load",  # 加载操作
            "no_grad",  # 禁用梯度操作
            "save",  # 保存操作
            "from_numpy",  # 从 numpy 转换操作
            "manual_seed",  # 手动设置随机种子操作
            "set_grad_enabled",  # 设置梯度是否可用操作
            "set_default_tensor_type",  # 设置默认张量类型操作
            "set_num_threads",  # 设置线程数操作
            "set_printoptions",  # 设置打印选项操作
            "numel",  # 张量元素数操作
            "set_default_dtype",  # 设置默认数据类型操作
            "sparse_coo_tensor",  # 稀疏张量操作
            "set_rng_state",  # 设置随机数生成器状态操作
            "get_rng_state",  # 获取随机数生成器状态操作
            "get_default_dtype",  # 获取默认数据类型操作
            "initial_seed",  # 初始种子操作
            "get_num_threads",  # 获取线程数操作
            "quantize_per_tensor",  # 对每个张量量化操作
            "hann_window",  # 汉宁窗操作
            "is_tensor",  # 判断是否为张量操作
            "as_tensor",  # 转换为张量操作
            "equal",  # 判断相等操作
            "enable_grad",  # 启用梯度操作
            "seed",  # 随机种子操作
            "is_storage",  # 判断是否为存储操作
            "is_floating_point",  # 判断是否为浮点数操作
            "nn.functional.torch",  # Torch 函数空间命名
            "set_flush_denormal",  # 设置 flush denormal 操作
            "set_num_interop_threads",  # 设置互操作线程数操作
            "dequantize",  # 反量化操作
            "get_num_interop_threads",  # 获取互操作线程数操作
            "nn.functional.math",  # 数学函数空间命名
            "nn.functional.threshold_",  # 阈值函数操作
            "nn.functional.selu_",  # SELU 函数操作
            "nn.functional.elu_",  # ELU 函数操作
            "nn.functional.rrelu_",  # RReLU 函数操作
            "nn.functional.leaky_relu_",  # Leaky ReLU 函数操作
            "nn.functional.hardtanh_",  # HardTanh 函数操作
            "nn.functional.has_torch_function",  # 是否有 Torch 函数操作
            "nn.functional.has_torch_function_unary",  # 是否有 Torch 一元函数操作
            "nn.functional.has_torch_function_variadic",  # 是否有 Torch 多元函数操作
            "nn.functional.handle_torch_function",  # 处理 Torch 函数操作
            "nn.functional.adaptive_max_pool1d_with_indices",  # 自适应最大池化操作（1D）
            "nn.functional.adaptive_max_pool2d_with_indices",  # 自适应最大池化操作（2D）
            "nn.functional.adaptive_max_pool3d_with_indices",  # 自适应最大池化操作（3D）
            "nn.functional.fractional_max_pool2d_with_indices",  # 分数最大池化操作（2D）
            "nn.functional.fractional_max_pool3d_with_indices",  # 分数最大池化操作（3D）
            "is_complex",  # 判断是否为复数操作
            "grad",  # 梯度操作
            "quantize_per_channel",  # 按通道量化操作
            "nn.functional.max_pool2d_with_indices",  # 最大池化操作（2D）
            "nn.functional.max_pool3d_with_indices",  # 最大池化操作（3D）
            "nn.functional.max_pool1d_with_indices",  # 最大池化操作（1D）
            "nn.functional.celu_",  # CELU 函数操作
            "nn.functional.grad",  # 梯度函数操作
            "nn.functional.relu_",  # ReLU 函数操作
            "nn.functional.boolean_dispatch",  # 布尔调度操作
            "nn.functional.assert_int_or_pair",  # 断言整数或对操作
            "fft",  # FFT 命名空间
        }
    )

    torch_ops = top_ops.top_torch
    nn_fn_ops = top_ops.get_nn_functional_top_list()

    # 过滤掉 denylist 中的操作，获取剩余的 Torch 操作
    torch_ops = [op for op in torch_ops if op[0] not in denylist]
    # 过滤掉 denylist 中的操作，获取剩余的 nn.functional 操作
    nn_fn_ops = [op for op in nn_fn_ops if op[0] not in denylist]

    # 将 torch_ops 和 nn_fn_ops 合并并按优先级降序排序
    ops = torch_ops[:torch_threshold] + nn_fn_ops[:nn_fn_threshold]

    # 根据优先级降序排序操作列表
    ops.sort(reverse=True, key=lambda op: op[1])

    # 如果不需要带有计数，只保留操作名
    if not with_counts:
        ops = [op[0] for op in ops]

    # 返回排序后的操作列表
    return ops
# 计算操作的覆盖率百分比
def get_ops_percentage(torch_threshold, nn_fn_threshold):
    # 获取所有 Torch 和 nn.functional 的顶级操作及其使用数据
    data = top_ops.top_torch + top_ops.get_nn_functional_top_list()

    def get_num_usages(opname):
        # 如果操作名为 "t"，则返回0，表示忽略这个特定操作
        if opname == "t":
            return 0
        # 在数据中查找指定操作名的使用数据，并确保只有一个结果
        result = [op[1] for op in data if op[0] == opname]
        assert len(result) == 1
        return result[0]

    # 获取所有操作的顶级列表
    all_ops = get_top_ops(999999, 999999)
    # 计算所有操作的总使用次数
    total_op_usages = sum(get_num_usages(op) for op in all_ops)

    # 获取子集操作的顶级列表
    subset_ops = get_top_ops(torch_threshold, nn_fn_threshold)
    # 计算子集操作的总使用次数
    subset_op_usages = sum(get_num_usages(op) for op in subset_ops)

    # 返回子集操作的使用次数占所有操作总使用次数的比例
    return subset_op_usages / total_op_usages


# 获取未被操作信息覆盖的顶级操作列表
def get_top_ops_not_covered_by_opinfo(torch_threshold=0, nn_fn_threshold=0):
    # 获取所有顶级操作
    ops = get_top_ops(torch_threshold, nn_fn_threshold)

    # 收集包含操作信息的操作列表
    ops_with_opinfo = []
    for op in op_db:
        ops_with_opinfo.append(op.name)
        ops_with_opinfo.extend([op.name for op in op.aliases])
    ops_with_opinfo = set(ops_with_opinfo)

    # 找出未被操作信息覆盖且不在拒绝列表中的操作
    result = [op for op in ops if op not in ops_with_opinfo]
    result = [op for op in result if op not in denylist]
    result = [op for op in result if op not in factory_fns]
    return result


# 获取被覆盖的操作
def get_covered_ops(ops_list, invert=False):
    # 获取被操作信息覆盖的操作
    ops_covered_by_opinfo = get_ops_covered_by_opinfos()
    overridable_outplace_ops = ops_list
    results = {}
    for key, op in overridable_outplace_ops.items():
        cond = op in ops_covered_by_opinfo
        if invert:
            cond = not cond
        if cond:
            results[key] = op
    return results


# 定义枚举类型 Status，包含 Correct 和 Fast 两个状态
class Status(Enum):
    Correct = 0
    Fast = 1


# 测试名称集合
tests = {
    "test_vmap_exhaustive",
    "test_op_has_batch_rule",
    "test_vjp",
    "test_vmapvjp",
    "test_vmapvjp_has_batch_rule",
    "test_jvp",
    "test_vmapjvp",
}


# 检查 decorateinfo 是否跳过或预期失败
def is_decorateinfo_skip_or_xfail(decorateinfo):
    assert len(decorateinfo.decorators) == 1
    actual_decorator = decorateinfo.decorators[0]
    if isinstance(actual_decorator, toleranceOverride):
        return False
    if actual_decorator == unittest.expectedFailure:
        return True
    # 假设其余情况为跳过
    return True


# 获取所有被测试的操作
def get_all_tested_ops():
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = set({})
    for op in get_covered_ops(overridable_outplace_we_care_about).values():
        opinfos = op_to_opinfo[op]
        result.update(opinfo.name for opinfo in opinfos)
    return result


# 获取指定测试名称的跳过或预期失败的操作
def get_skipped_or_xfailed_ops_for(test_name):
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = set({})
    # 遍历被指定函数操作集合中覆盖的操作
    for op in get_covered_ops(overridable_outplace_we_care_about).values():
        # 获取每个操作对应的操作信息
        opinfos = op_to_opinfo[op]
        # 遍历每个操作信息
        for opinfo in opinfos:
            # 遍历操作信息中的装饰器列表
            for decorator in opinfo.decorators:
                # 如果装饰器没有 "test_name" 属性，则继续下一个装饰器
                if not hasattr(decorator, "test_name"):
                    continue
                # 如果装饰器的测试名不等于指定的测试名，则继续下一个装饰器
                if decorator.test_name != test_name:
                    continue
                # 如果装饰器被标记为跳过或者标记为预期失败，则将操作信息的名称添加到结果集合中
                if is_decorateinfo_skip_or_xfail(decorator):
                    result.add(opinfo.name)
    # 返回最终的结果集合
    return result
def get_statuses(for_subset=None, invert=False):
    # 获取所有可以重写的公共非就地操作信息
    overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()
    
    if for_subset is not None:
        # 如果指定了子集，筛选出在子集中的操作信息（去除"torch."前缀）
        overridable_outplace_we_care_about = {
            k: v
            for k, v in overridable_outplace_we_care_about.items()
            if k[6:] in for_subset
        }
    
    # 获取所有操作与其相关的操作信息
    op_to_opinfo = get_ops_covered_by_opinfos()
    result = {}
    # 获取所有覆盖的操作
    _ = get_covered_ops(overridable_outplace_we_care_about)

    def get_covered_tests(op):
        # 获取给定操作相关的所有通过的测试集合
        opinfos = op_to_opinfo[op]
        result = copy.deepcopy(tests)
        for opinfo in opinfos:
            for decorator in opinfo.decorators:
                if not hasattr(decorator, "test_name"):
                    continue
                # 如果测试名称在测试集合中并且在结果中，则移除
                if decorator.test_name in tests and decorator.test_name in result:
                    result.remove(decorator.test_name)
        return result

    def get_all_aliases(op):
        # 获取给定操作的所有别名集合
        opinfos = op_to_opinfo[op]
        result = []
        for opinfo in opinfos:
            result.append(opinfo.name)
            result.extend(opinfo.aliases)
        return set(result)

    # 遍历覆盖的操作与其操作对象，并根据invert标志获取测试结果
    for name, op in get_covered_ops(overridable_outplace_we_care_about).items():
        successful_tests = get_covered_tests(op)
        failed_tests = tests - successful_tests
        result[name] = failed_tests if invert else successful_tests
    return result


def transpose_statuses(for_subset=None, invert=False):
    # 获取状态信息（测试通过或失败）并转置结果，返回测试到操作的映射关系
    statuses = get_statuses(for_subset, invert=invert)
    result = {}
    for test in tests:
        result[test] = set({})
    for op, supported in statuses.items():
        for test in supported:
            result[test].add(op)
    return result


# 获取所有公共可重写的API列表
overridable_apis = get_public_overridable_apis()

# 获取所有公共可重写的操作列表
overridable_ops = get_public_overridable_ops()

# 获取所有公共可重写的非就地操作列表
overridable_outplace_ops = get_public_overridable_outplace_ops()

# 获取所有我们关心的公共可重写的非就地操作信息
overridable_outplace_we_care_about = get_public_overridable_outplace_we_care_about()

# 获取经过测试的所有可重写的非就地操作
tested_overridable_outplace_ops = get_covered_ops(overridable_outplace_we_care_about)

# 获取未经过测试的所有可重写的非就地操作
untested_overridable_outplace_ops = get_covered_ops(
    overridable_outplace_we_care_about, invert=True
)

# 打印以下内容（注释掉的部分）
# print("List of OpInfos we need:")
# for key in untested_overridable_outplace_ops.keys():
#     print(key)
# print("-" * 80)
# print("")

# 打印各种数量信息
print(f"Overridable public APIs: {len(overridable_apis)}")
print(f"Overridable public ops: {len(overridable_ops)}")
print(f"Overridable public outplace ops: {len(overridable_outplace_ops)}")
print(
    f"Overridable public outplace ops we care about: {len(overridable_outplace_we_care_about)}"
)
print(
    f"OpInfo-tested overridable public outplace ops: {len(tested_overridable_outplace_ops)}"
)


def remove_torch(name):
    # 确保名称以"torch."开头并去除该前缀
    assert name[:6] == "torch."
    return name[6:]


def get_list_of_all_tests():
    # 获取所有测试名称列表并去除"torch."前缀
    all_tests = list(tested_overridable_outplace_ops.keys())
    return {remove_torch(test) for test in all_tests}


mytest = {
    "test_vmap_exhaustive",
    "test_op_has_batch_rule",  # 测试函数名称字符串 "test_op_has_batch_rule"
    "test_vjp",  # 测试函数名称字符串 "test_vjp"
    "test_vmapvjp",  # 测试函数名称字符串 "test_vmapvjp"
    "test_vmapvjp_has_batch_rule",  # 测试函数名称字符串 "test_vmapvjp_has_batch_rule"
}

# 打印一行由'*'字符组成的80个字符，用于分隔输出
print("*" * 80)

# 获得所有测试用例的列表
all_tests = get_list_of_all_tests()

# 遍历每个测试用例
for test in mytest:
    # 获取跳过或预期失败的操作结果
    result = get_skipped_or_xfailed_ops_for(test)
    # 计算未通过测试的数量
    diff = len(all_tests - result)
    # 打印测试用例名称及其未通过测试的数量
    print(f"{test}: {diff}")

# 定义函数get_jvp_coverage，用于计算JVP覆盖率
def get_jvp_coverage(subset=None):
    # 获取覆盖到的操作信息字典
    op_to_opinfo = get_ops_covered_by_opinfos()
    # 获取被测试的可覆盖非原位操作字典
    ops_dct = tested_overridable_outplace_ops
    
    # 如果subset参数不为None，筛选ops_dct中符合subset条件的操作
    if subset is not None:
        ops_dct = {
            name: op for name, op in ops_dct.items() if remove_torch(name) in subset
        }
    
    # 获取支持自动求导的操作字典
    supports_autograd_ops_dct = {
        name: op_to_opinfo[fn]
        for name, fn in ops_dct.items()
        if op_to_opinfo[fn][0].supports_autograd
    }
    
    # 获取支持forward_ad的操作字典
    supports_forwardad_ops_dct = {
        name: op_to_opinfo[fn]
        for name, fn in ops_dct.items()
        if op_to_opinfo[fn][0].supports_forward_ad
    }

    # 提取操作名称集合
    ops = {remove_torch(test) for test in list(ops_dct.keys())}
    # 提取支持自动求导操作名称集合
    supports_autograd = {
        remove_torch(test) for test in list(supports_autograd_ops_dct.keys())
    }
    # 提取支持forward_ad操作名称集合
    supports_forward_ad = {
        remove_torch(test) for test in list(supports_forwardad_ops_dct.keys())
    }
    
    # 断言支持forward_ad的操作集合应是支持自动求导的操作集合的子集
    assert supports_forward_ad.issubset(supports_autograd)
    # 断言支持自动求导的操作集合应是操作集合的子集
    assert supports_autograd.issubset(ops)

    # 获取测试名为test_jvp的未通过测试的操作集合
    failed_ops = get_skipped_or_xfailed_ops_for("test_jvp")

    # 计算覆盖支持forward_ad操作的数量
    coverage = len(supports_forward_ad - failed_ops)
    # 计算不支持forward_ad操作的数量
    no_forward_ad = len(supports_autograd) - len(supports_forward_ad)
    # 打印测试覆盖信息
    print(f"test_jvp, {coverage}, {no_forward_ad}, {len(ops)}")

# 调用get_jvp_coverage函数，计算JVP覆盖率
get_jvp_coverage()
# 调用get_jvp_coverage函数，计算部分操作的JVP覆盖率
get_jvp_coverage(get_top_ops(100, 25))

# 遍历get_top_ops函数返回的前100个操作
for op in get_top_ops(100, 25):
    # 打印每个操作的名称
    print(op)

# 打印一行由'*'字符组成的80个字符，用于分隔输出
print("*" * 80)

# 注释掉下面未使用的代码块，用于避免其执行
# result = get_skipped_or_xfailed_ops_for('test_vmap_exhaustive')
# result = get_skipped_or_xfailed_ops_for('test_op_has_batch_rule')
# result = get_skipped_or_xfailed_ops_for('test_vjp')
# result = get_skipped_or_xfailed_ops_for('test_vmapvjp')
# result = get_skipped_or_xfailed_ops_for('test_vmapvjp_has_batch_rule')
# import pdb; pdb.set_trace()

# 转置测试状态信息，获取每个测试的覆盖率
statuses = transpose_statuses()
# 遍历每个测试
for test in tests:
    # 打印每个测试及其对应的状态列表的长度
    print(f"{test} coverage {len(statuses[test])}")

# 获取关注的仅方法操作集合
method_only_ops = get_method_only_ops_we_care_about()
# 注释掉遍历method_only_ops的代码块，用于避免其执行
# for op in method_only_ops:
#     print(f'    {op},')

# 获取未被操作信息覆盖的前100个操作
top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(100, 25)
# 打印一行由'='字符组成的80个字符，用于分隔输出
print("=" * 80)
# 遍历未被操作信息覆盖的操作列表
for op in top_ops_not_covered_by_opinfo:
    # 打印每个操作及其使用计数
    print(f"{op}, {top_ops.usage_count[op]}")

# 注释掉未使用的代码块，用于避免其执行
# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(200, 50)
# for op in top_ops_not_covered_by_opinfo:
#     print(f'{op}, {top_ops.usage_count[op]}')

# 注释掉未使用的代码块，用于避免其执行
# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(220, 92)
# for op in top_ops_not_covered_by_opinfo:
#    print(f'{op}, {top_ops.usage_count[op]}')

# 注释掉未使用的代码块，用于避免其执行
# print("top ops not covered by opinfo: ")
# top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(999, 999)
# 打印未被 OpInfo 覆盖的顶级操作及其使用次数
for op in top_ops_not_covered_by_opinfo:
    print(f'{op}, {top_ops.usage_count[op]}')

# 从集合 parent 中移除 to_remove 中的元素
def remove_from_set(parent, to_remove):
    for to_remove_elt in to_remove:
        if to_remove_elt in parent:
            parent.remove(to_remove_elt)

# 打印覆盖信息，对于前 th 个操作和后 nn 个操作
def print_coverage_info(th=100, nn=25):
    print("=" * 80)
    print(f"top {th}, {nn} coverage")
    
    # 获取反转后的顶级操作状态
    statuses = transpose_statuses(get_top_ops(th, nn), invert=True)
    
    # 获取未被 OpInfo 覆盖的顶级操作
    top_ops_not_covered_by_opinfo = get_top_ops_not_covered_by_opinfo(th, nn)

    # testing problems
    exemptions = {
        "torch.nn.functional.dropout",  # 随机性问题
    }

    # 允许的豁免
    vmap_exemptions = {
        "torch.randn_like",  # 随机性
        "torch.rand_like",  # 随机性
        "torch.allclose",  # 数值输出
        "torch.unique",  # 动态性
        "torch.nonzero",  # 动态性
        "torch.masked_select",  # 动态性
        "torch.prod",  # 动态性（反向传播）
        "torch.norm",  # 核范数不常用；我们支持其他情况。
        "torch.svd",  # 没有 bug，只是不确定性太高，无法测试。
        "torch.nn.functional.embedding",  # 我们支持除了稀疏选项外的所有情况。
    }

    # 从对应的 statuses 中移除 vmap 豁免列表中的项
    remove_from_set(statuses["test_vmap_exhaustive"], vmap_exemptions)
    remove_from_set(statuses["test_vmapvjp"], vmap_exemptions)
    remove_from_set(statuses["test_vmapvjp_has_batch_rule"], vmap_exemptions)
    remove_from_set(statuses["test_op_has_batch_rule"], vmap_exemptions)
    remove_from_set(statuses["test_vmapjvp"], vmap_exemptions)
    
    # 对于每个测试，从 exemptions 列表中移除对应的状态
    for test in tests:
        remove_from_set(statuses[test], exemptions)

    # 打印总共的操作数
    print(f"total ops in set: {th + nn}")
    # 打印被 OpInfo 测试过的操作数
    print(f"tested by OpInfo: {th + nn - len(top_ops_not_covered_by_opinfo)}")
    
    # 对于每个测试，打印测试失败的覆盖率信息
    for test in tests:
        if test in {"test_jvp", "test_vmapjvp"}:
            continue
        print(f"{test} failing coverage {len(statuses[test])}")

    # 我们暂时不关心以下这些测试
    del statuses["test_jvp"]
    del statuses["test_vmapjvp"]

    # 使用 pprint 打印最终的状态信息
    pprint.pprint(statuses)


# 获取操作名称到 OpInfo 对象的映射
def get_name_to_opinfo_map():
    dct = {}
    for op in op_db + additional_op_db:

        def add(name, op):
            if name not in dct:
                dct[name] = []
            dct[name].append(op)

        add(op.name, op)
        for alias in op.aliases:
            add(alias.name, op)
    return dct


# 使用 get_name_to_opinfo_map() 函数获取名称到 OpInfo 对象的映射，并赋值给 NAME_TO_OPINFO
NAME_TO_OPINFO = get_name_to_opinfo_map()


# 定义 Support 枚举类型，包含 NO，YES，UNKNOWN 三种选项
class Support(enum.Enum):
    NO = 0
    YES = 1
    UNKNOWN = 2


# 定义 FACTORY_FNS 集合，包含多种张量工厂函数名称
FACTORY_FNS = {
    "tensor",
    "zeros",
    "ones",
    "randn",
    "arange",
    "rand",
    "empty",
    "range",
    "full",
    "randperm",
    "eye",
    "randint",
    "linspace",
    "logspace",
}

# 定义 VJP_EXEMPTIONS 集合，包含对梯度的豁免项
VJP_EXEMPTIONS = {
    "nn.functional.dropout",  # 实际上不是问题，与随机性测试相关
    "nn.functional.dropout2d",  # 实际上不是问题，与随机性测试相关
    "nn.functional.rrelu",  # 实际上不是问题，与随机性测试相关
}
    "bernoulli",  # 伯努利分布，实际上不是问题，是随机性测试的产物
    "normal",  # 正态分布，实际上不是问题，是随机性测试的产物
}

# 定义一个名为 VMAP_EXEMPTIONS 的集合，其中包含了不需要进行 VMap 检测的函数名称集合
VMAP_EXEMPTIONS = {
    "randn_like",  # randomness 随机性
    "rand_like",  # randomness 随机性
    "allclose",  # number output 数字输出
    "unique",  # dynamic 动态性
    "nonzero",  # dynamic 动态性
    "masked_select",  # dynamic 动态性
    "prod",  # dynamic (backward) 动态性（反向）
    "norm",  # norm with nuc is not commonly used; we support the other cases. 通常不使用带核范数的规范化；我们支持其他情况。
    "svd",  # There isn't a bug, it is just nondeterministic so we can't test it. 没有 bug，只是不确定性的，因此我们无法测试它。
    "nn.functional.embedding",  # We support everything except the sparse option. 我们支持除了稀疏选项以外的所有内容。
    "nn.functional.dropout",  # randomness 随机性
    "nn.functional.dropout2d",  # randomness 随机性
    "bernoulli",  # randomness 随机性
    "multinomial",  # randomness 随机性
    "normal",  # randomness 随机性
}

# 定义一个名为 JVP_EXEMPTIONS 的集合，其中包含了不需要进行 JVP 检测的函数名称集合
JVP_EXEMPTIONS = {
    "nn.functional.dropout",  # not actually problem, randomness testing artifact 实际上不是问题，随机性测试的产物
    "nn.functional.dropout2d",  # not actually problem, randomness testing artifact 实际上不是问题，随机性测试的产物
    "nn.functional.rrelu",  # not actually problem, randomness testing artifact 实际上不是问题，随机性测试的产物
    "normal",  # not actually problem, randomness testing artifact 实际上不是问题，随机性测试的产物
    "bernoulli",  # not actually problem, randomness testing artifact 实际上不是问题，随机性测试的产物
}

# 定义一个名为 Operator 的类
class Operator:
    def __init__(self, name):
        self.name = name
        # 初始化操作信息为给定名称的操作信息，如果没有则为 None
        self.opinfos = NAME_TO_OPINFO.get(name, None)
        # 确保操作信息是空或者长度大于0
        assert self.opinfos is None or len(self.opinfos) > 0

    # 检查是否存在操作信息
    def has_opinfo(self):
        return self.opinfos is not None

    # 返回操作的字符串表示形式
    def __repr__(self):
        return f'Operator("{self.name}")'

    # 计算操作名称的哈希值
    def __hash__(self):
        return hash(self.name)

    # 如果任何操作信息有测试名称的 skip 或 xfail，则返回 NO；否则返回 YES 或 UNKNOWN
    def no_opinfos_skip_test(self, test_name):
        if not self.has_opinfo():
            return Support.UNKNOWN
        for opinfo in self.opinfos:
            for decorator in opinfo.decorators:
                if not hasattr(decorator, "test_name"):
                    continue
                if decorator.test_name != test_name:
                    continue
                if is_decorateinfo_skip_or_xfail(decorator):
                    return Support.NO
        return Support.YES

    # 检查是否有任何操作信息具有指定属性
    def any_opinfo_attr(self, attr):
        if not self.has_opinfo():
            raise RuntimeError
        return any(getattr(opinfo, attr) for opinfo in self.opinfos)

    # 检查是否所有操作信息具有指定属性
    def all_opinfo_attr(self, attr):
        if not self.has_opinfo():
            raise RuntimeError
        return all(getattr(opinfo, attr) for opinfo in self.opinfos)

    # 检查操作是否支持 VJP（反向传播的一种形式）
    def supports_vjp(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VJP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_vjp")

    # 检查操作是否支持 VMap（向量化映射的一种形式）
    def supports_vmap(self):
        if self.name in FACTORY_FNS:
            return Support.YES
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        return self.no_opinfos_skip_test("test_vmap_exhaustive")
    # 检查是否支持快速的 vmap 操作
    def supports_fast_vmap(self):
        # 如果操作名称在工厂函数列表中，则返回支持
        if self.name in FACTORY_FNS:
            return Support.YES
        # 如果操作名称在 vmap 豁免列表中，则返回支持
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        # 否则，执行指定的测试来决定是否支持 vmap 操作
        return self.no_opinfos_skip_test("test_op_has_batch_rule")

    # 检查是否支持 vmapvjp 操作
    def supports_vmapvjp(self):
        # 如果操作名称在工厂函数列表中，则返回支持
        if self.name in FACTORY_FNS:
            return Support.YES
        # 如果操作名称在 vmap 豁免列表中，则返回支持
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        # 否则，执行指定的测试来决定是否支持 vmapvjp 操作
        return self.no_opinfos_skip_test("test_vmapvjp")

    # 检查是否支持快速的 vmapvjp 操作
    def supports_fast_vmapvjp(self):
        # 如果操作名称在工厂函数列表中，则返回支持
        if self.name in FACTORY_FNS:
            return Support.YES
        # 如果操作名称在 vmap 豁免列表中，则返回支持
        if self.name in VMAP_EXEMPTIONS:
            return Support.YES
        # 否则，执行指定的测试来决定是否支持快速的 vmapvjp 操作
        return self.no_opinfos_skip_test("test_vmapvjp_has_batch_rule")

    # 检查是否支持 jvp 操作
    def supports_jvp(self):
        # 如果操作名称在工厂函数列表中，则返回支持
        if self.name in FACTORY_FNS:
            return Support.YES
        # 如果操作名称在 jvp 豁免列表中，则返回支持
        if self.name in JVP_EXEMPTIONS:
            return Support.YES
        # 如果没有 OpInfo，则返回未知的支持情况
        if not self.has_opinfo():
            return Support.UNKNOWN
        # 如果支持自动微分但不支持前向自动微分，则返回不支持
        if self.any_opinfo_attr("supports_autograd") and not self.all_opinfo_attr(
            "supports_forward_ad"
        ):
            return Support.NO
        # 否则，执行指定的测试来决定是否支持 jvp 操作
        return self.no_opinfos_skip_test("test_jvp")

    # 检查是否支持 jvpvjp 操作
    def supports_jvpvjp(self):
        # 如果操作名称在工厂函数列表中，则返回支持
        if self.name in FACTORY_FNS:
            return Support.YES
        # 如果操作名称在 jvpvjp 豁免列表中，则返回支持
        exemptions = {
            # 以下操作在 OpInfo 中有支持，这里仅用于测试
            "nn.functional.dropout2d",
            "nn.functional.dropout",
            # 例外：甚至不支持双向反向传播
            "nn.functional.hardswish",
            "bernoulli",  # 不可微分
            "normal",  # 不可微分
        }
        if self.name in exemptions:
            return Support.YES
        # 否则，执行指定的测试来决定是否支持 jvpvjp 操作
        return self.no_opinfos_skip_test("test_jvpvjp")

    # 内部方法：检查是否支持 vmapjvp 基础操作
    def _supports_vmapjvp_base(self, test):
        # 如果操作名称在工厂函数列表中，则返回支持
        if self.name in FACTORY_FNS:
            return Support.YES
        # 如果操作名称在 vmapjvp 豁免列表中，则返回支持
        VMAPJVP_EXEMPTIONS = {
            "prod",  # 动态（反向传播）
            "nn.functional.batch_norm",  # 测试问题
            "normal",  # 实际上不是问题，随机性测试
            "bernoulli",  # 实际上不是问题，随机性测试
            "nn.functional.dropout2d",  # 实际上不是问题，随机性测试
            "nn.functional.dropout",  # 实际上不是问题，随机性测试
            # 不是问题。
            # 只是 max_norm 测试会改变输入...
            # （我们有自己的 functorch OpInfo 变种，没有 max_norm）
            "nn.functional.embedding",
        }
        if self.name in VMAPJVP_EXEMPTIONS:
            return Support.YES
        # 如果没有 OpInfo，则返回未知的支持情况
        if not self.has_opinfo():
            return Support.UNKNOWN
        # 如果支持自动微分但不支持前向自动微分，则返回不支持
        if self.any_opinfo_attr("supports_autograd") and not self.all_opinfo_attr(
            "supports_forward_ad"
        ):
            return Support.NO
        # 否则，执行指定的测试来决定是否支持 vmapjvp 基础操作
        return self.no_opinfos_skip_test(test)
    # 检查当前对象是否支持 vmapjvp，通过调用 _supports_vmapjvp_base 方法并传入特定测试名称来确定
    def supports_vmapjvp(self):
        return self._supports_vmapjvp_base("test_vmapjvpall")
    
    # 检查当前对象是否支持快速的 vmapjvp，通过调用 _supports_vmapjvp_base 方法并传入特定测试名称来确定
    def supports_fast_vmapjvp(self):
        return self._supports_vmapjvp_base("test_vmapjvpall_has_batch_rule")
class OperatorSet:
    # 定义操作符集合类

    def __init__(self, operators):
        # 初始化方法，接收操作符集合并存储在实例变量中
        self.data = set(operators)

    @classmethod
    def from_names(cls, names):
        # 类方法：根据操作符名称列表创建操作符集合对象
        return OperatorSet([Operator(name) for name in names])

    @classmethod
    def from_top_ops_threshold(cls, torch_threshold, nn_fn_threshold):
        # 类方法：根据两个阈值获取前置操作符名称列表，并创建操作符集合对象
        names = get_top_ops(torch_threshold, nn_fn_threshold)
        return cls.from_names(names)

    @classmethod
    def from_top125(cls):
        # 类方法：获取前125个操作符名称列表，并创建操作符集合对象
        return cls.from_top_ops_threshold(100, 25)

    @classmethod
    def from_top160(cls):
        # 类方法：获取前160个操作符名称列表，并创建操作符集合对象
        return cls.from_top_ops_threshold(107, 53)

    @classmethod
    def all(cls):
        # 类方法：获取所有公共可重写的且重要的操作符名称，并创建操作符集合对象
        dct = get_public_overridable_outplace_we_care_about()
        names = dct.keys()
        names_sanitized = []
        for n in names:
            torch_tensor = "torch.Tensor."
            torch_dot = "torch."
            if n.startswith(torch_tensor):
                names_sanitized.append(n[len(torch_tensor) :])
            elif n.startswith(torch_dot):
                names_sanitized.append(n[len(torch_dot) :])
            else:
                raise AssertionError
        return cls.from_names(names_sanitized)

    def query(self, operator_method, filter=(Support.NO, Support.YES, Support.UNKNOWN)):
        # 查询方法：根据给定的操作符方法和过滤条件查询结果
        result = {}
        for key in filter:
            result[key] = set()
        for op in self.data:
            support_status = operator_method(op)
            if support_status in filter:
                result[support_status].add(op)
        return result

    def summary(self):
        # 汇总方法：生成操作符支持情况的摘要信息
        checks = [
            "supports_vjp",
            "supports_vmap",
            "supports_fast_vmap",
            "supports_vmapvjp",
            "supports_fast_vmapvjp",
            "supports_jvp",
            "supports_vmapjvp",
            "supports_fast_vmapjvp",
            "supports_jvpvjp",
        ]
        result = ["test, yes, no, unknown"]
        for check in checks:
            accessor = getattr(Operator, check)
            all_results = self.query(accessor)
            yes_amt = len(all_results[Support.YES])
            no_amt = len(all_results[Support.NO])
            unknown_amt = len(all_results[Support.UNKNOWN])
            result.append(f"{check}, {yes_amt}, {no_amt}, {unknown_amt}")
        return "\n".join(result)


opset = OperatorSet.all()
# 获取所有操作符集合对象

has_no_opinfo = opset.query(Operator.has_opinfo, (False,))
# 查询没有操作符信息的操作符集合

print("=" * 30 + " Summary " + "=" * 30)
# 打印摘要信息的分隔符
print(f"% of usages on github: {get_ops_percentage(99999, 99999)}")
# 打印操作符在 GitHub 上的使用百分比估计
print(opset.summary())
# 打印操作符集合的摘要信息

# sanity checks
# 健全性检查
result = opset.query(Operator.supports_vjp, (Support.NO, Support.UNKNOWN))
# 查询支持 vjp 的操作符集合，并过滤出不支持和未知的结果
# pprint.pprint(result)

print("=" * 30 + " Top 60 Summary " + "=" * 30)
# 打印前60个操作符的摘要信息的分隔符
print(f"% of usages on github: {get_ops_percentage(35, 25)}")
# 打印前60个操作符在 GitHub 上的使用百分比估计
opset = OperatorSet.from_top_ops_threshold(35, 25)
# 根据给定的阈值重新创建操作符集合对象
# result = opset.query(Operator.supports_vmapjvp, (Support.NO, Support.UNKNOWN))
# 查询支持 vmapjvp 的操作符集合，并过滤出不支持和未知的结果
# pprint.pprint(result)
# result = opset.query(Operator.supports_jvp, (Support.NO, Support.UNKNOWN))
# 查询支持 jvp 的操作符集合，并过滤出不支持和未知的结果
# pprint.pprint(result)
# 打印当前操作符集对象的摘要信息
print(opset.summary())

# 打印分隔线和标题，用于标记顶部125的摘要信息
print("=" * 30 + " Top 125 Summary " + "=" * 30)

# 打印在 GitHub 上使用的操作符的百分比
print(f"% of usages on github: {get_ops_percentage(100, 25)}")

# 从顶部125操作符集中创建操作符集对象
opset = OperatorSet.from_top125()

# 查询是否支持 vjp 操作，传入 (Support.NO, Support.UNKNOWN) 作为参数
print("supports_vjp")
result = opset.query(Operator.supports_vjp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)

# 查询是否支持 jvp 操作，传入 (Support.NO, Support.UNKNOWN) 作为参数
print("supports_jvp")
result = opset.query(Operator.supports_jvp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)

# 查询是否支持 vmapjvp 操作，传入 (Support.NO, Support.UNKNOWN) 作为参数
print("supports_vmapjvp")
result = opset.query(Operator.supports_vmapjvp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)

# 查询是否支持 jvpvjp 操作，传入 (Support.NO, Support.UNKNOWN) 作为参数
print("supports_jvpvjp")
result = opset.query(Operator.supports_jvpvjp, (Support.NO, Support.UNKNOWN))
pprint.pprint(result)

# 打印操作符集对象的摘要信息
print(opset.summary())

# 打印分隔线和标题，用于标记顶部160的摘要信息，但被注释掉了
# print("=" * 30 + " Top 160 Summary " + "=" * 30)
# 从顶部160操作符集中创建操作符集对象，但被注释掉了
# opset = OperatorSet.from_top160()
# 查询是否支持 jvpvjp 操作，传入 (Support.NO, Support.UNKNOWN) 作为参数，但被注释掉了
# result = opset.query(Operator.supports_jvpvjp, (Support.NO, Support.UNKNOWN))
# pprint.pprint(result)
# 打印操作符集对象的摘要信息，但被注释掉了
# print(opset.summary())

# 打印列表中的所有操作符及其出现次数，但被注释掉了
# all_ops = get_top_ops(999999, 999999, with_counts=True)
# for op, count in all_ops:
#     print(f'{op}, {count}')
```