# `.\pytorch\test\functorch\xfail_suggester.py`

```py
# 导入正则表达式模块
import re

# 导入 PyTorch 模块
import torch

# 包含测试执行指南的多行注释

# 打开并读取结果文件中的所有行
with open("result.txt") as f:
    lines = f.readlines()

# 筛选出所有以 "FAILED" 开头的行，表示测试失败的信息
failed = [line for line in lines if line.startswith("FAILED")]

# 使用正则表达式定义匹配失败测试信息的模式
p = re.compile("FAILED test/test_\w+.py::\w+::(\S+)")  # noqa: W605

# 从匹配的行中提取出失败的测试名称
def get_failed_test(line):
    m = p.match(line)
    if m is None:
        return None
    return m.group(1)

# 预定义的测试基本名称集合
base_names = {
    "test_grad_",
    "test_vjp_",
    "test_vmapvjp_",
    "test_vmapvjp_has_batch_rule_",
    "test_vjpvmap_",
    "test_jvp_",
    "test_vmapjvp_",
    "test_vmapjvpall_has_batch_rule_",
    "test_vmapjvpall_",
    "test_jvpvjp_",
    "test_vjpvjp_",
    "test_decomposition_",
    "test_make_fx_exhaustive_",
    "test_vmap_exhaustive_",
    "test_op_has_batch_rule_",
    "test_vmap_autograd_grad_",
}

# 提取所有失败测试的名称
failed_tests = [get_failed_test(line) for line in lines]
failed_tests = [match for match in failed_tests if match is not None]
failed_tests = sorted(failed_tests)

# 建立建议的 XFAIL（expected failure）字典
suggested_xfails = {}

# 移除测试名称中的设备和数据类型信息
def remove_device_dtype(test):
    return "_".join(test.split("_")[:-2])

# 检查测试名称是否属于特定基本名称
def belongs_to_base(test, base):
    if not test.startswith(base):
        return False
    candidates = [try_base for try_base in base_names if len(try_base) > len(base)]
    for candidate in candidates:
        if test.startswith(candidate):
            return False
    return True

# 解析测试名称的命名空间
def parse_namespace(base):
    mappings = {
        "nn_functional_": "nn.functional",
        "fft_": "fft",
        "linalg_": "linalg",
        "_masked_": "_masked",
        "sparse_": "sparse",
        "special_": "special",
    }
    for heading in mappings.keys():
        if base.startswith(heading):
            return mappings[heading], base[len(heading):]
    return None, base

# 根据命名空间获取相应的 Torch 模块
def get_torch_module(namespace):
    if namespace is None:
        return torch
    if namespace == "nn.functional":
        return torch.nn.functional
    return getattr(torch, namespace)

# 解析基本测试名称，获取命名空间、API 和变体
def parse_base(base):
    namespace, rest = parse_namespace(base)

    apis = dir(get_torch_module(namespace))
    apis = sorted(apis, key=lambda x: -len(x))

    api = rest
    variant = ""
    for candidate in apis:
        if rest.startswith(candidate):
            api = candidate
            variant = rest[len(candidate) + 1 :]
            break
    print(base, namespace, api, variant)
    return namespace, api, variant

# 检查给定列表中是否有字符串以特定字符串开头
def any_starts_with(strs, thing):
    for s in strs:
        if s.startswith(thing):
            return True
    return False

# 获取建议的 XFAIL 列表，包括基本名称和对应的测试列表
def get_suggested_xfails(base, tests):
    result = []
    tests = [test[len(base):] for test in tests if belongs_to_base(test, base)]

    base_tests = {remove_device_dtype(test) for test in tests}
    tests = set(tests)
    # 遍历给定的 base_tests 列表中的每个基础测试用例
    for base in base_tests:
        # 构建 CPU 变体名称
        cpu_variant = base + "_cpu_float32"
        # 构建 CUDA 变体名称
        cuda_variant = base + "_cuda_float32"
        # 解析基础测试用例，获取命名空间、API 和变体信息
        namespace, api, variant = parse_base(base)
        
        # 如果未解析到命名空间，则保持 API 不变，否则将 API 格式化为带命名空间的形式
        if namespace is None:
            api = api
        else:
            api = f"{namespace}.{api}"
        
        # 根据 CPU 变体和 CUDA 变体是否都在 tests 中，决定是否标记为 xfail
        if cpu_variant in tests and cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}'),")
            continue
        
        # 如果只有 CPU 变体在 tests 中，则标记为在 CPU 上 xfail
        if cpu_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cpu'),")
            continue
        
        # 如果只有 CUDA 变体在 tests 中，则标记为在 CUDA 上 xfail
        if cuda_variant in tests:
            result.append(f"xfail('{api}', '{variant}', device_type='cuda'),")
            continue
        
        # 如果 CPU 变体和 CUDA 变体都不在 tests 中，则标记为跳过
        result.append(f"skip('{api}', '{variant}',")
    
    # 返回处理后的结果列表
    return result
# 使用推荐的 xfails 来构建结果字典，遍历 base_names 列表中的每个基础名称
result = {base: get_suggested_xfails(base, failed_tests) for base in base_names}

# 遍历结果字典中的每个键值对，输出分隔线并打印键名
for k, v in result.items():
    print("=" * 50)
    # 打印当前处理的基础名称
    print(k)
    print("=" * 50)
    # 将推荐的 xfails 列表转换为字符串并逐行打印
    print("\n".join(v))
```