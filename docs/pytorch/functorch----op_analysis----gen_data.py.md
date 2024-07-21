# `.\pytorch\functorch\op_analysis\gen_data.py`

```py
import csv  # 导入处理 CSV 文件的模块
from collections import defaultdict  # 导入 defaultdict 数据结构

import yaml  # 导入处理 YAML 文件的模块

import torch  # 导入 PyTorch 深度学习框架


def get_ops_for_key(key):
    # 获取指定调度键（dispatch key）对应的操作集合
    # 需要修改的 PyTorch C++ 代码来工作
    if key is None:
        ops = torch._C._dispatch_get_registrations_for_dispatch_key()
    else:
        ops = torch._C._dispatch_get_registrations_for_dispatch_key(key)
    cleaned_ops = []
    # 过滤掉不是 "aten::" 开头的操作名称
    for i in ops:
        if "aten::" not in i:
            continue
        cleaned_ops.append(i[6:].strip())
    return set(cleaned_ops)


def gen_data(special_op_lists, analysis_name):
    # 获取所有操作的注册信息，不指定特定的调度键
    all_ops = get_ops_for_key(None)
    # 获取组合操作的注册信息，特定的调度键为 "CompositeImplicitAutograd"
    composite_ops = get_ops_for_key("CompositeImplicitAutograd")
    # 非组合操作为所有操作减去组合操作的集合
    noncomposite_ops = all_ops - composite_ops

    # 从本地 YAML 文件中加载操作的定义
    ops = yaml.load(
        open("../../aten/src/ATen/native/native_functions.yaml").read(),
        Loader=yaml.CLoader,
    )

    # 从 CSV 文件中读取已注释操作的列表，存储为字典
    annotated_ops = {
        a.strip(): b.strip() for a, b in list(csv.reader(open("annotated_ops")))
    }

    from collections import defaultdict  # 再次导入 defaultdict 数据结构（此处应删除）

    # 存储唯一操作的列表和操作名称的集合
    uniq_ops = []
    uniq_names = set()
    # 存储函数重载类型的 defaultdict 列表
    overload_types = defaultdict(list)
    cnt = 0
    # 遍历操作列表
    for op in ops:
        func_str = op["func"]
        # 提取操作函数的名称
        name = func_str[: func_str.index("(")]
        # 根据名称是否含有 "." 分隔符判断唯一名称
        if "." in name:
            uniq_name = name[: name.index(".")]
            overload_types[name[name.index(".") + 1 :]].append(name)
        else:
            uniq_name = name
        # 添加额外的操作属性
        op["name"] = uniq_name
        full_name = func_str[: func_str.index("(")]
        op["full_name"] = full_name
        ret_type = func_str[func_str.index("->") + 3 :]
        op["ret_type"] = ret_type
        cnt += 1
        # 如果唯一名称已存在，则跳过当前操作
        if uniq_name in uniq_names:
            continue
        uniq_names.add(uniq_name)
        uniq_ops.append(op)
    # 定义函数 annotate_ops，用于对操作列表 ops 进行分类注释
    def annotate_ops(ops, is_unique):
        # 创建一个默认字典用于存储不同类别操作的计数
        categorization = defaultdict(int)
        # 遍历操作列表 ops 中的每一个操作 op
        for op in ops:
            # 如果操作名称以下划线结尾，则认为是原地操作
            if op["name"][-1] == "_":
                categorization["inplace"] += 1
                # 将操作的 meta 属性标记为 "inplace"
                op["meta"] = "inplace"
                continue
            # 如果不要求唯一且操作函数名包含 "a!"，则标记为输出操作
            if not is_unique and "a!" in op["func"].lower():
                categorization["out"] += 1
                op["meta"] = "out"
                continue
            # 如果操作名称中包含 "conv"，则认为是卷积操作
            if "conv" in op["name"]:
                categorization["conv"] += 1
                op["meta"] = "conv"
                continue
            # 如果操作名称中包含 "pool"，则认为是池化操作
            if "pool" in op["name"]:
                categorization["pool"] += 1
                op["meta"] = "pool"
                continue
            # 如果操作名称中包含 "backward"，则认为是反向操作
            if "backward" in op["name"]:
                categorization["backward"] += 1
                op["meta"] = "backward"
                continue
            # 如果操作名称以单下划线开头且第二个字符不是下划线，则标记为私有操作
            if op["name"][0] == "_" and op["name"][1] != "_":
                categorization["private"] += 1
                op["meta"] = "private"
                continue
            # 如果操作名称中包含 "batch_norm"，则认为是批量归一化操作
            if "batch_norm" in op["name"]:
                categorization["batch_norm"] += 1
                op["meta"] = "batch_norm"
                continue
            # 如果操作不涉及 Tensor 或其返回类型，标记为非张量操作
            if "Tensor" not in op["func"] or "Tensor" not in op["ret_type"]:
                categorization["non_tensor"] += 1
                op["meta"] = "non_tensor"
                continue
            # 如果操作名称中包含特定的后端引擎关键词，标记为后端操作
            if (
                "cudnn" in op["name"]
                or "mkldnn" in op["name"]
                or "miopen" in op["name"]
                or "native" in op["name"]
                or "thnn" in op["name"]
                or "slow" in op["name"]
            ):
                categorization["backend"] += 1
                op["meta"] = "backend"
                continue
            # 如果操作名称在 annotated_ops 中，标记为核心操作，并根据 annotated_ops 给出具体类型
            if op["name"] in annotated_ops:
                categorization["core"] += 1
                op["meta"] = "core " + annotated_ops[op["name"]]
                continue
            # 否则，标记为未知核心操作
            categorization["core"] += 1
            op["meta"] = "core unknown"
        # 返回各类操作的计数结果字典 categorization
        return categorization

    # 调用 annotate_ops 函数，对操作列表 ops 进行分类注释，不要求唯一性
    annotate_ops(ops, is_unique=False)

    # 打开文件 analysis_name 以写入模式
    with open(f"{analysis_name}", "w") as f:
        # 遍历操作列表 ops 中的每一个操作 op
        for op in ops:
            # 组装输出的信息列表 info，包括操作的完整名称、meta 属性、是否为非复合操作
            info = [
                op["full_name"],
                op["meta"],
                op["full_name"] not in noncomposite_ops,
            ] + [check(op) for check in special_op_lists]
            # 将 info 中的各项转换为字符串，并用逗号连接后写入文件 f，同时换行
            f.write(",".join([str(i) for i in info]) + "\n")
# 检查给定列表中是否存在与输入字典 x 的 "name" 键对应的值
def name_check(lst):
    return lambda x: x["name"] in lst

# 检查给定列表中是否存在与输入字典 x 的 "full_name" 键对应的值
def full_name_check(lst):
    return lambda x: x["full_name"] in lst

# 生成批处理规则数据，使用 get_ops_for_key 函数获取键为 "FuncTorchBatched" 的操作列表
gen_data([full_name_check(get_ops_for_key("FuncTorchBatched"))], "vmap.txt")

# 去除输入字符串的结尾处指定的后缀
def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string

# 去除输入字符串的开头处指定的前缀
def remove_prefix(input_string, prefix):
    if prefix and input_string.startswith(prefix):
        return input_string[len(prefix) :]
    return input_string

# 如果条件为真，执行以下代码块
if True:
    # 从 "run_ops.txt" 文件中读取每一行并去除结尾的 ".default" 后缀
    with open("run_ops.txt") as f:
        opinfo_ops = [remove_suffix(i.strip(), ".default") for i in f]
    # 从 "count_ops.txt" 文件中读取每一行，去除首尾空白字符，并创建 opinfo_counts 字典，使用 opinfo_ops 作为键
    with open("count_ops.txt") as f:
        opinfo_counts = [i.strip() for i in f]
        opinfo_counts = defaultdict(int, dict(zip(opinfo_ops, opinfo_counts)))

    # 定义一个函数 count_fn，用于返回给定操作的计数，根据操作的 "full_name" 查找 opinfo_counts 中的计数值
    def count_fn(x):
        return opinfo_counts[x["full_name"]]

    # 从 "run_decompositions.txt" 文件中读取每一行并去除结尾的 ".default" 后缀
    with open("run_decompositions.txt") as f:
        decomposed_ops = [remove_suffix(i.strip(), ".default") for i in f]

    # 从 "public_api" 文件中读取每一行并去除首尾空白字符，形成 ref_api 列表
    with open("public_api") as f:
        ref_api = [i.strip() for i in f]

    # 定义一个函数 has_ref_impl，用于检查输入字典 x 的 "name" 是否在 ref_api 中的任意前缀或直接存在于 ref_api 中
    def has_ref_impl(x):
        name = x["name"]
        # 去除 name 的前缀 "linalg_" 和 "special_"
        for prefix in ["linalg_", "special_"]:
            name = remove_prefix(name, prefix)
        # 定义可能的前缀列表
        prefixes = ["nn.functional", "fft", "special", "linalg"]
        # 检查是否存在任意前缀加上 name 是否在 ref_api 中，或者 name 是否直接存在于 ref_api 中
        return (
            any(f"{prefix}.{name}" in ref_api for prefix in prefixes) or name in ref_api
        )

    # 生成批处理数据，传入 full_name_check(opinfo_ops), full_name_check(decomposed_ops), count_fn, has_ref_impl 四个函数作为规则
    gen_data(
        [
            full_name_check(opinfo_ops),
            full_name_check(decomposed_ops),
            count_fn,
            has_ref_impl,
        ],
        "decompositions.txt",
    )
```