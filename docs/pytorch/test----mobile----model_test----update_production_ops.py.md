# `.\pytorch\test\mobile\model_test\update_production_ops.py`

```py
"""
This is a script to aggregate production ops from xplat/pytorch_models/build/all_mobile_model_configs.yaml.
Specify the file path in the first argument. The results will be dumped to model_ops.yaml
"""

import sys  # 导入sys模块，用于处理命令行参数
import yaml  # 导入yaml模块，用于读写YAML格式的文件

root_operators = {}  # 初始化空字典，用于存储根操作符及其出现次数
traced_operators = {}  # 初始化空字典，用于存储跟踪操作符及其出现次数
kernel_metadata = {}  # 初始化空字典，用于存储内核元数据中的数据类型列表

with open(sys.argv[1]) as input_yaml_file:  # 打开命令行参数指定的YAML文件
    model_infos = yaml.safe_load(input_yaml_file)  # 使用yaml模块加载YAML文件内容为Python对象
    for info in model_infos:  # 遍历YAML文件中的每个模型信息
        for op in info["root_operators"]:  # 遍历每个模型信息中的根操作符列表
            # 统计每个根操作符出现的次数
            root_operators[op] = 1 + (root_operators[op] if op in root_operators else 0)
        for op in info["traced_operators"]:  # 遍历每个模型信息中的跟踪操作符列表
            # 统计每个跟踪操作符出现的次数
            traced_operators[op] = 1 + (traced_operators[op] if op in traced_operators else 0)
        for kernal, dtypes in info["kernel_metadata"].items():  # 遍历每个模型信息中的内核元数据
            # 合并每个内核的数据类型列表，确保每种数据类型只出现一次
            new_dtypes = dtypes + (kernel_metadata[kernal] if kernal in kernel_metadata else [])
            kernel_metadata[kernal] = list(set(new_dtypes))  # 更新内核元数据的数据类型列表

# 仅保留这些内置操作符，不包括自定义操作符或非CPU操作符
namespaces = ["aten", "prepacked", "prim", "quantized"]
root_operators = {
    x: root_operators[x] for x in root_operators if x.split("::")[0] in namespaces
}
traced_operators = {
    x: traced_operators[x] for x in traced_operators if x.split("::")[0] in namespaces
}

out_path = "test/mobile/model_test/model_ops.yaml"  # 输出文件路径
with open(out_path, "w") as f:  # 打开输出文件以便写入
    yaml.safe_dump({"root_operators": root_operators}, f)  # 将根操作符及其出现次数写入YAML文件
```