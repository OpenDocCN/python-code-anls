# `.\pytorch\benchmarks\dynamo\microbenchmarks\operator_inp_utils.py`

```py
import functools
import logging
import math
import os
from collections import Counter, defaultdict
from functools import partial
from typing import Any, Dict, Generator, Iterable, Tuple

import torch
from torch.testing import make_tensor
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map

log = logging.getLogger(__name__)

# 设置操作输入日志文件夹路径
OP_INP_DIRECTORY = os.path.join(os.path.dirname(__file__), "operator_inp_logs")

# 设置不同任务模块的日志文件夹路径
TIMM_DIR = os.path.join(OP_INP_DIRECTORY, "timm_train")
HF_DIR = os.path.join(OP_INP_DIRECTORY, "hf_train")
TORCHBENCH_DIR = os.path.join(OP_INP_DIRECTORY, "torchbench_train")

# 引入 torch 中的 aten 操作
aten = torch.ops.aten
# 获取 torch 中的 TensorType 类型
tensor_type = torch._C.TensorType.get()

# 定义不同数据类型的缩写字典
dtype_abbrs = {
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.float32: "f32",
    torch.float16: "f16",
    torch.complex32: "c32",
    torch.complex64: "c64",
    torch.complex128: "c128",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
    torch.bool: "b8",
    torch.uint8: "u8",
}

# 创建从缩写到数据类型的反向字典
dtype_abbrs_parsing = {value: key for key, value in dtype_abbrs.items()}


# 函数：将输入参数截断为对应的缩写
def truncate_inp(arg):
    if arg in dtype_abbrs:
        return dtype_abbrs[arg]
    elif isinstance(arg, torch.device):
        return arg.type
    else:
        return arg


# 类：序列化函数调用
class FuncCallWrapper:
    def __init__(self, call, *args, **kwargs):
        self.call = call
        # 使用 tree_map 将 args 和 kwargs 中的每个参数截断为对应的缩写
        self.args = tree_map(truncate_inp, args)
        self.kwargs = tree_map(truncate_inp, kwargs) if kwargs is not None else {}

    def __repr__(self):
        # 构建函数调用的字符串表示
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = "".join(
            [f", {str(key)}={value}" for key, value in self.kwargs.items()]
        )
        out = f"{self.call}({args}{kwargs})".strip('"')
        # 替换字符串中的单引号为对应的数据类型
        for key in dtype_abbrs_parsing:
            out = out.replace(f"'{key}'", key)
        return out


# 函数：序列化稀疏张量
def serialize_sparse_tensor(e):
    if isinstance(e, torch._subclasses.FakeTensor):
        return FuncCallWrapper("ST", list(e.shape), e.dtype, e.layout, e.is_coalesced())
    else:
        return FuncCallWrapper(
            "ST", list(e.shape), e.dtype, e.layout, e.is_coalesced(), e._nnz()
        )


# 函数：反序列化稀疏张量（未实现）
def deserialize_sparse_tensor(size, dtype, layout, is_coalesced, nnz=None):
    raise NotImplementedError


# 函数：反序列化张量
def deserialize_tensor(size, dtype, stride=None):
    if stride is not None:
        out = torch.empty_strided(size, stride, dtype=dtype)
    else:
        out = torch.empty(size, dtype=dtype)
    try:
        out.copy_(make_tensor(size, dtype=dtype, device="cpu"))
    except Exception as e:
        print(e)
        return out
    return out


# 函数：序列化张量
def serialize_tensor(e):
    if not e.is_contiguous():
        return FuncCallWrapper("T", list(e.shape), e.dtype, stride=e.stride())
    else:
        return FuncCallWrapper("T", list(e.shape), e.dtype)
# 利用 Torch 序列化参数的函数，根据参数类型执行不同的序列化操作
def serialize_torch_args(e):
    if isinstance(e, torch.Tensor):  # 检查是否为 Torch 张量
        if e.is_sparse:  # 如果是稀疏张量，调用稀疏张量的序列化函数
            return serialize_sparse_tensor(e)
        return serialize_tensor(e)  # 否则调用普通张量的序列化函数
    else:
        return truncate_inp(e)  # 对于非张量参数，截断处理返回

# 检查给定元素中是否包含 Torch 张量
def contains_tensor(elems):
    for elem in pytree.tree_leaves(elems):  # 遍历元素树的所有叶子节点
        if isinstance(elem, torch.Tensor):  # 如果叶子节点是 Torch 张量，返回 True
            return True
    return False  # 如果没有找到 Torch 张量，则返回 False

# 检查给定元素中是否包含特定类型的 Torch 对象
def skip_args(elems):
    for i in pytree.tree_leaves(elems):  # 遍历元素树的所有叶子节点
        # 只在构造函数和类似操作中显示
        if isinstance(i, (torch.memory_format, torch.storage.UntypedStorage)):
            return True  # 如果叶子节点是指定的 Torch 类型，返回 True
    return False  # 如果没有找到指定的 Torch 类型，则返回 False

# 检查类型是否包含 Torch 张量类型
def contains_tensor_types(type):
    return type.isSubtypeOf(tensor_type) or any(
        contains_tensor_types(e) for e in type.containedTypes()
    )  # 检查类型是否是 Tensor 类型的子类型或者包含 Tensor 类型的子类型

# 缓存不进行计算的操作符
@functools.lru_cache(None)
def non_compute_operator(op):
    schema = op._schema  # 获取操作符的模式

    # 跳过构造函数
    if not any(contains_tensor_types(arg.type) for arg in schema.arguments):
        return True  # 如果操作符的参数中没有包含 Tensor 类型，则返回 True
    if "_like" in op.name():  # 如果操作符名包含 "_like" 字符串，则返回 True
        return True

    # 允许原地写操作
    if schema.is_mutable:  # 如果操作符是可变的，则返回 False
        return False

    tensor_inps = [arg for arg in schema.arguments if arg.type is tensor_type]  # 获取所有 Tensor 类型的输入参数
    tensor_outputs = [ret for ret in schema.returns if ret.type is tensor_type]  # 获取所有 Tensor 类型的输出参数

    # 跳过别名操作，除非有多个输出
    if len(tensor_outputs) != 1:  # 如果输出的 Tensor 数量不等于 1，则返回 False
        return False

    for inp in tensor_inps:
        if inp.alias_info and tensor_outputs[0].alias_info:
            if inp.alias_info.before_set.intersection(
                tensor_outputs[0].alias_info.after_set
            ):
                return True  # 如果输入的别名信息与输出的别名信息交叉，则返回 True

    return False  # 否则返回 False

# 定义一个特定的 Torch 调度模式，用于操作符的输入处理
class OperatorInputsMode(TorchDispatchMode):
    def __init__(self, func_db=None):
        self.func_db = defaultdict(Counter) if func_db is None else func_db  # 初始化函数数据库

    def __torch_dispatch__(self, func_overload, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}  # 确保 kwargs 不为 None
        arg_meta, kwarg_meta = tree_map(serialize_torch_args, (args, kwargs))  # 序列化输入参数和关键字参数

        out = func_overload(*args, **kwargs)  # 执行函数重载

        inps = (args, kwargs)
        if contains_tensor(inps) and not skip_args(inps) and contains_tensor(out):
            # 如果输入参数包含 Tensor，不跳过输入参数，且输出包含 Tensor，则进行序列化并更新函数数据库
            serialized_str = repr((arg_meta, kwarg_meta))
            self.func_db[str(func_overload)][serialized_str] += 1

        return out  # 返回函数执行结果
    # 定义一个实例方法用于将日志记录到文件中，参数包括输出文件名和一个命名关键字参数
    def log_to_file(self, output_filename, *, skip_non_compute_operators=True):
        # 按照操作符字典中的键排序操作符
        sorted_operators = sorted(self.func_db.keys())
        # 打开输出文件，模式为写入模式，使用 'w' 参数
        with open(output_filename, "w") as f:
            # 遍历排序后的操作符列表
            for operator in sorted_operators:
                # 如果 skip_non_compute_operators 为 True 并且该操作符是非计算操作符，则跳过
                if skip_non_compute_operators and non_compute_operator(eval(operator)):
                    continue
                # 写入操作符的标识信息到文件
                f.write(f"Operator: {operator}\n")
                # 获取该操作符在函数数据库中的输入信息字典
                operator_inputs = self.func_db[operator]
                # 遍历操作符输入信息字典中的键值对
                for inps, count in operator_inputs.items():
                    # 写入输入数量和输入内容到文件
                    f.write(f"cnt: {count}, ")
                    # 将输入内容中的数据类型缩写替换为对应的全称，使用 dtype_abbrs 字典进行映射
                    for dtype_abbr in dtype_abbrs.values():
                        inps = inps.replace("'" + dtype_abbr + "'", dtype_abbr)
                    f.write(inps)
                    f.write("\n")
# OperatorInputsLoader 类负责从给定的 JSON 文件加载运算符输入的统计信息
class OperatorInputsLoader:
    # 初始化方法，接受一个 JSON 文件路径作为参数
    def __init__(self, json_file_path):
        # 使用 defaultdict 创建一个计数器字典来存储运算符输入的统计信息
        self.operator_db = defaultdict(Counter)

        # 打开并读取 JSON 文件内容
        with open(json_file_path) as f:
            lines = f.readlines()

        i = 0
        # 循环处理文件的每一行
        while i < len(lines):
            # 去除行尾换行符
            op_line = lines[i].strip("\n")
            # 断言检查行中是否包含 "Operator: "，如果不包含则抛出异常
            assert "Operator: " in op_line, op_line
            # 提取运算符名称
            operator = op_line[len("Operator: "):]
            # 如果运算符名称是 "aten.sum.SymInt"，则修改为 "aten.sum.dim_IntList"
            operator = operator if operator != "aten.sum.SymInt" else "aten.sum.dim_IntList"
            # 创建一个计数器来存储该运算符的输入及其计数
            op_inps = Counter()
            i += 1
            # 处理当前运算符下的输入行
            while i < len(lines) and "Operator: " not in lines[i]:
                line = lines[i]
                # 解析输入行中的计数
                cnt = eval(line[len("cnt: "):line.find(",")])
                # 提取输入内容并去除首尾引号
                inps = line[line.find(",") + 2:].strip("'")
                # 将输入内容及其计数添加到计数器中
                op_inps[inps] += cnt
                i += 1
            # 将当前运算符及其输入统计信息存储到 operator_db 中
            self.operator_db[operator] = op_inps

    # 方法用于获取特定运算符的输入统计信息
    def get_inputs_for_operator(self, operator, dtype=None, device="cuda"):
    # 定义一个生成器函数，返回一个元组，包含可迭代对象和字典
    def __iter__(self) -> Generator[Tuple[Iterable[Any], Dict[str, Any]], None, None]:
        # 检查操作符是否在操作符数据库中
        assert (
            str(operator) in self.operator_db
        ), f"Could not find {operator}, must provide overload"

        # 如果操作符中包含"embedding"，则输出警告信息并返回空值
        if "embedding" in str(operator):
            log.warning("Embedding inputs NYI, input data cannot be randomized")
            yield
            return

        # 遍历操作符数据库中指定操作符的输入
        for line in self.operator_db[str(operator)].items():
            inps = line[0]

            # 反序列化输入参数
            args, kwargs = deserialize_args(inps)

            # 如果指定了数据类型且不是torch.float16，则将数据类型转换为指定类型
            if dtype and dtype != torch.float16:
                to_dtype = partial(map_to_dtype, dtype=dtype)
                args, kwargs = tree_map(to_dtype, (args, kwargs))

            # 如果指定了设备，则将数据移动到指定设备
            if device:
                to_device = partial(map_to_device, device=torch.device(device))
                args, kwargs = tree_map(to_device, (args, kwargs))

            yield args, kwargs

    # 获取所有操作符
    def get_all_ops(self):
        for key in self.operator_db.keys():
            try:
                op = eval(key)
            except AttributeError as ae:
                log.warning("Evaluating an op name into an OpOverload: %s", ae)
                continue
            yield op

    # 获取操作符的调用频率
    def get_call_frequency(self, op):
        assert (
            str(op) in self.operator_db
        ), f"Could not find {op}, must provide overload"

        count = 0
        for counter in self.operator_db[str(op)].values():
            count += counter
        return count

    # 合并操作符输入
    def merge(self, other):
        for operator, counter_dict in other.operator_db.items():
            for inps, cnt in counter_dict.items():
                self.operator_db[operator][inps] += cnt

    # 获取timm加载器
    @staticmethod
    def get_timm_loader():
        return OperatorInputsLoader._load_directory(TIMM_DIR)

    # 获取huggingface加载器
    @staticmethod
    def get_huggingface_loader():
        return OperatorInputsLoader._load_directory(HF_DIR)

    # 获取torchbench加载器
    @staticmethod
    def get_torchbench_loader():
        return OperatorInputsLoader._load_directory(TORCHBENCH_DIR)

    # 加载指定目录下的文件
    @staticmethod
    def _load_directory(inp_dir):
        assert os.path.isdir(inp_dir), inp_dir
        union = None
        for inp in os.listdir(inp_dir):
            if inp[-4:] != ".txt":
                continue
            path = os.path.join(inp_dir, inp)
            if union is None:
                union = OperatorInputsLoader(path)
            else:
                union.merge(OperatorInputsLoader(path))
        return union
```