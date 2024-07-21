# `.\pytorch\torch\ao\ns\_numeric_suite.py`

```py
# mypy: allow-untyped-defs
# 导入 PyTorch 库及其量化相关模块
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq
import torch.ao.nn.quantized.dynamic as nnqd
from torch.ao.quantization import prepare
from typing import Dict, List, Optional, Any, Union, Callable, Set

# 导入量化模块的默认比较输出模块列表
from torch.ao.quantization.quantization_mappings import (
    get_default_compare_output_module_list,
)

# 非叶子模块添加观察者的允许列表，包括动态量化和静态量化的线性层及 LSTM 层
NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST = {
    nnqd.Linear,
    nnq.Linear,
    nnqd.LSTM,
    nn.LSTM,
}

def _find_match(
    str_list: Union[Dict[str, Any], List[str]], key_str: str,
    postfix: str,
) -> Optional[str]:
    # 将 key_str 按 "." 分割成列表
    split_str = key_str.split(".")
    # 如果最后一个分割项等于 postfix
    if split_str[-1] == postfix:
        # 构造匹配字符串，去掉最后一个分割项
        match_string = "".join(key_str.split(".")[0:-1])
        # 遍历 str_list 中的字符串
        for s2 in str_list:
            # 构造两种模式的匹配字符串
            pattern1 = "".join(s2.split(".")[0:-1])
            pattern2 = "".join(s2.split(".")[0:-2])
            # 如果匹配成功则返回匹配的字符串 s2
            if match_string == pattern1:
                return s2
            if match_string == pattern2:
                return s2

        # 对于 "fc.weight" 和 "fc._packed_params._packed_params" 的特殊匹配情况
        if postfix == "_packed_params":
            # 构造匹配字符串，去掉最后两个分割项
            match_string = "".join(key_str.split(".")[0:-2])
            # 如果匹配字符串长度为 0，则返回 None
            if len(match_string) == 0:
                return None
            # 再次遍历 str_list 中的字符串
            for s2 in str_list:
                pattern1 = "".join(s2.split(".")[0:-1])
                pattern2 = "".join(s2.split(".")[0:-2])
                # 如果匹配成功则返回匹配的字符串 s2
                if match_string == pattern1:
                    return s2
                if match_string == pattern2:
                    return s2
        return None
    else:
        return None


def compare_weights(
    float_dict: Dict[str, Any], quantized_dict: Dict[str, Any]
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Compare the weights of the float module with its corresponding quantized
    module. Return a dict with key corresponding to module names and each entry being
    a dictionary with two keys 'float' and 'quantized', containing the float and
    quantized weights. This dict can be used to compare and compute the quantization
    error of the weights of float and quantized models.

    Example usage::

        wt_compare_dict = compare_weights(
            float_model.state_dict(), qmodel.state_dict())
        for key in wt_compare_dict:
            print(
                key,
                compute_error(
                    wt_compare_dict[key]['float'],
                    wt_compare_dict[key]['quantized'].dequantize()
                )
            )

    Args:
        float_dict: state dict of the float model
        quantized_dict: state dict of the quantized model

    Return:
        weight_dict: dict with key corresponding to module names and each entry being
        a dictionary with two keys 'float' and 'quantized', containing the float and
        quantized weights
    """
    # 记录一次量化比较权重的 API 使用
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_weights")
    # 初始化一个空的字典 weight_dict，用于存储比较结果
    weight_dict: Dict[str, Dict] = {}
    # 遍历 quantized_dict 字典的键
    for key in quantized_dict:
        # 在 float_dict 中查找与 key 相匹配的项，匹配的属性为 "weight"
        match_key = _find_match(float_dict, key, "weight")
        # 如果找到匹配项
        if match_key is not None:
            # 创建 key 对应的空字典
            weight_dict[key] = {}
            # 将 float_dict 中匹配项的值赋给 weight_dict[key]["float"]
            weight_dict[key]["float"] = float_dict[match_key]
            # 将 quantized_dict 中的值赋给 weight_dict[key]["quantized"]
            weight_dict[key]["quantized"] = quantized_dict[key]
            # 继续下一个循环
            continue
        
        # 若未找到与 key 匹配的项，尝试匹配 "_packed_params"
        match_key = _find_match(float_dict, key, "_packed_params")
        # 如果找到匹配项
        if match_key is not None:
            # 创建 key 对应的空字典
            weight_dict[key] = {}
            # 将 float_dict 中匹配项的值赋给 weight_dict[key]["float"]
            weight_dict[key]["float"] = float_dict[match_key]
            # 将 quantized_dict 中的值的第一个元素赋给 weight_dict[key]["quantized"]
            weight_dict[key]["quantized"] = quantized_dict[key][0]

        # 处理 LSTM 的情况
        split_str = key.split(".")
        # 如果 key 是形如 "*_all_weight_values.param.*" 的结构
        if split_str[-1] == "param" and split_str[-3] == "_all_weight_values":
            # 获取 LSTM 层号码
            layer = split_str[-2]
            # 获取模块名
            module_name = ".".join(split_str[:-3])
            # 构造 float_weight_ih_key 和 float_weight_hh_key
            float_weight_ih_key = module_name + ".weight_ih_l" + layer
            float_weight_hh_key = module_name + ".weight_hh_l" + layer
            # 如果这两个键存在于 float_dict 中
            if float_weight_ih_key in float_dict and float_weight_hh_key in float_dict:
                # 创建 key 对应的空字典
                weight_dict[key] = {}
                # 将 float_dict 中的 weight_ih_key 对应的值赋给 weight_dict[key]["float"]
                weight_dict[key]["float"] = float_dict[float_weight_ih_key]
                # 获取 quantized_dict[key] 的特定结构的值赋给 weight_dict[key]["quantized"]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][0].__getstate__()[0][0]
                )
                # 将 float_dict 中的 weight_hh_key 对应的值赋给 weight_dict[key]["float"]
                weight_dict[key]["float"] = float_dict[float_weight_hh_key]
                # 获取 quantized_dict[key] 的特定结构的值赋给 weight_dict[key]["quantized"]
                weight_dict[key]["quantized"] = (
                    quantized_dict[key].__getstate__()[0][4][1].__getstate__()[0][0]
                )

    # 返回包含匹配权重的字典 weight_dict
    return weight_dict
def _get_logger_dict_helper(
    mod: nn.Module, target_dict: Dict[str, Any],
    prefix: str = "",
) -> None:
    r"""This is the helper function for get_logger_dict

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module
        target_dict: the dictionary used to save all logger stats
    """

    def get_prefix(prefix):
        return prefix if prefix == "" else prefix + "."

    # Iterate over all child modules of the given module
    for name, child in mod.named_children():
        # Check if the child module is an instance of Logger
        if isinstance(child, Logger):
            # Save the statistics of the logger into the target dictionary
            target_dict[get_prefix(prefix) + "stats"] = child.stats
            break  # Break after the first logger found

    # Recursively call _get_logger_dict_helper for each child module
    for name, child in mod.named_children():
        module_prefix = get_prefix(prefix) + name if prefix else name
        _get_logger_dict_helper(child, target_dict, module_prefix)


def get_logger_dict(mod: nn.Module, prefix: str = "") -> Dict[str, Dict]:
    r"""Traverse the modules and save all logger stats into target dict.
    This is mainly used for quantization accuracy debug.

    Type of loggers supported:
        ShadowLogger: used to log the outputs of the quantized module and its matching float shadow module,
        OutputLogger: used to log the outputs of the modules

    Args:
        mod: module we want to save all logger stats
        prefix: prefix for the current module

    Return:
        target_dict: the dictionary used to save all logger stats

    """
    # Log the usage of the quantization API
    torch._C._log_api_usage_once("quantization_api._numeric_suite.get_logger_dict")

    # Initialize an empty dictionary to store logger stats
    target_dict: Dict[str, Dict] = {}

    # Call the helper function to populate the target_dict with logger stats recursively
    _get_logger_dict_helper(mod, target_dict, prefix)

    # Return the populated target dictionary
    return target_dict


class Logger(nn.Module):
    r"""Base class for stats logging
    """

    def __init__(self):
        super().__init__()
        # Initialize an empty dictionary to store statistics
        self.stats = {}
        # Set dtype to quint8 when observing activations in static quantization mode
        self.dtype = torch.quint8

    def forward(self, x):
        """
        """  # blank docblock to make autodoc happy
        pass


class ShadowLogger(Logger):
    r"""Class used in Shadow module to record the outputs of the original and
    shadow modules.
    """

    def __init__(self):
        super().__init__()
        # Initialize lists in stats dictionary to store float and quantized outputs
        self.stats["float"] = []
        self.stats["quantized"] = []

    def forward(self, x, y):
        """
        """  # blank docblock to make autodoc happy
        # Store quantized output in stats["quantized"] and float output in stats["float"]
        if len(x) > 1:
            x = x[0]
        if len(y) > 1:
            y = y[0]
        self.stats["quantized"].append(x.detach())
        self.stats["float"].append(y.detach())


class OutputLogger(Logger):
    r"""Class used to log the outputs of the module
    """

    def __init__(self):
        super().__init__()
        # Initialize a list in stats dictionary to store tensor values
        self.stats["tensor_val"] = []

    def forward(self, x):
        """
        """  # blank docblock to make autodoc happy
        # Append tensor x to stats["tensor_val"]
        self.stats["tensor_val"].append(x)
        return x
# 定义一个函数 _convert_tuple_to_list，将输入的元组递归转换成列表，否则直接返回输入
def _convert_tuple_to_list(t: Any) -> Any:
    return [_convert_tuple_to_list(x) for x in t] if type(t) is tuple else t

# 定义一个函数 _dequantize_tensor_list，将输入的列表递归地去量化（如果是量化的话），否则返回输入
def _dequantize_tensor_list(t: Any) -> Any:
    return (
        [_dequantize_tensor_list(x) for x in t]
        if type(t) is list
        else t.dequantize()  # 如果是量化的张量，则去量化
        if t.is_quantized  # 检查张量是否被量化
        else t  # 否则直接返回张量
    )

# 定义一个名为 Shadow 的类，继承自 nn.Module
class Shadow(nn.Module):
    r"""Shadow module attaches the float module to its matching quantized module
    as the shadow. Then it uses Logger module to process the outputs of both
    modules.

    Args:
        q_module: module quantized from float_module that we want to shadow
        float_module: float module used to shadow q_module
        logger_cls: type of logger used to process the outputs of q_module and
            float_module. ShadowLogger or custom loggers can be used.
    """

    # 初始化方法，接收 quantized 模块 q_module、对应的 float 模块 float_module 和 logger 类 logger_cls
    def __init__(self, q_module, float_module, logger_cls):
        super().__init__()
        # 存储 quantized 模块
        self.orig_module = q_module
        # 存储 float 模块作为影子模块
        self.shadow_module = float_module
        # 创建去量化器对象
        self.dequant = nnq.DeQuantize()
        # 创建 logger 对象
        self.logger = logger_cls()

    # 前向传播方法，接收任意数量的输入 *x，并返回 torch.Tensor 类型的输出
    def forward(self, *x) -> torch.Tensor:
        """
        """  # 空白的文档字符串，用于让自动文档生成工具正常工作
        # 将输入 *x 转换为列表 xl
        xl = _convert_tuple_to_list(x)
        # 使用 quantized 模块处理输入 xl，得到输出
        output = self.orig_module(*xl)
        # 对 xl 中的每个张量进行去量化处理，得到浮点数列表 xl_float
        xl_float = _dequantize_tensor_list(xl)
        # 使用 float 模块处理去量化后的输入 xl_float，得到影子模块的输出
        shadow_output = self.shadow_module(*xl_float)
        # 使用 logger 记录 quantized 模块的输出和影子模块的输出
        self.logger(output, shadow_output)
        # 返回 quantized 模块的输出
        return output

    # 加法操作方法，接收两个 torch.Tensor 类型的输入 x 和 y，返回 torch.Tensor 类型的输出
    def add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        """  # 空白的文档字符串，用于让自动文档生成工具正常工作
        # 使用 quantized 模块执行输入 x 和 y 的加法操作，得到输出
        output = self.orig_module.add(x, y)
        # 对输入 x 和 y 分别进行去量化处理
        x = x.dequantize()
        y = y.dequantize()
        # 使用 float 模块执行去量化后的输入 x 和 y 的加法操作，得到影子模块的输出
        shadow_output = self.shadow_module.add(x, y)
        # 使用 logger 记录 quantized 模块的输出和影子模块的输出
        self.logger(output, shadow_output)
        # 返回 quantized 模块的输出
        return output

    # 标量加法操作方法，接收 torch.Tensor 类型的输入 x 和 float 类型的输入 y，返回 torch.Tensor 类型的输出
    def add_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        """
        """  # 空白的文档字符串，用于让自动文档生成工具正常工作
        # 使用 quantized 模块执行输入 x 和标量 y 的加法操作，得到输出
        output = self.orig_module.add_scalar(x, y)
        # 对输入 x 进行去量化处理
        x = x.dequantize()
        # 使用 float 模块执行去量化后的输入 x 和标量 y 的加法操作，得到影子模块的输出
        shadow_output = self.shadow_module.add_scalar(x, y)
        # 使用 logger 记录 quantized 模块的输出和影子模块的输出
        self.logger(output, shadow_output)
        # 返回 quantized 模块的输出
        return output

    # 乘法操作方法，接收两个 torch.Tensor 类型的输入 x 和 y，返回 torch.Tensor 类型的输出
    def mul(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        """  # 空白的文档字符串，用于让自动文档生成工具正常工作
        # 使用 quantized 模块执行输入 x 和 y 的乘法操作，得到输出
        output = self.orig_module.mul(x, y)
        # 对输入 x 和 y 分别进行去量化处理
        x = x.dequantize()
        y = y.dequantize()
        # 使用 float 模块执行去量化后的输入 x 和 y 的乘法操作，得到影子模块的输出
        shadow_output = self.shadow_module.mul(x, y)
        # 使用 logger 记录 quantized 模块的输出和影子模块的输出
        self.logger(output, shadow_output)
        # 返回 quantized 模块的输出
        return output

    # 标量乘法操作方法，接收 torch.Tensor 类型的输入 x 和 float 类型的输入 y，返回 torch.Tensor 类型的输出
    def mul_scalar(self, x: torch.Tensor, y: float) -> torch.Tensor:
        """
        """  # 空白的文档字符串，用于让自动文档生成工具正常工作
        # 使用 quantized 模块执行输入 x 和标量 y 的乘法操作，得到输出
        output = self.orig_module.mul_scalar(x, y)
        # 对输入 x 进行去量化处理
        x = x.dequantize()
        # 使用 float 模块执行去量化后的输入 x 和标量 y 的乘法操作，得到影子模块的输出
        shadow_output = self.shadow_module.mul_scalar(x, y)
        # 使用 logger 记录 quantized 模块的输出和影子模块的输出
        self.logger(output, shadow_output)
        # 返回 quantized 模块的输出
        return output
    def cat(self, x: List[torch.Tensor], dim: int = 0) -> torch.Tensor:
        """
        """  # 空白文档块，用于自动文档工具
        # 调用原始模块的 cat 方法，对输入张量列表 x 进行在指定维度 dim 上的拼接
        output = self.orig_module.cat(x, dim)
        # 对输入张量列表 x 中的每个张量执行去量化操作
        x = [y.dequantize() for y in x]
        # 使用影子模块的 cat 方法，对去量化后的输入张量列表 x 进行在指定维度 dim 上的拼接
        shadow_output = self.shadow_module.cat(x, dim)
        # 记录原始输出和影子输出的日志
        self.logger(output, shadow_output)
        # 返回原始模块拼接后的输出张量
        return output

    def add_relu(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        """  # 空白文档块，用于自动文档工具
        # 调用原始模块的 add_relu 方法，对输入张量 x 和 y 执行 ReLU 相加操作
        output = self.orig_module.add_relu(x, y)
        # 对输入张量 x 和 y 分别执行去量化操作
        x = x.dequantize()
        y = y.dequantize()
        # 使用影子模块的 add_relu 方法，对去量化后的输入张量 x 和 y 执行 ReLU 相加操作
        shadow_output = self.shadow_module.add_relu(x, y)
        # 记录原始输出和影子输出的日志
        self.logger(output, shadow_output)
        # 返回原始模块执行 ReLU 相加后的输出张量
        return output
# 准备模型，根据指定的条件为模型添加桩（stubs）。如果浮点模块的类型在module_swap_list中，
# 则将浮点模块附加到其匹配的量化模块上作为其影子。
def prepare_model_with_stubs(
    float_module: nn.Module, q_module: nn.Module,
    module_swap_list: Set[type], logger_cls: Callable,
) -> None:
    # 记录使用量化 API 的调用
    torch._C._log_api_usage_once("quantization_api._numeric_suite.prepare_model_with_stubs")

    # 获取浮点模块的子模块字典
    float_module_children = {}
    for name, mod in float_module.named_children():
        float_module_children[name] = mod

    # 为量化模块重新分配子模块
    reassign = {}
    for name, mod in q_module.named_children():
        # 如果浮点模块没有对应的子模块，则跳过
        if name not in float_module_children:
            continue

        # 获取对应的浮点模块
        float_mod = float_module_children[name]

        # 如果浮点模块的类型不在module_swap_list中，则递归调用prepare_model_with_stubs
        if type(float_mod) not in module_swap_list:
            prepare_model_with_stubs(float_mod, mod, module_swap_list, logger_cls)

        # 只有当浮点模块的类型在module_swap_list中，且浮点模块和量化模块的类型不相同时，
        # 才插入影子模块
        if type(float_mod) in module_swap_list and not _is_identical_module_type(mod, float_mod):
            reassign[name] = Shadow(mod, float_mod, logger_cls)

    # 将重新分配的影子模块添加到量化模块中
    for key, value in reassign.items():
        q_module._modules[key] = value

# 比较模型的桩（stubs）
def _is_identical_module_type(mod1, mod2):
    # 比较两个模块是否具有相同的类型
    mod1_module_types = [type(mod) for mod in mod1.modules()]
    mod2_module_types = [type(mod) for mod in mod2.modules()]
    return mod1_module_types == mod2_module_types

# 比较模型的桩（stubs）
def compare_model_stub(
    float_model: nn.Module, q_model: nn.Module, module_swap_list: Set[type],
    *data, logger_cls=ShadowLogger
) -> Dict[str, Dict]:
    # 比较模型中量化模块和其对应的浮点模块，使用相同的输入。返回一个字典，其中键对应
    # 模块名称，每个条目都是一个字典，包含两个键'float'和'quantized'，分别包含量化模块
    # 和其匹配的浮点影子模块的输出张量。这个字典可用于比较和计算模块级的量化误差。

    # 首先调用prepare_model_with_stubs()来交换我们想要比较的量化模块和影子模块，
    # 它接受量化模块、对应的浮点模块和日志记录器作为输入，并在内部创建一个前向路径，
    # 使浮点模块成为量化模块的影子，共享相同的输入。日志记录器可以自定义，默认日志记录器是ShadowLogger。
    pass  # 这里只是占位符，函数体尚未实现，因此没有实际的代码块需要注释
    # 记录 API 使用情况，仅记录一次，用于量化 API 的比较模型存根功能
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_model_stub")
    # 使用存根准备模型，将浮点模型转换为量化模型，并在指定的浮点模块类型处添加影子模块
    prepare_model_with_stubs(float_model, q_model, module_swap_list, logger_cls)
    # 运行准备好的量化模型，对输入数据进行处理
    q_model(*data)
    # 获取量化模型的日志字典，包含浮点模型和量化模型的输出信息
    ob_dict = get_logger_dict(q_model)
    # 返回日志字典，其中包含了浮点模型和量化模型的输出数据
    return ob_dict
# 导入需要的模块
from typing import Dict
import torch
import torch.nn as nn
from torch.quantization import prepare
from torch.ao.quantization import OutputLogger

# 查找在浮点和量化模块之间匹配的激活
def get_matching_activations(
    float_module: nn.Module, q_module: nn.Module,
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Find the matching activation between float and quantized modules.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module

    Return:
        act_dict: dict with key corresponding to quantized module names and each
        entry being a dictionary with two keys 'float' and 'quantized', containing
        the matching float and quantized activations
    """
    # 记录 API 使用情况
    torch._C._log_api_usage_once("quantization_api._numeric_suite.get_matching_activations")
    
    # 获取浮点模块的日志字典
    float_dict = get_logger_dict(float_module)
    # 获取量化模块的日志字典
    quantized_dict = get_logger_dict(q_module)
    
    # 初始化匹配激活字典
    act_dict: Dict[str, Dict] = {}
    
    # 遍历量化模块的日志字典
    for key in quantized_dict:
        # 如果量化模块的张量值列表为空，则跳过
        if len(quantized_dict[key]["tensor_val"]) == 0:
            continue
        
        # 在浮点模块的日志字典中查找与当前量化模块名称匹配的键
        match_key = _find_match(sorted(float_dict, reverse=True), key, "stats")
        
        # 如果找到匹配的键，则将匹配的浮点和量化激活值存入字典中
        if match_key is not None:
            act_dict[key] = {}
            act_dict[key]["float"] = float_dict[match_key]["tensor_val"]
            act_dict[key]["quantized"] = quantized_dict[key]["tensor_val"]
    
    # 返回匹配的激活字典
    return act_dict


# 为模型准备输出
def prepare_model_outputs(
    float_module: nn.Module,
    q_module: nn.Module,
    logger_cls=OutputLogger,
    allow_list=None
) -> None:
    r"""Prepare the model by attaching the logger to both float module
    and quantized module if they are in the allow_list.

    Args:
        float_module: float module used to generate the q_module
        q_module: module quantized from float_module
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger
    """
    # 记录 API 使用情况
    torch._C._log_api_usage_once("quantization_api._numeric_suite.prepare_model_outputs")
    
    # 如果 allow_list 为 None，则使用默认的比较输出模块列表
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()

    # 设置量化配置为 debug 模式
    qconfig_debug = torch.ao.quantization.QConfig(activation=logger_cls, weight=None)
    
    # 将浮点模块的 qconfig 设置为 debug 模式
    float_module.qconfig = qconfig_debug  # type: ignore[assignment]
    # 为浮点模块准备量化配置
    prepare(float_module, inplace=True, allow_list=allow_list, prepare_custom_config_dict={})
    
    # 将量化模块的 qconfig 设置为 debug 模式
    q_module.qconfig = qconfig_debug  # type: ignore[assignment]
    # 为量化模块准备量化配置
    prepare(
        q_module,
        inplace=True,
        allow_list=allow_list,
        observer_non_leaf_module_list=NON_LEAF_MODULE_TO_ADD_OBSERVER_ALLOW_LIST,
        prepare_custom_config_dict={}
    )


# 比较模型输出
def compare_model_outputs(
    float_model: nn.Module,
    q_model: nn.Module,
    *data,
    logger_cls=OutputLogger,
    allow_list=None
) -> Dict[str, Dict[str, torch.Tensor]]:
    r"""Compare output activations between float and quantized models at
    corresponding locations for the same input. Return a dict with key corresponding
    to quantized module names and each entry being a dictionary with two keys
    'float' and 'quantized', containing the activations of quantized model and
    """
    """
    Log API usage for comparing model outputs in the context of quantization.

    Args:
        float_model: float model used to generate the q_model
        q_model: model quantized from float_model
        data: input data used to run the prepared float_model and q_model
        logger_cls: type of logger to be attached to float_module and q_module
        allow_list: list of module types to attach logger

    Return:
        act_compare_dict: dict with key corresponding to quantized module names
        and each entry being a dictionary with two keys 'float' and 'quantized',
        containing the matching float and quantized activations
    """
    # 记录一次 API 使用，用于量化模型输出比较
    torch._C._log_api_usage_once("quantization_api._numeric_suite.compare_model_outputs")

    # 如果 allow_list 为空，则使用默认的比较输出模块列表
    if allow_list is None:
        allow_list = get_default_compare_output_module_list()

    # 准备模型输出，包括连接日志记录器到 float_model 和 q_model
    prepare_model_outputs(float_model, q_model, logger_cls, allow_list)

    # 运行准备好的 float_model 和 q_model，使用输入数据
    float_model(*data)
    q_model(*data)

    # 获取匹配的激活值字典，包括浮点模型和量化模型的匹配激活
    act_compare_dict = get_matching_activations(float_model, q_model)

    # 返回激活值比较结果的字典
    return act_compare_dict
    ```
```