# `.\pytorch\torch\ao\quantization\_correct_bias.py`

```
# mypy: allow-untyped-defs
# 导入 PyTorch 模块
import torch
import torch.nn as nn
import torch.ao.nn.quantized as nnq

# 导入 PyTorch AO (Autograd Operator) 模块，用于量化
import torch.ao.quantization
import torch.ao.ns._numeric_suite as ns

# 导出的模块列表
__all__ = [
    "get_module",
    "parent_child_names",
    "get_param",
    "MeanShadowLogger",
    "bias_correction",
]

# 支持的标准模块集合，包括线性层和二维卷积层
_supported_modules = {nn.Linear, nn.Conv2d}
# 支持的量化模块集合，包括量化线性层和量化二维卷积层
_supported_modules_quantized = {nnq.Linear, nnq.Conv2d}

def get_module(model, name):
    """Given name of submodule, this function grabs the submodule from given model."""
    return dict(model.named_modules())[name]

def parent_child_names(name):
    """Split full name of submodule into parent submodule's full name and submodule's name."""
    split_name = name.rsplit('.', 1)
    if len(split_name) == 1:
        return '', split_name[0]
    else:
        return split_name[0], split_name[1]

def get_param(module, attr):
    """Get the parameter given a module and attribute.

    Sometimes the weights/bias attribute gives you the raw tensor, but sometimes
    gives a function that will give you the raw tensor, this function takes care of that logic
    """
    param = getattr(module, attr, None)
    if callable(param):
        return param()
    else:
        return param

class MeanShadowLogger(ns.Logger):
    """Mean Logger for a Shadow module.

    A logger for a Shadow module whose purpose is to record the rolling mean
    of the data passed to the floating point and quantized models
    """

    def __init__(self):
        """Set up initial values for float and quantized stats, count, float sum, and quant sum."""
        super().__init__()
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

    def forward(self, x, y):
        """Compute the average of quantized and floating-point data from modules.

        The inputs x,y are output data from the quantized and floating-point modules.
        x is for the quantized module, y is for the floating point module
        """
        if x.is_quantized:
            x = x.dequantize()

        self.count += 1
        if self.stats["quantized"] is None:
            self.stats["quantized"] = x
            self.quant_sum = x
        else:
            self.quant_sum += x
            self.stats["quantized"] = self.quant_sum / self.count

        if self.stats["float"] is None:
            self.stats["float"] = y
            self.float_sum = y
        else:
            self.float_sum += y
            self.stats["float"] = self.float_sum / self.count

    def clear(self):
        """Reset all statistics to initial values."""
        self.stats["float"] = None
        self.stats["quantized"] = None
        self.count = 0
        self.float_sum = None
        self.quant_sum = None

def bias_correction(float_model, quantized_model, img_data, target_modules=_supported_modules_quantized, neval_batches=None):
    """Perform bias correction on a module.

    This function corrects the biases between a floating-point model and a quantized model
    using the given image data. It evaluates the specified number of batches (neval_batches)
    to adjust the biases of the target quantized modules.
    """
    # 使用数值套件影子模块，记录浮点和量化模块的预期输出。使用这些数据来调整支持模块的偏差，以补偿量化引起的漂移。
    # 论文参考：https://arxiv.org/pdf/1906.04721.pdf（第4.2节）

    # 参数说明:
    # float_model: 作为参考的已训练模型，用于偏差校正的目标
    # quantized_model: float_model 的量化形式，需要进行偏差校正
    # img_data: 用于估计预期输出的校准数据（用于找出量化误差）
    # target_modules: 指定 quantized_model 中需要进行偏差校正的子模块（可以扩展到未量化的子模块）
    # neval_batches: 用于估计预期输出的批次数上限
    """
    # 使用 ns 模块准备带有存根的模型，用于偏差校正
    ns.prepare_model_with_stubs(float_model, quantized_model, _supported_modules, MeanShadowLogger)

    # 初始化一个空字典来存储未校正的模块
    uncorrected_modules = {}

    # 遍历 quantized_model 中的所有模块
    for name, submodule in quantized_model.named_modules():
        # 如果子模块的类型在 target_modules 中，则将其加入未校正的模块字典中
        if type(submodule) in target_modules:
            uncorrected_modules[name] = submodule

    # 遍历未校正的模块字典中的每个未校正模块
    for uncorrected_module in uncorrected_modules:
        # 获取 quantized_model 中指定名称的模块
        quantized_submodule = get_module(quantized_model, uncorrected_module)
        
        # 获取模块的偏差参数
        bias = get_param(quantized_submodule, 'bias')
        
        # 如果存在偏差参数
        if bias is not None:
            # 初始化计数器
            count = 0
            # 遍历校准数据中的每个数据项
            for data in img_data:
                # 对 quantized_model 进行前向传播
                quantized_model(data[0])
                count += 1
                # 达到指定批次数上限时停止
                if count == neval_batches:
                    break
            
            # 获取 quantized_model 的日志字典
            ob_dict = ns.get_logger_dict(quantized_model)
            # 获取未校正模块的父模块名称和子模块名称
            parent_name, _ = parent_child_names(uncorrected_module)
            
            # 获取浮点数统计和量化统计数据
            float_data = ob_dict[parent_name + '.stats']['float']
            quant_data = ob_dict[parent_name + '.stats']['quantized']
            
            # 计算量化误差
            quantization_error = quant_data - float_data
            dims = list(range(quantization_error.dim()))
            dims.remove(1)  # 不要在输出通道维度上取均值
            expected_error = torch.mean(quantization_error, dims)
            
            # 更新偏差
            updated_bias = bias.data - expected_error
            bias.data = updated_bias
            
            # 清空日志记录器中的数据
            for name, submodule in quantized_model.named_modules():
                if isinstance(submodule, MeanShadowLogger):
                    submodule.clear()
```