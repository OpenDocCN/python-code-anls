# `.\pytorch\test\quantization\core\experimental\apot_fx_graph_mode_ptq.py`

```
import torch
import torch.nn as nn
import torch.ao.quantization
from torchvision.models.quantization.resnet import resnet18
from torch.ao.quantization.experimental.quantization_helper import (
    evaluate,
    prepare_data_loaders
)

# 设置验证数据集：完整的 ImageNet 数据集
data_path = '~/my_imagenet/'

# 准备数据加载器，用于训练和测试
data_loader, data_loader_test = prepare_data_loaders(data_path)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 加载预训练的 ResNet-18 模型（全精度）
float_model = resnet18(pretrained=True)
float_model.eval()

# 深拷贝模型以便保留原始模型
import copy
model_to_quantize = copy.deepcopy(float_model)

# 将深拷贝的模型设置为评估模式
model_to_quantize.eval()

"""
Prepare models
"""

# 注意：这部分内容是临时的，将在正式发布后暴露给 torch.ao.quantization
from torch.ao.quantization.quantize_fx import prepare_qat_fx

# 定义用于模型校准的函数
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for image, target in data_loader:
            model(image)

# 导入量化配置
from torch.ao.quantization.experimental.qconfig import (
    uniform_qconfig_8bit,
    apot_weights_qconfig_8bit,
    apot_qconfig_8bit,
    uniform_qconfig_4bit,
    apot_weights_qconfig_4bit,
    apot_qconfig_4bit
)

"""
Prepare full precision model
"""
# 使用全精度模型进行评估
full_precision_model = float_model
top1, top5 = evaluate(full_precision_model, criterion, data_loader_test)
print(f"Model #0 Evaluation accuracy on test dataset: {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model PTQ for specified qconfig for torch.nn.Linear
"""
# 准备指定 qconfig 的 PTQ 模型函数
def prepare_ptq_linear(qconfig):
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}
    # 对深拷贝的全精度模型应用量化操作，包括模块融合和插入观察者
    prepared_model = prepare_qat_fx(copy.deepcopy(float_model), qconfig_dict)
    # 在测试数据上运行校准过程
    calibrate(prepared_model, data_loader_test)
    return prepared_model

"""
Prepare model with uniform activation, uniform weight
b=8, k=2
"""

# 使用均匀激活和均匀权重配置（8位，2位小数）准备 PTQ 模型
prepared_model = prepare_ptq_linear(uniform_qconfig_8bit)
quantized_model = convert_fx(prepared_model)  # 将校准后的模型转换为量化模型  # noqa: F821

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, uniform weight
b=4, k=2
"""

# 使用均匀激活和均匀权重配置（4位，2位小数）准备 PTQ 模型
prepared_model = prepare_ptq_linear(uniform_qconfig_4bit)
quantized_model = convert_fx(prepared_model)  # 将校准后的模型转换为量化模型  # noqa: F821

top1, top5 = evaluate(quantized_model, criterion, data_loader_test)  # noqa: F821
print(f"Model #1 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
Prepare model with uniform activation, APoT weight
(b=8, k=2)
"""

# 使用均匀激活和 APoT 权重配置（8位，2位小数）准备 PTQ 模型
prepared_model = prepare_ptq_linear(apot_weights_qconfig_8bit)

top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")
# 准备具有统一激活函数和 APoT 权重的模型
prepared_model = prepare_ptq_linear(apot_weights_qconfig_4bit)

# 评估准备好的模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
# 打印模型 #2 在测试数据集上的评估准确率 (b=4, k=2)
print(f"Model #2 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

# 准备具有 APoT 激活函数和权重的模型
prepared_model = prepare_ptq_linear(apot_qconfig_8bit)

# 评估准备好的模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
# 打印模型 #3 在测试数据集上的评估准确率 (b=8, k=2)
print(f"Model #3 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

# 准备具有 APoT 激活函数和权重的模型
prepared_model = prepare_ptq_linear(apot_qconfig_4bit)

# 评估准备好的模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
# 打印模型 #3 在测试数据集上的评估准确率 (b=4, k=2)
print(f"Model #3 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

# 准备贪婪模式量化的模型
eager_quantized_model = resnet18(pretrained=True, quantize=True).eval()

# 评估贪婪模式量化的模型在测试数据集上的性能
top1, top5 = evaluate(eager_quantized_model, criterion, data_loader_test)
# 打印贪婪模式量化模型在测试数据集上的评估准确率
print(f"Eager mode quantized model evaluation accuracy on test dataset: {top1.avg:2.2f}, {top5.avg:2.2f}")
```