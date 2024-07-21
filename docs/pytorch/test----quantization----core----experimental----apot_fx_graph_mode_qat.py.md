# `.\pytorch\test\quantization\core\experimental\apot_fx_graph_mode_qat.py`

```
# 导入需要的模块和函数
from torchvision.models.quantization.resnet import resnet18  # 导入resnet18模型
from torch.ao.quantization.experimental.quantization_helper import (
    evaluate,  # 导入评估函数evaluate
    prepare_data_loaders,  # 导入准备数据加载器函数prepare_data_loaders
    training_loop  # 导入训练循环函数training_loop
)

# 定义数据集路径和批量大小
data_path = '~/my_imagenet/'
train_batch_size = 30
eval_batch_size = 50

# 准备训练和测试数据加载器
data_loader, data_loader_test = prepare_data_loaders(data_path)

# 定义交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 加载预训练的ResNet-18模型，并设为评估模式
float_model = resnet18(pretrained=True)
float_model.eval()

# 深拷贝模型以备后用
import copy
model_to_quantize = copy.deepcopy(float_model)
model_to_quantize.eval()

"""
为torch.nn.Linear准备模型QAT，使用指定的量化配置qconfig
"""
def prepare_qat_linear(qconfig):
    qconfig_dict = {"object_type": [(torch.nn.Linear, qconfig)]}  # 定义包含量化配置的字典
    prepared_model = prepare_fx(copy.deepcopy(float_model), qconfig_dict)  # 深拷贝模型并应用量化配置，插入观察器
    training_loop(prepared_model, criterion, data_loader)  # 执行训练循环
    prepared_model.eval()  # 设为评估模式
    return prepared_model

"""
使用统一激活、统一权重准备模型
b=8, k=2
"""
prepared_model = prepare_qat_linear(uniform_qconfig_8bit)  # 使用统一激活、统一权重的8位量化配置准备模型

# 评估模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
使用统一激活、统一权重准备模型
b=4, k=2
"""
prepared_model = prepare_qat_linear(uniform_qconfig_4bit)  # 使用统一激活、统一权重的4位量化配置准备模型

# 评估模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #1 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
使用统一激活、APoT权重准备模型
b=8, k=2
"""
prepared_model = prepare_qat_linear(apot_weights_qconfig_8bit)  # 使用统一激活、APoT权重的8位量化配置准备模型

# 评估模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
使用统一激活、APoT权重准备模型
b=4, k=2
"""
prepared_model = prepare_qat_linear(apot_weights_qconfig_4bit)  # 使用统一激活、APoT权重的4位量化配置准备模型

# 评估模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #2 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
使用APoT激活和权重准备模型
b=8, k=2
"""
prepared_model = prepare_qat_linear(apot_qconfig_8bit)  # 使用APoT激活和权重的8位量化配置准备模型

# 评估模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=8, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")

"""
使用APoT激活和权重准备模型
b=4, k=2
"""
prepared_model = prepare_qat_linear(apot_qconfig_4bit)  # 使用APoT激活和权重的4位量化配置准备模型

# 评估模型在测试数据集上的性能
top1, top5 = evaluate(prepared_model, criterion, data_loader_test)
print(f"Model #3 Evaluation accuracy on test dataset (b=4, k=2): {top1.avg:2.2f}, {top5.avg:2.2f}")
```