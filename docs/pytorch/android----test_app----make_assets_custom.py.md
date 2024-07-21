# `.\pytorch\android\test_app\make_assets_custom.py`

```
"""
This is a script for PyTorch Android custom selective build test. It prepares
MobileNetV2 TorchScript model, and dumps root ops used by the model for custom
build script to create a tailored build which only contains these used ops.
"""

import yaml  # 导入yaml模块，用于处理YAML格式的数据
from torchvision import models  # 从torchvision库中导入models模块

import torch  # 导入torch库

# Download and trace the model.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)  # 下载MobileNetV2模型的预训练权重
model.eval()  # 设置模型为评估模式，即禁用dropout和batch normalization的影响
example = torch.rand(1, 3, 224, 224)  # 创建一个示例输入张量，形状为[1, 3, 224, 224]

# TODO: create script model with `torch.jit.script`
traced_script_module = torch.jit.trace(model, example)  # 使用示例输入张量对模型进行追踪编译成TorchScript

# Save traced TorchScript model.
traced_script_module.save("MobileNetV2.pt")  # 将追踪后的TorchScript模型保存为文件MobileNetV2.pt

# Dump root ops used by the model (for custom build optimization).
ops = torch.jit.export_opnames(traced_script_module)  # 导出模型使用的根操作的名称

with open("MobileNetV2.yaml", "w") as output:
    yaml.dump(ops, output)  # 将导出的根操作名称写入YAML文件MobileNetV2.yaml
```