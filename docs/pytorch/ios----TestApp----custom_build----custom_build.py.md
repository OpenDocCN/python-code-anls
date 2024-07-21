# `.\pytorch\ios\TestApp\custom_build\custom_build.py`

```
# 导入yaml模块，用于将操作名列表写入YAML格式文件
import yaml
# 从torchvision模块中导入models子模块，该子模块包含预定义的模型架构
from torchvision import models
# 导入torch模块，用于基本的张量操作和模型训练
import torch

# 使用预定义的MobileNetV2模型架构，加载预训练的ImageNet权重
model = models.mobilenet_v2(weights=models.MobileNetV2Weights.IMAGENET1K_V1)
# 将模型设置为评估模式，这通常意味着关闭了特定于训练的操作，如Dropout
model.eval()
# 创建一个示例张量，形状为[1, 3, 224, 224]，代表一张RGB图像（224x224）
example = torch.rand(1, 3, 224, 224)
# 使用torch.jit.trace方法对模型和示例进行追踪，以创建一个脚本化的模型
traced_script_module = torch.jit.trace(model, example)
# 使用torch.jit.export_opnames方法获取脚本化模型中的操作名列表
ops = torch.jit.export_opnames(traced_script_module)

# 打开一个文件"mobilenetv2.yaml"，并使用yaml.dump方法将操作名列表ops写入文件
with open("mobilenetv2.yaml", "w") as output:
    yaml.dump(ops, output)
```