# `.\pytorch\android\test_app\make_assets.py`

```
# 导入torchvision库中的模型模块
from torchvision import models

# 导入torch库
import torch

# 打印当前torch版本号
print(torch.version.__version__)

# 实例化一个预训练的 ResNet-18 模型，使用 ImageNet 数据集的预训练权重
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
# 设置模型为评估模式，即不进行训练
resnet18.eval()
# 对 ResNet-18 模型进行追踪，并保存为一个 Torch 脚本文件到指定路径
resnet18_traced = torch.jit.trace(resnet18, torch.rand(1, 3, 224, 224)).save(
    "app/src/main/assets/resnet18.pt"
)

# 实例化一个预训练的 ResNet-50 模型，使用 ImageNet 数据集的预训练权重
resnet50 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
# 设置模型为评估模式，即不进行训练
resnet50.eval()
# 对 ResNet-50 模型进行追踪，并保存为一个 Torch 脚本文件到指定路径
torch.jit.trace(resnet50, torch.rand(1, 3, 224, 224)).save(
    "app/src/main/assets/resnet50.pt"
)

# 实例化一个预训练的 MobileNetV2 模型，开启量化选项，并使用预训练权重
mobilenet2q = models.quantization.mobilenet_v2(pretrained=True, quantize=True)
# 设置模型为评估模式，即不进行训练
mobilenet2q.eval()
# 对 MobileNetV2 模型进行追踪，并保存为一个 Torch 脚本文件到指定路径
torch.jit.trace(mobilenet2q, torch.rand(1, 3, 224, 224)).save(
    "app/src/main/assets/mobilenet2q.pt"
)
```