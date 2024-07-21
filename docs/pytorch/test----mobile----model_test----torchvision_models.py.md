# `.\pytorch\test\mobile\model_test\torchvision_models.py`

```
from torchvision import models  # 导入 torchvision 库中的 models 模块，用于加载预定义的神经网络模型

import torch  # 导入 PyTorch 库
from torch.utils.bundled_inputs import augment_model_with_bundled_inputs  # 导入 bundled_inputs 模块中的函数
from torch.utils.mobile_optimizer import optimize_for_mobile  # 导入 mobile_optimizer 模块中的优化函数


class MobileNetV2Module:
    def getModule(self):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()  # 设置模型为评估模式，不进行梯度计算
        example = torch.zeros(1, 3, 224, 224)  # 创建一个示例输入张量
        traced_script_module = torch.jit.trace(model, example)  # 对模型进行 TorchScript 脚本化
        optimized_module = optimize_for_mobile(traced_script_module)  # 对 TorchScript 模块进行移动端优化
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),  # 使用 bundled_inputs 函数为模型添加示例输入
            ],
        )
        optimized_module(example)  # 对优化后的模块进行一次前向传播，确保可用性
        return optimized_module  # 返回优化后的模块对象


class MobileNetV2VulkanModule:
    def getModule(self):
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module, backend="vulkan")  # 使用 Vulkan 后端进行移动端优化
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module


class Resnet18Module:
    def getModule(self):
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.eval()
        example = torch.zeros(1, 3, 224, 224)
        traced_script_module = torch.jit.trace(model, example)
        optimized_module = optimize_for_mobile(traced_script_module)
        augment_model_with_bundled_inputs(
            optimized_module,
            [
                (example,),
            ],
        )
        optimized_module(example)
        return optimized_module
```