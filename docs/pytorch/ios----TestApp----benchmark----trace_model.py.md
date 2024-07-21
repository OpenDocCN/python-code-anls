# `.\pytorch\ios\TestApp\benchmark\trace_model.py`

```
# 从torchvision模块中导入models子模块，该子模块包含预定义的视觉模型
from torchvision import models

# 导入torch模块，用于处理神经网络和张量操作
import torch

# 从torch.utils.mobile_optimizer模块中导入optimize_for_mobile函数，用于优化移动端模型
from torch.utils.mobile_optimizer import optimize_for_mobile

# 使用预定义的MobileNetV2模型，加载预训练的ImageNet权重
model = models.mobilenet_v2(weights=models.MobileNetV2Pretrained.IMAGENET1K_V1)

# 将模型设置为评估模式，即禁用dropout等训练特定操作
model.eval()

# 创建一个示例张量example，形状为[1, 3, 224, 224]，模拟输入给模型的数据
example = torch.rand(1, 3, 224, 224)

# 使用torch.jit.trace函数对模型进行脚本化追踪，以获得脚本化模块
traced_script_module = torch.jit.trace(model, example)

# 对脚本化的模块进行优化，以准备在移动设备上运行
optimized_scripted_module = optimize_for_mobile(traced_script_module)

# 将优化后的脚本化模块保存到文件"../models/model.pt"中
torch.jit.save(optimized_scripted_module, "../models/model.pt")

# 使用_lite_interpreter方法将优化后的脚本化模块保存为适合Lite解释器的文件"../models/model.ptl"
exported_optimized_scripted_module = (
    optimized_scripted_module._save_for_lite_interpreter("../models/model.ptl")
)
```