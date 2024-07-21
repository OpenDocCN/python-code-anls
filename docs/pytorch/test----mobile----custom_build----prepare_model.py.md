# `.\pytorch\test\mobile\custom_build\prepare_model.py`

```
"""
This is a script for end-to-end mobile custom build test purpose. It prepares
MobileNetV2 TorchScript model, and dumps root ops used by the model for custom
build script to create a tailored build which only contains these used ops.
"""

import yaml  # 导入yaml模块，用于处理YAML格式的数据
from torchvision import models  # 从torchvision模块中导入models子模块

import torch  # 导入torch模块，用于深度学习框架PyTorch的功能

# Download and trace the model.
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
# 下载MobileNetV2模型，并加载预训练的ImageNet权重
model.eval()  # 设置模型为评估模式，即禁用dropout等影响推断的操作
example = torch.rand(1, 3, 224, 224)  # 创建一个随机张量作为示例输入
traced_script_module = torch.jit.trace(model, example)
# 对模型进行跟踪，将模型和示例输入作为参数，得到一个追踪过的TorchScript模块

# Save traced TorchScript model.
traced_script_module.save("MobileNetV2.pt")
# 将追踪后的TorchScript模型保存为文件MobileNetV2.pt

# Dump root ops used by the model (for custom build optimization).
ops = torch.jit.export_opnames(traced_script_module)
# 获取模型使用的根操作（用于定制构建优化）

# Besides the ops used by the model, custom c++ client code might use some extra
# ops, too. For example, the dummy predictor.cpp driver in this test suite calls
# `aten::ones` to create all-one-tensor for testing purpose, which is not used
# by the MobileNetV2 model itself.
#
# This is less a problem for Android, where we expect users to use the limited
# set of Java APIs. To actually solve this problem, we probably need ask users
# to run code analyzer against their client code to dump these extra root ops.
# So it will require more work to switch to custom build with dynamic dispatch -
# in static dispatch case these extra ops will be kept by linker automatically.
#
# For CI purpose this one-off hack is probably fine? :)
EXTRA_CI_ROOT_OPS = ["aten::ones"]
# 额外的根操作，可能会被自定义的C++客户端代码使用，例如在测试套件中的dummy predictor.cpp驱动程序中调用`aten::ones`创建全1张量

ops.extend(EXTRA_CI_ROOT_OPS)
# 将额外的根操作加入到ops列表中

with open("MobileNetV2.yaml", "w") as output:
    yaml.dump(ops, output)
# 将ops列表写入到文件MobileNetV2.yaml中，使用YAML格式进行保存
```