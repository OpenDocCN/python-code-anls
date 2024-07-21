# `.\pytorch\ios\TestApp\benchmark\coreml_backend.py`

```py
# 导入torchvision模块中的models子模块，用于加载预训练的MobileNetV2模型
from torchvision import models

# 导入torch模块，用于深度学习框架支持
import torch

# 导入torch.backends._coreml.preprocess模块中的CompileSpec、CoreMLComputeUnit、TensorSpec类
from torch.backends._coreml.preprocess import CompileSpec, CoreMLComputeUnit, TensorSpec


# 定义函数mobilenetv2_spec，返回MobileNetV2模型的编译规格
def mobilenetv2_spec():
    return {
        # 定义"forward"方法的编译规格
        "forward": CompileSpec(
            inputs=(
                TensorSpec(
                    shape=[1, 3, 224, 224],  # 定义输入张量的形状
                ),
            ),
            outputs=(
                TensorSpec(
                    shape=[1, 1000],  # 定义输出张量的形状
                ),
            ),
            backend=CoreMLComputeUnit.CPU,  # 指定后端为CoreML的CPU计算单元
            allow_low_precision=True,  # 允许低精度计算
        ),
    }


# 主函数，用于执行模型的加载、转换和保存
def main():
    # 使用预训练的MobileNetV2模型，加载ImageNet数据集上的权重
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()  # 将模型设置为评估模式，不进行梯度计算
    example = torch.rand(1, 3, 224, 224)  # 创建一个随机输入张量作为示例
    model = torch.jit.trace(model, example)  # 对模型进行跟踪，以便后续转换
    compile_spec = mobilenetv2_spec()  # 获取MobileNetV2模型的编译规格
    mlmodel = torch._C._jit_to_backend("coreml", model, compile_spec)  # 将模型转换为CoreML格式
    print(mlmodel._c._get_method("forward").graph)  # 打印转换后模型的前向方法的图结构
    mlmodel._save_for_lite_interpreter("../models/model_coreml.ptl")  # 保存为用于轻量级解释器的模型
    torch.jit.save(mlmodel, "../models/model_coreml.pt")  # 保存为.pth格式的模型文件


if __name__ == "__main__":
    main()
```