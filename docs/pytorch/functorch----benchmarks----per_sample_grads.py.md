# `.\pytorch\functorch\benchmarks\per_sample_grads.py`

```py
import time  # 导入时间模块

import torchvision.models as models  # 导入PyTorch视觉模型
from opacus import PrivacyEngine  # 导入Opacus中的隐私引擎
from opacus.utils.module_modification import convert_batchnorm_modules  # 导入批归一化模块转换函数

import torch  # 导入PyTorch
import torch.nn as nn  # 导入PyTorch神经网络模块

from functorch import grad, make_functional, vmap  # 从functorch导入梯度计算、函数化和向量映射函数

device = "cuda"  # 指定设备为CUDA
batch_size = 128  # 批量大小
torch.manual_seed(0)  # 设置随机种子以保证结果可重现

model_functorch = convert_batchnorm_modules(models.resnet18(num_classes=10))  # 转换ResNet18模型为functorch模型
model_functorch = model_functorch.to(device)  # 将模型移动到指定设备
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

images = torch.randn(batch_size, 3, 32, 32, device=device)  # 生成随机图像张量
targets = torch.randint(0, 10, (batch_size,), device=device)  # 生成随机目标张量
func_model, weights = make_functional(model_functorch)  # 获取functorch模型的函数化形式和权重

def compute_loss(weights, image, target):
    images = image.unsqueeze(0)  # 在第0维度上增加一个维度
    targets = target.unsqueeze(0)  # 在第0维度上增加一个维度
    output = func_model(weights, images)  # 使用函数化模型计算输出
    loss = criterion(output, targets)  # 计算损失
    return loss  # 返回损失值

def functorch_per_sample_grad():
    compute_grad = grad(compute_loss)  # 获取损失函数关于权重的梯度函数
    compute_per_sample_grad = vmap(compute_grad, (None, 0, 0))  # 使用向量映射处理梯度函数

    start = time.time()  # 记录开始时间
    result = compute_per_sample_grad(weights, images, targets)  # 计算每个样本的梯度
    torch.cuda.synchronize()  # 同步CUDA流
    end = time.time()  # 记录结束时间

    return result, end - start  # 返回计算结果和时间差

torch.manual_seed(0)  # 重新设置随机种子以保证结果可重现
model_opacus = convert_batchnorm_modules(models.resnet18(num_classes=10))  # 转换ResNet18模型为opacus模型
model_opacus = model_opacus.to(device)  # 将模型移动到指定设备
criterion = nn.CrossEntropyLoss()  # 定义交叉熵损失函数

for p_f, p_o in zip(model_functorch.parameters(), model_opacus.parameters()):
    assert torch.allclose(p_f, p_o)  # 检查两个模型的参数是否近似相等，用于验证

privacy_engine = PrivacyEngine(
    model_opacus,
    sample_rate=0.01,
    alphas=[10, 100],
    noise_multiplier=1,
    max_grad_norm=10000.0,
)  # 创建Opacus隐私引擎对象

def opacus_per_sample_grad():
    start = time.time()  # 记录开始时间
    output = model_opacus(images)  # 使用Opacus模型进行前向传播
    loss = criterion(output, targets)  # 计算损失
    loss.backward()  # 反向传播计算梯度
    torch.cuda.synchronize()  # 同步CUDA流
    end = time.time()  # 记录结束时间
    expected = [p.grad_sample for p in model_opacus.parameters()]  # 获取期望的梯度样本
    for p in model_opacus.parameters():
        delattr(p, "grad_sample")  # 删除参数对象的grad_sample属性
        p.grad = None  # 清空梯度
    return expected, end - start  # 返回期望的梯度样本和时间差

for _ in range(5):
    _, seconds = functorch_per_sample_grad()  # 执行functorch每个样本梯度计算
    print(seconds)  # 打印运行时间

result, seconds = functorch_per_sample_grad()  # 再次执行functorch每个样本梯度计算
print(seconds)  # 打印运行时间

for _ in range(5):
    _, seconds = opacus_per_sample_grad()  # 执行opacus每个样本梯度计算
    print(seconds)  # 打印运行时间

expected, seconds = opacus_per_sample_grad()  # 再次执行opacus每个样本梯度计算
print(seconds)  # 打印运行时间

result = [r.detach() for r in result]  # 分离出functorch计算得到的梯度结果
print(len(result))  # 打印梯度结果列表的长度

# TODO: 以下显示计算出的每个样本梯度不同。
# 这让我有些担忧；我们应该与真实数据进行比较。
# for i, (r, e) in enumerate(list(zip(result, expected))[::-1]):
#     if torch.allclose(r, e, rtol=1e-5):
#         continue
#     print(-(i+1), ((r - e)/(e + 0.000001)).abs().max())
```