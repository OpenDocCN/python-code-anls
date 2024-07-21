# `.\pytorch\benchmarks\tensorexpr\pt_engine.py`

```
# 导入 torch 库，用于张量操作和神经网络功能
import torch


# 定义一个使用 Torch 张量操作的引擎类
class TorchTensorEngine:
    # 返回指定形状的随机张量
    def rand(self, shape, device=None, dtype=None, requires_grad=False):
        return torch.rand(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    # 返回指定形状的标准正态分布的随机张量
    def randn(self, shape, device=None, dtype=None, requires_grad=False):
        return torch.randn(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    # 返回指定形状的随机张量，按照 NCHW 格式（用于卷积网络）
    def nchw_rand(self, shape, device=None, requires_grad=False):
        return self.rand(shape, device=device, requires_grad=requires_grad)

    # 重置引擎状态（未实现功能）
    def reset(self, _):
        pass

    # 返回与输入张量相同形状的随机张量
    def rand_like(self, v):
        return torch.rand_like(v)

    # 将张量转换为 numpy 数组
    def numpy(self, t):
        return t.cpu().numpy()

    # 返回两个张量的按元素乘积
    def mul(self, t1, t2):
        return t1 * t2

    # 返回两个张量的按元素加法结果
    def add(self, t1, t2):
        return t1 + t2

    # 执行批归一化操作
    def batch_norm(self, data, mean, var, training):
        return torch.nn.functional.batch_norm(data, mean, var, training=training)

    # 执行实例归一化操作
    def instance_norm(self, data):
        return torch.nn.functional.instance_norm(data)

    # 执行层归一化操作
    def layer_norm(self, data, shape):
        return torch.nn.functional.layer_norm(data, shape)

    # 同步 CUDA 设备上的操作
    def sync_cuda(self):
        torch.cuda.synchronize()

    # 执行反向传播
    def backward(self, tensors, grad_tensors, _):
        torch.autograd.backward(tensors, grad_tensors=grad_tensors)

    # 沿指定维度对张量进行求和
    def sum(self, data, dims):
        return torch.sum(data, dims)

    # 执行 softmax 操作
    def softmax(self, data, dim=None, dtype=None):
        return torch.nn.functional.softmax(data, dim, dtype)

    # 沿指定维度拼接张量序列
    def cat(self, inputs, dim=0):
        return torch.cat(inputs, dim=dim)

    # 将张量限制在指定范围内
    def clamp(self, data, min, max):
        return torch.clamp(data, min=min, max=max)

    # 执行 ReLU 激活函数
    def relu(self, data):
        return torch.nn.functional.relu(data)

    # 执行 tanh 激活函数
    def tanh(self, data):
        return torch.tanh(data)

    # 执行 2D 最大池化操作
    def max_pool2d(self, data, kernel_size, stride=1):
        return torch.nn.functional.max_pool2d(data, kernel_size, stride=stride)

    # 执行 2D 平均池化操作
    def avg_pool2d(self, data, kernel_size, stride=1):
        return torch.nn.functional.avg_pool2d(data, kernel_size, stride=stride)

    # 创建 2D 卷积层
    def conv2d_layer(self, ic, oc, kernel_size, groups=1):
        return torch.nn.Conv2d(ic, oc, kernel_size, groups=groups)

    # 执行矩阵乘法
    def matmul(self, t1, t2):
        return torch.matmul(t1, t2)

    # 将模块移动到指定设备
    def to_device(self, module, device):
        return module.to(device)
```