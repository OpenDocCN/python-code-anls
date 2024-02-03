# `stable-diffusion-webui\extensions-builtin\Lora\network_lokr.py`

```
# 导入 torch 库
import torch

# 导入 lyco_helpers 模块
import lyco_helpers

# 导入 network 模块
import network

# 定义 ModuleTypeLokr 类，继承自 network.ModuleType
class ModuleTypeLokr(network.ModuleType):
    # 创建模块的方法，接受网络和权重作为参数
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        # 检查权重中是否包含 lokr_w1 或 lokr_w1_a 和 lokr_w1_b
        has_1 = "lokr_w1" in weights.w or ("lokr_w1_a" in weights.w and "lokr_w1_b" in weights.w)
        # 检查权重中是否包含 lokr_w2 或 lokr_w2_a 和 lokr_w2_b
        has_2 = "lokr_w2" in weights.w or ("lokr_w2_a" in weights.w and "lokr_w2_b" in weights.w)
        # 如果都包含，则返回 NetworkModuleLokr 对象，否则返回 None
        if has_1 and has_2:
            return NetworkModuleLokr(net, weights)

        return None

# 定义 make_kron 函数，接受原始形状、w1 和 w2 作为参数
def make_kron(orig_shape, w1, w2):
    # 如果 w2 的维度为 4，则对 w1 进行维度扩展
    if len(w2.shape) == 4:
        w1 = w1.unsqueeze(2).unsqueeze(2)
    # 使 w2 连续
    w2 = w2.contiguous()
    # 返回 w1 和 w2 的 Kronecker 乘积，并重塑为原始形状
    return torch.kron(w1, w2).reshape(orig_shape)

# 定义 NetworkModuleLokr 类，继承自 network.NetworkModule
class NetworkModuleLokr(network.NetworkModule):
    # 初始化方法，接受网络和权重作为参数
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 获取权重中的 lokr_w1、lokr_w1_a、lokr_w1_b
        self.w1 = weights.w.get("lokr_w1")
        self.w1a = weights.w.get("lokr_w1_a")
        self.w1b = weights.w.get("lokr_w1_b")
        # 如果 lokr_w1_b 存在，则将其维度赋给 dim
        self.dim = self.w1b.shape[0] if self.w1b is not None else self.dim
        # 获取权重中的 lokr_w2、lokr_w2_a、lokr_w2_b
        self.w2 = weights.w.get("lokr_w2")
        self.w2a = weights.w.get("lokr_w2_a")
        self.w2b = weights.w.get("lokr_w2_b")
        # 如果 lokr_w2_b 存在，则将其维度赋给 dim
        self.dim = self.w2b.shape[0] if self.w2b is not None else self.dim
        # 获取权重中的 lokr_t2
        self.t2 = weights.w.get("lokr_t2")
    # 计算上采样和下采样的权重矩阵
    def calc_updown(self, orig_weight):
        # 如果存在 self.w1，则将其转换为与 orig_weight 相同设备和数据类型的张量
        if self.w1 is not None:
            w1 = self.w1.to(orig_weight.device, dtype=orig_weight.dtype)
        else:
            # 否则将 self.w1a 和 self.w1b 转换为与 orig_weight 相同设备和数据类型的张量，并计算矩阵乘积
            w1a = self.w1a.to(orig_weight.device, dtype=orig_weight.dtype)
            w1b = self.w1b.to(orig_weight.device, dtype=orig_weight.dtype)
            w1 = w1a @ w1b

        # 如果存在 self.w2，则将其转换为与 orig_weight 相同设备和数据类型的张量
        if self.w2 is not None:
            w2 = self.w2.to(orig_weight.device, dtype=orig_weight.dtype)
        # 如果不存在 self.w2 且 self.t2 为 None，则将 self.w2a 和 self.w2b 转换为与 orig_weight 相同设备和数据类型的张量，并计算矩阵乘积
        elif self.t2 is None:
            w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = w2a @ w2b
        else:
            # 否则将 self.t2、self.w2a 和 self.w2b 转换为与 orig_weight 相同设备和数据类型的张量，并调用 make_weight_cp 函数
            t2 = self.t2.to(orig_weight.device, dtype=orig_weight.dtype)
            w2a = self.w2a.to(orig_weight.device, dtype=orig_weight.dtype)
            w2b = self.w2b.to(orig_weight.device, dtype=orig_weight.dtype)
            w2 = lyco_helpers.make_weight_cp(t2, w2a, w2b)

        # 计算输出形状
        output_shape = [w1.size(0) * w2.size(0), w1.size(1) * w2.size(1)]
        # 如果 orig_weight 的维度为 4，则输出形状与 orig_weight 的形状相同
        if len(orig_weight.shape) == 4:
            output_shape = orig_weight.shape

        # 调用 make_kron 函数生成上采样和下采样的权重矩阵
        updown = make_kron(output_shape, w1, w2)

        # 调用 finalize_updown 函数对上采样和下采样的权重矩阵进行最终处理并返回结果
        return self.finalize_updown(updown, orig_weight, output_shape)
```