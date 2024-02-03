# `stable-diffusion-webui\extensions-builtin\Lora\network_lora.py`

```
# 导入 torch 库
import torch

# 导入 lyco_helpers 模块
import lyco_helpers

# 导入 network 模块
import network

# 从 modules 模块中导入 devices
from modules import devices

# 定义一个继承自 network.ModuleType 的类 ModuleTypeLora
class ModuleTypeLora(network.ModuleType):
    
    # 创建模块的方法
    def create_module(self, net: network.Network, weights: network.NetworkWeights):
        
        # 如果权重中包含 "lora_up.weight" 和 "lora_down.weight"，则返回 NetworkModuleLora 对象
        if all(x in weights.w for x in ["lora_up.weight", "lora_down.weight"]):
            return NetworkModuleLora(net, weights)

        return None

# 定义一个继承自 network.NetworkModule 的类 NetworkModuleLora
class NetworkModuleLora(network.NetworkModule):
    
    # 初始化方法
    def __init__(self,  net: network.Network, weights: network.NetworkWeights):
        
        # 调用父类的初始化方法
        super().__init__(net, weights)

        # 创建上行模型
        self.up_model = self.create_module(weights.w, "lora_up.weight")
        
        # 创建下行模型
        self.down_model = self.create_module(weights.w, "lora_down.weight")
        
        # 创建中间模型，如果权重中不存在 "lora_mid.weight"，则设置为 None
        self.mid_model = self.create_module(weights.w, "lora_mid.weight", none_ok=True)

        # 获取权重 "lora_down.weight" 的维度作为模块的维度
        self.dim = weights.w["lora_down.weight"].shape[0]
    # 创建一个模块，根据给定的权重和键值，如果允许权重为None，则返回None
    def create_module(self, weights, key, none_ok=False):
        # 从权重字典中获取对应键值的权重
        weight = weights.get(key)

        # 如果权重为None且允许为None，则返回None
        if weight is None and none_ok:
            return None

        # 判断当前模块是否为线性模块
        is_linear = type(self.sd_module) in [torch.nn.Linear, torch.nn.modules.linear.NonDynamicallyQuantizableLinear, torch.nn.MultiheadAttention]
        # 判断当前模块是否为卷积模块
        is_conv = type(self.sd_module) in [torch.nn.Conv2d]

        # 如果是线性模块
        if is_linear:
            # 重塑权重的形状
            weight = weight.reshape(weight.shape[0], -1)
            # 创建一个线性模块
            module = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        # 如果是卷积模块并且键值为"lora_down.weight"或"dyn_up"
        elif is_conv and key == "lora_down.weight" or key == "dyn_up":
            # 如果权重的维度为2，则重塑形状
            if len(weight.shape) == 2:
                weight = weight.reshape(weight.shape[0], -1, 1, 1)

            # 如果权重的高度和宽度不为1，则创建一个卷积模块
            if weight.shape[2] != 1 or weight.shape[3] != 1:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
            else:
                module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        # 如果是卷积模块并且键值为"lora_mid.weight"
        elif is_conv and key == "lora_mid.weight":
            # 创建一个卷积模块
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], self.sd_module.kernel_size, self.sd_module.stride, self.sd_module.padding, bias=False)
        # 如果是卷积模块并且键值为"lora_up.weight"或"dyn_down"
        elif is_conv and key == "lora_up.weight" or key == "dyn_down":
            # 创建一个卷积模块
            module = torch.nn.Conv2d(weight.shape[1], weight.shape[0], (1, 1), bias=False)
        else:
            # 抛出异常，表示不支持的模块类型
            raise AssertionError(f'Lora layer {self.network_key} matched a layer with unsupported type: {type(self.sd_module).__name__}')

        # 使用torch.no_grad()上下文管理器，确保不会计算梯度
        with torch.no_grad():
            # 如果权重的形状与模块的权重形状不一致，则重塑权重的形状
            if weight.shape != module.weight.shape:
                weight = weight.reshape(module.weight.shape)
            # 将权重复制到模块的权重中
            module.weight.copy_(weight)

        # 将模块移动到指定设备上，并设置数据类型
        module.to(device=devices.cpu, dtype=devices.dtype)
        # 设置模块的权重不需要梯度计算
        module.weight.requires_grad_(False)

        # 返回创建的模块
        return module
    # 计算上采样和下采样的权重
    def calc_updown(self, orig_weight):
        # 将上采样模型的权重转移到与原始权重相同的设备和数据类型
        up = self.up_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
        # 将下采样模型的权重转移到与原始权重相同的设备和数据类型
        down = self.down_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)

        # 初始化输出形状为上采样和下采样权重的维度
        output_shape = [up.size(0), down.size(1)]
        if self.mid_model is not None:
            # 如果存在中间模型，则进行cp-decomposition
            mid = self.mid_model.weight.to(orig_weight.device, dtype=orig_weight.dtype)
            # 重建cp-decomposition，得到上采样和下采样的权重
            updown = lyco_helpers.rebuild_cp_decomposition(up, down, mid)
            # 更新输出形状，加上中间模型的维度
            output_shape += mid.shape[2:]
        else:
            if len(down.shape) == 4:
                # 如果下采样权重的维度为4，则更新输出形状，加上下采样权重的维度
                output_shape += down.shape[2:]
            # 重建传统的上采样和下采样权重
            updown = lyco_helpers.rebuild_conventional(up, down, output_shape, self.network.dyn_dim)

        # 返回最终的上采样和下采样权重
        return self.finalize_updown(updown, orig_weight, output_shape)

    # 前向传播函数
    def forward(self, x, y):
        # 将上采样模型和下采样模型转移到指定设备
        self.up_model.to(device=devices.device)
        self.down_model.to(device=devices.device)

        # 返回y加上上采样模型作用于下采样模型作用于x的结果，乘以乘数和缩放系数
        return y + self.up_model(self.down_model(x)) * self.multiplier() * self.calc_scale()
```